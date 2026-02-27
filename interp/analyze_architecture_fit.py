import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
from transformer_lens import HookedTransformer
import pandas as pd
import re
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def parse_capture(path):
    with open(path, 'r') as f: lines = f.readlines()
    packets, t0, t1 = [], None, None
    ip_map = {"1": "DU", "11": "CU", "5": "CORE"}
    chunk_map = {
        "INIT ACK": "INIT_ACK", "COOKIE ECHO": "COOKIE_ECHO", "COOKIE ACK": "COOKIE_ACK",
        "HEARTBEAT": "HB", "DATA": "DATA", "SACK": "SACK", "SHUTDOWN": "SHUTDOWN", "INIT": "INIT"
    }

    for line in lines:
        line = line.strip()
        if not line: continue
        try:
            parts = line.split()
            ts = datetime.strptime(parts[0], "%H:%M:%S.%f")
            t0, t1 = (t0 or ts), ts
            
            if len(parts) < 5 or parts[1] != 'IP': continue
            src, dst = parts[2], parts[4].rstrip(':')
            src_suf = src.split('.')[3] if len(src.split('.')) >= 5 else src.split('.')[-1]
            dst_suf = dst.split('.')[3] if len(dst.split('.')) >= 5 else dst.split('.')[-1]
            
            chunks = []
            for c in re.findall(r'\[([A-Z ]+)\]', line):
                c = c.strip()
                if c in ["OS", "MIS"]: continue
                chunks.append(chunk_map.get(c, "HB" if "HEARTBEAT" in c else "ABORT" if "ABORT" in c else "ERROR" if "ERROR" in c else "MSG"))
            
            if chunks:
                packets.append(f"{ip_map.get(src_suf, src_suf)}>{dst_suf}:{'+'.join(chunks)}")
        except Exception: continue

    if not packets: return ""
    dur = (t1 - t0).total_seconds() if t1 and t0 else 0
    return f"[LEN={len(packets)}] [DUR={dur:.2f}s] " + " ".join(packets)

def analyze_full_layer_components(model, cache, target_idx, hf_score):
    print("\n--- Full Component Utility (Linearized Approximation) ---")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # 1. Get Gradient Direction (Linearized LayerNorm)
    final_resid = cache[f'blocks.{n_layers-1}.hook_resid_post'][:, -1, :] # [batch, d_model]
    
    # LN Scale: w / std
    ln_scale = model.ln_final.w / torch.sqrt(final_resid.var(dim=-1, keepdim=True) + model.cfg.eps)
    
    # Unembed Direction: ln_scale * W_U[target]
    W_U = hf_score.weight[target_idx] # [d_model]
    direction = ln_scale * W_U # [batch, d_model] (broadcasting W_U)
    
    results = []

    # 2. Embeddings
    # [batch, d_model]
    embeds = cache['blocks.0.hook_resid_pre'][:, -1, :] 
    attr_embed = (embeds * direction).sum(dim=-1).mean().item()
    results.append({"Component": "Embeddings", "DLA": attr_embed})

    # 3. Heads
    head_stack = cache.stack_head_results(layer=-1, pos_slice=-1) 
    # Shape: [n_layers, n_heads, batch, d_model]
    # Handle potentially missing batch dim if 3D
    if len(head_stack.shape) == 3:
        batch = 1
        d_model = head_stack.shape[-1]
    else:
        batch = head_stack.shape[1]
        d_model = head_stack.shape[-1]

    # Reshape to [n_layers, n_heads, batch, d_model]
    head_stack = head_stack.reshape(n_layers, n_heads, batch, d_model)
    
    # Attribution: (Head * Direction).sum(-1)
    # direction broadcast to [1, 1, batch, d_model]
    attr_heads = (head_stack * direction.reshape(1, 1, -1, direction.shape[-1])).sum(dim=-1)
    attr_heads = attr_heads.mean(dim=-1) # Average over batch -> [n_layers, n_heads]

    for l in range(n_layers):
        for h in range(n_heads):
            results.append({"Component": f"L{l}H{h}", "DLA": attr_heads[l, h].item()})

    # 4. MLPs
    mlp_stack = cache.decompose_resid(layer=-1, mode="mlp", pos_slice=-1)
    # Shape: [n_layers, batch, d_model]
    if len(mlp_stack.shape) == 2:
         mlp_stack = mlp_stack.unsqueeze(1)
         
    if mlp_stack.shape[0] > n_layers:
        mlp_stack = mlp_stack[-n_layers:]
    
    # Attribution
    # direction broadcast to [1, batch, d_model]
    if len(mlp_stack.shape) == 3:
        attr_mlps = (mlp_stack * direction.unsqueeze(0)).sum(dim=-1).mean(dim=-1)
    else:
        # Unexpected shape, try reshape
        attr_mlps = (mlp_stack * direction.reshape(1, 1, -1, direction.shape[-1])).sum(dim=-1).mean(dim=-1).squeeze()

    for l in range(n_layers):
        results.append({"Component": f"L{l}_MLP", "DLA": attr_mlps[l].item()})

    # 5. Display
    df = pd.DataFrame(results)
    print(df.sort_values("DLA", ascending=False).head(20).to_string(index=False))
    return df

def generate_component_heatmaps(model, cache, hf_score):
    print("\n--- Generating Component Heatmaps (Linearized) ---")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    target_classes = {
        2: "INTEROPERATING",
        3: "PROTOCOL",
        5: "TRANSPORT"
    }
    
    # 1. Get Common Data
    final_resid = cache[f'blocks.{n_layers-1}.hook_resid_post'][:, -1, :]
    ln_scale = model.ln_final.w / torch.sqrt(final_resid.var(dim=-1, keepdim=True) + model.cfg.eps)
    
    # Stacks
    embeds = cache['blocks.0.hook_resid_pre'][:, -1, :] # [batch, d_model]
    
    head_stack = cache.stack_head_results(layer=-1, pos_slice=-1) 
    if len(head_stack.shape) == 3:
        batch = 1
        d_model = head_stack.shape[-1]
    else:
        batch = head_stack.shape[1]
        d_model = head_stack.shape[-1]
    head_stack = head_stack.reshape(n_layers, n_heads, batch, d_model)
    
    mlp_stack = cache.decompose_resid(layer=-1, mode="mlp", pos_slice=-1)
    if len(mlp_stack.shape) == 2: mlp_stack = mlp_stack.unsqueeze(1)
    if mlp_stack.shape[0] > n_layers: mlp_stack = mlp_stack[-n_layers:]
    
    # 2. Project to Logits (just used for actual logit check)
    accumulated = cache.accumulated_resid(layer=-1, pos_slice=-1)
    ln_accumulated = cache.apply_ln_to_stack(accumulated, layer=-1, pos_slice=-1)
    component_logits = hf_score(ln_accumulated)

    for class_idx, class_name in target_classes.items():
        print(f"Generating heatmap for {class_name}...")
        
        # Direction
        W_U = hf_score.weight[class_idx]
        direction = ln_scale * W_U # [batch, d_model]
        
        # Calculate Attributions (average over batch)
        # Embeds
        attr_embed = (embeds * direction).sum(dim=-1).mean().item()
        
        # Heads
        attr_heads = (head_stack * direction.reshape(1, 1, -1, direction.shape[-1])).sum(dim=-1).mean(dim=-1).detach().cpu().numpy()
        
        # MLPs
        if len(mlp_stack.shape) == 3:
             attr_mlps = (mlp_stack * direction.unsqueeze(0)).sum(dim=-1).mean(dim=-1).detach().cpu().numpy()
        else:
             attr_mlps = (mlp_stack * direction.reshape(1, 1, -1, direction.shape[-1])).sum(dim=-1).mean(dim=-1).squeeze().detach().cpu().numpy()
        
        attr_heads_sum = attr_heads.sum()
        attr_mlps_sum = attr_mlps.sum()
        total_attr = attr_embed + attr_heads_sum + attr_mlps_sum
        
        # Bias
        if hf_score.bias is not None:
            bias = hf_score.bias[class_idx].item()
        else:
            bias = 0.0
            
        print(f"Stats: {class_name} | Embed: {attr_embed:.4f} | Heads: {attr_heads_sum:.4f} | MLPs: {attr_mlps_sum:.4f}")
        print(f"Stats: {class_name} | Total Attr (Linearized): {total_attr:.4f} | Bias: {bias:.4f} | Sum+Bias: {total_attr + bias:.4f}")
        
        # Check actual logit
        avg_logit = component_logits[-1, 0, class_idx].item() # Last layer
        print(f"Stats: {class_name} | Actual Final Logit: {avg_logit:.4f}")
    
        # Matrix: [n_layers, n_heads + 1]
        matrix = np.zeros((n_layers, n_heads + 1))
        matrix[:, :n_heads] = attr_heads
        matrix[:, n_heads] = attr_mlps

        # Plot
        cols = [f"H{h}" for h in range(n_heads)] + ["MLP"]
        dla_df = pd.DataFrame(matrix, columns=cols)
        dla_df.index.name = "Layer"
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(dla_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        
        # Add Embedding info to title
        plt.title(f"Attr: {class_name} | Embed: {attr_embed:.2f} | Bias: {bias:.2f}")
        plt.tight_layout()
        output_filename = f"heatmap_{class_name}.png"
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved {output_filename}")

def analyze_fit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_path = "final_gpt2_5g_classifier"
    tokenizer_path = "tokenizer_oran"
    # input_file = "dataset_output_5k/10/capture.txt" # Misclassified
    input_file = "dataset_output_5k/8/capture.txt" # Correctly classified as INTEROPERATING
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    hf_model = GPT2ForSequenceClassification.from_pretrained(model_path).to(device)
    
    lm = GPT2LMHeadModel(hf_model.config).to(device)
    lm.transformer.load_state_dict(hf_model.transformer.state_dict())
    model = HookedTransformer.from_pretrained("gpt2", hf_model=lm, tokenizer=tokenizer, device=device, fold_ln=False, center_unembed=False, use_attn_result=True)
    
    # data = parse_capture(input_file)
    # Using raw file content as it matches training distribution better
    with open(input_file, 'r') as f:
        data = f.read()
        
    if not data:
        print("Failed to read capture")
        return

    tokens = tokenizer(data, return_tensors="pt").to(device)
    logits, cache = model.run_with_cache(tokens["input_ids"])
    
    hf_logits = hf_model(**tokens).logits[0]
    pred_idx = torch.argmax(hf_logits).item()
    pred_label = hf_model.config.id2label[pred_idx]
    print(f"\nPrediction: {pred_label} (Logit: {hf_logits[pred_idx]:.2f})")
    
    # --- 1. Layer Utility (Accumulated) ---
    print("\n--- Layer Utility (Logit Accumulation) ---")
    accumulated, labels = cache.accumulated_resid(layer=-1, pos_slice=-1, return_labels=True)
    ln_accumulated = cache.apply_ln_to_stack(accumulated, layer=-1, pos_slice=-1)
    
    layer_logits = hf_model.score(ln_accumulated)
    target_logits = layer_logits[:, 0, pred_idx].detach().cpu().numpy()

    print(f"{'Layer':<20} {'Logit':<10} {'Gain':<10}")
    print("-" * 40)
    gains = np.diff(target_logits, prepend=target_logits[0])
    for label, score, gain in zip(labels, target_logits, gains):
        print(f"{label:<20} {score:.2f}      {gain:+.2f}")

    # --- 2. Head Utility (Specific) ---
    try:
        # Clear previous summary
        if os.path.exists("attribution_summary.txt"):
            os.remove("attribution_summary.txt")
            
        analyze_full_layer_components(model, cache, pred_idx, hf_model.score)
        generate_component_heatmaps(model, cache, hf_model.score)
    except Exception as e:
        print(f"Head Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        analyze_fit()
    except Exception as e:
        print(f"Error: {e}")