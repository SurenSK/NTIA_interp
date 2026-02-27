
import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
from transformer_lens import HookedTransformer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_html_visualization(tokens, scores, filename="token_importance.html"):
    """
    Generates an HTML file with tokens highlighted based on their scores.
    Args:
        tokens: List of token strings.
        scores: List of scores corresponding to tokens.
        filename: Output filename.
    """
    # Normalize scores to 0-1 range for opacity
    max_score = max(scores)
    if max_score > 0:
        normalized_scores = [max(0, s) / max_score for s in scores]
    else:
        normalized_scores = [0] * len(scores)

    html_content = """
    <html>
    <head>
        <style>
            body { font-family: monospace; line-height: 1.5; padding: 20px; }
            .token { display: inline-block; padding: 2px 0; }
            .tooltip { position: relative; display: inline-block; cursor: default; }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 120px;
                background-color: black;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px 0;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -60px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 12px;
            }
            .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        </style>
    </head>
    <body>
        <h2>weighted Attention Scores</h2>
        <p>Hover over tokens to see exact scores.</p>
        <div style="border: 1px solid #ccc; padding: 10px;">
    """

    for token, score, norm_score in zip(tokens, scores, normalized_scores):
        # Clean token for display
        display_token = token.replace('Ġ', ' ').replace('Ċ', '\\n')
        if display_token == '\\n':
            html_content += "<br>"
            continue
            
        # Color: Strong Blue (0, 0, 255) with varying opacity
        # Using rgba(100, 149, 237, alpha) - Cornflower Blue looks nice, or pure blue
        # User asked for "strong blue" for heavy attention.
        color = f"rgba(0, 0, 255, {norm_score:.2f})"
        
        # Text color: white if background is dark, black otherwise
        text_color = "white" if norm_score > 0.5 else "black"
        
        html_content += f"""
        <div class="tooltip token" style="background-color: {color}; color: {text_color};">
            {display_token}
            <span class="tooltiptext">Score: {score:.4f}</span>
        </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML visualization to {filename}")

def analyze_token_importance(input_file, output_prefix="token_importance"):
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_path = "final_gpt2_5g_classifier"
    tokenizer_path = "tokenizer_oran"
    
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    hf_model = GPT2ForSequenceClassification.from_pretrained(model_path).to(device)
    
    lm = GPT2LMHeadModel(hf_model.config).to(device)
    lm.transformer.load_state_dict(hf_model.transformer.state_dict())
    model = HookedTransformer.from_pretrained("gpt2", hf_model=lm, tokenizer=tokenizer, device=device, fold_ln=False, center_unembed=False, use_attn_result=True)
    
    with open(input_file, 'r') as f:
        data = f.read()
        
    # Truncate to 1024 to avoid positional embedding errors
    tokens = tokenizer(data, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_ids = tokens["input_ids"]
    token_str = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # 2. Forward Pass
    logits, cache = model.run_with_cache(input_ids)
    
    hf_logits = hf_model(**tokens).logits[0]
    pred_idx = torch.argmax(hf_logits).item()
    pred_label = hf_model.config.id2label[pred_idx]
    print(f"\nPrediction: {pred_label} (Logit: {hf_logits[pred_idx]:.2f})")

    # 3. Calculate Head DLA
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    final_resid = cache[f'blocks.{n_layers-1}.hook_resid_post'][:, -1, :]
    ln_scale = model.ln_final.w / torch.sqrt(final_resid.var(dim=-1, keepdim=True) + model.cfg.eps)
    W_U = hf_model.score.weight[pred_idx]
    direction = ln_scale * W_U # [batch, d_model]
    
    # Get all head results using stack_head_results which handles extraction
    # layer=-1 implies all layers? No, usually stack_head_results returns for ONE layer if specified.
    # But analyze_architecture_fit.py suggests it returns [n_layers, n_heads...] if layer=-1?
    # Let's verify behavior. If not, we iterate.
    # Actually, let's use the exact pattern from analyze_architecture_fit.py (lines 73-84)
    # It seems to assume stack_head_results returns all layers if layer=-1.
    
    try:
        # returns [n_layers, n_heads, batch, d_model]
        head_stack = cache.stack_head_results(layer=-1, pos_slice=-1) 
    except Exception:
        # Fallback if layer=-1 doesn't return all
        head_stack = []
        for l in range(n_layers):
            head_stack.append(cache.stack_head_results(layer=l, pos_slice=-1))
        head_stack = torch.stack(head_stack)

    # Handle shape
    if len(head_stack.shape) == 3: # [n_heads, batch, d_model] ? No, expected 4D
        # Maybe n_layers is merged?
        # Let's reshape to be safe as per previous script
        head_stack = head_stack.reshape(n_layers, n_heads, head_stack.shape[-2], head_stack.shape[-1])
        
    # direction: [batch, d_model] -> [1, 1, batch, d_model]
    dla = (head_stack * direction.reshape(1, 1, -1, direction.shape[-1])).sum(dim=-1)
    head_dla = dla.mean(dim=-1) # [n_layers, n_heads]

    # 4. Aggregation (Top-K)
    TOP_K = 5
    print(f"Aggregating Attention (Top {TOP_K} Heads)...")
    
    # Identify Top-K heads
    flat_indices = torch.topk(head_dla.flatten(), TOP_K).indices
    top_heads = []
    for idx in flat_indices:
        l = (idx // n_heads).item()
        h = (idx % n_heads).item()
        top_heads.append((l, h))
        
    print(f"Top {TOP_K} Heads (L, H): {top_heads}")
    
    total_weighted_attn = torch.zeros(input_ids.shape[1], device=device)
    
    for l, h in top_heads:
        # Get attention pattern for this specific head
        # [batch, head, dest, src] -> [batch, src] (for dest=-1)
        pattern = cache[f'blocks.{l}.attn.hook_pattern'][:, h, -1, :]
        
        # Weight by DLA
        dla_val = head_dla[l, h]
        weighted_pattern = pattern * dla_val
        
        # Accumulate
        total_weighted_attn += weighted_pattern.mean(dim=0)

    # 5. Visualization / Output
    token_scores = total_weighted_attn.detach().cpu().numpy()
    
    # Create HTML Visualization
    generate_html_visualization(token_str, token_scores, filename=f"{output_prefix}.html")
    
    # Text Summary
    df = pd.DataFrame({
        "Token": token_str,
        "Score": token_scores,
        "Index": range(len(token_str))
    })
    
    print("\nTop 20 Most Important Tokens:")
    print(df.sort_values("Score", ascending=False).head(20).to_string(index=False))
    
    # Plot Bar Chart
    plt.figure(figsize=(15, 6))
    plt.bar(df["Index"], df["Score"], color='royalblue')
    plt.xlabel("Token Position")
    plt.ylabel(f"Weighted Attention Score (Top {TOP_K} Heads)")
    plt.title(f"Token Importance for Prediction: {pred_label} (Top {TOP_K} Heads)")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png")
    print(f"\nSaved {output_prefix}.png")
    
    # Plot Individual Head Patterns
    for i, (l, h) in enumerate(top_heads):
        pattern = cache[f'blocks.{l}.attn.hook_pattern'][0, h, -1, :].detach().cpu().numpy()
        dla_val = head_dla[l, h].item()
        
        plt.figure(figsize=(12, 4))
        plt.plot(pattern, label=f"L{l}H{h}", color='darkblue')
        plt.fill_between(range(len(pattern)), pattern, color='darkblue', alpha=0.3)
        plt.title(f"Attention Pattern - Head L{l}H{h} (DLA={dla_val:.2f})")
        plt.xlabel("Token Pos")
        plt.ylabel("Attn Weight")
        output_filename = f"{output_prefix}_attn_L{l}H{h}.png"
        plt.savefig(output_filename)
        print(f"Saved {output_filename}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset_output_5k/8/capture.txt", help="Input capture file")
    parser.add_argument("--output_prefix", default="token_importance", help="Prefix for output files")
    args = parser.parse_args()
    
    analyze_token_importance(args.input, args.output_prefix)
