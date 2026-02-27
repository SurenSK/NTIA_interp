import os
import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
from transformer_lens import HookedTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_gpt2_5g_classifier")
    parser.add_argument("--tokenizer", default="tokenizer_oran")
    parser.add_argument("--input", default="dataset_output_5k/10/capture.txt")
    parser.add_argument("--output", default="interp_results")
    return parser.parse_args()

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

    if not packets: raise ValueError("No packets parsed")
    dur = (t1 - t0).total_seconds() if t1 and t0 else 0
    return f"[LEN={len(packets)}] [DUR={dur:.2f}s] " + " ".join(packets)

def analyze(args, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    hf_model = GPT2ForSequenceClassification.from_pretrained(args.model).to(device)
    
    # Wrap in LM for HookedTransformer compatibility
    lm = GPT2LMHeadModel(hf_model.config).to(device)
    lm.transformer.load_state_dict(hf_model.transformer.state_dict())
    model = HookedTransformer.from_pretrained("gpt2", hf_model=lm, tokenizer=tokenizer, device=device, fold_ln=False, center_unembed=False, use_attn_result=True)
    
    tokens = tokenizer(text, return_tensors="pt").to(device)
    logits, cache = model.run_with_cache(tokens["input_ids"])
    
    # Predictions
    hf_logits = hf_model(**tokens).logits[0]
    pred_idx = torch.argmax(hf_logits).item()
    top3 = torch.topk(hf_logits, 3).indices.tolist()
    labels = hf_model.config.id2label
    print(f"Prediction: {labels[pred_idx]} ({hf_logits[pred_idx]:.2f})")

    # Logit Lens
    resids = list(cache.items()) 
    layer_logits = [hf_model.score(model.ln_final(cache['blocks.0.hook_resid_pre'][0, -1])).detach().cpu().numpy()]
    for i in range(model.cfg.n_layers):
        r = model.ln_final(cache[f'blocks.{i}.hook_resid_post'][0, -1])
        layer_logits.append(hf_model.score(r).detach().cpu().numpy())
    
    L = np.array(layer_logits)
    plt.figure(figsize=(10, 6))
    for i in top3: plt.plot(L[:, i], label=labels[i])
    plt.legend(); plt.grid(True, alpha=0.3); plt.savefig(os.path.join(args.output, "logit_lens.png"))

    # Direct Logit Attribution
    final_resid = cache[f'blocks.{model.cfg.n_layers-1}.hook_resid_post'][0, -1]
    scale = model.ln_final.w / torch.sqrt(final_resid.var(dim=-1, keepdim=True) + model.cfg.eps).squeeze()
    
    for cls_idx in top3:
        W = hf_model.score.weight[cls_idx].detach()
        start = (scale * W).detach()
        hm = np.empty((model.cfg.n_layers, model.cfg.n_heads))
        for l in range(model.cfg.n_layers):
            block_res = cache[f'blocks.{l}.attn.hook_result'][0, -1]
            hm[l] = torch.einsum('hd,d->h', block_res, start).cpu().numpy()
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(hm, cmap="RdBu_r", center=0); plt.title(f"Head Attr: {labels[cls_idx]}")
        plt.savefig(os.path.join(args.output, f"head_map_{labels[cls_idx]}.png"))

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    analyze(args, parse_capture(args.input))
