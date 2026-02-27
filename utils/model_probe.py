import torch
import glob
import os
import random
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from transformer_lens import HookedTransformer, HookedTransformerConfig

# --- CONFIG ---
MODEL_PATH = "model_pico_oran_final"
TOKENIZER_PATH = "tokenizer_oran"
DATA_DIR = "dataset_output_5k"

def load_hooked_model():
    print("--- 1. Loading PicoGPT ---")
    hf_model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    cfg = HookedTransformerConfig(
        n_layers=2, d_model=128, n_ctx=2048, d_head=32, n_heads=4,
        d_vocab=len(tokenizer), act_fn="gelu_new", normalization_type="LN",
        d_mlp=512, use_attn_result=True
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(hf_model.state_dict(), strict=False)
    model.set_tokenizer(tokenizer)
    return model, tokenizer

def probe_force_feed():
    model, tokenizer = load_hooked_model()
    
    # 2. Hunt for Ghost
    files = glob.glob(os.path.join(DATA_DIR, "*", "capture.txt"))
    target_pcap = None
    for f in files:
        with open(f, 'r') as file:
            content = file.read()
            if "[ABORT]" in content:
                target_pcap = content.strip()
                print(f"\nFound Ghost Specimen: {f}")
                break
    
    if not target_pcap: return

    # 3. FORCE FEED PROMPT
    # We explicitly ADD the answer [FAM_GHOST] to the end.
    # We want to see how the model attends FROM this token.
    # Note: We rely on the tokenizer to handle the spacing around [FAM_GHOST]
    prompt = f"[START_PCAP]\n{target_pcap}\n[END_PCAP]\n[START_RESULT] [FAM_GHOST]"
    
    print("\n--- 2. Tokenization Debug ---")
    str_tokens = model.to_str_tokens(prompt)
    print(f"Last 5 tokens: {str_tokens[-5:]}")
    
    # Verify the last token is indeed the family
    if "GHOST" not in str_tokens[-1]:
        print("⚠️ CRITICAL: The last token is NOT [FAM_GHOST].")
        print("This means the tokenizer split it weirdly.")
        return

    # 4. Run Forward Pass
    print("\n--- 3. Running Force-Feed Pass ---")
    # We don't care about logits, just the cache
    _, cache = model.run_with_cache(prompt)
    
    # 5. Attention Analysis
    print("\n--- 4. Attention Analysis ---")
    
    # Find [ABORT] index
    abort_indices = [i for i, t in enumerate(str_tokens) if "ABORT" in t]
    if not abort_indices:
        print("❌ Could not locate [ABORT] token.")
        return
    target_idx = abort_indices[0] 
    
    print(f"Source: Last Token ('{str_tokens[-1]}')")
    print(f"Target: '{str_tokens[target_idx]}' at Index {target_idx}")
    
    # Check Layer 1 (Final Layer)
    layer = 1
    attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]
    
    print(f"\n\033[1;33m=== LAYER {layer} ATTENTION SCORES ===\033[0m")
    scores = []
    for h in range(4): 
        # Score: Head h, Last Position (-1), ABORT Position (target_idx)
        score = attn_pattern[h, -1, target_idx].item()
        scores.append((h, score))
        
        # Scale bar for visibility (x100 since scores are usually 0.0-1.0)
        bar_len = int(score * 50) 
        bar = "█" * bar_len
        print(f"Head 1.{h}: {score:.4f}  {bar}")

    best_head, best_score = max(scores, key=lambda x: x[1])
    
    print("\n------------------------------------------------")
    if best_score > 0.15:
        print(f"🚀 \033[1;32mCONFIRMED:\033[0m Head 1.{best_head} is the GHOST CIRCUIT.")
        print(f"It attended strongly ({best_score:.2f}) to the [ABORT] tag.")
    elif best_score > 0.05:
         print(f"⚠️ WEAK SIGNAL: Head 1.{best_head} is suspicious ({best_score:.2f}), but not dominant.")
    else:
        print("❌ NO SIGNAL: The model is not looking at [ABORT] directly.")
        print("It might be using an induction head in Layer 0 or a different mechanism.")

if __name__ == "__main__":
    probe_force_feed()