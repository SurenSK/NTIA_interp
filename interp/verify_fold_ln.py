
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
from transformer_lens import HookedTransformer

MODEL_PATH = "final_gpt2_5g_classifier"
TOKENIZER_PATH = "tokenizer_oran"

def verify():
    print("Loading HF model...")
    hf_model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)
    lm_model = GPT2LMHeadModel(hf_model.config)
    lm_model.transformer.load_state_dict(hf_model.transformer.state_dict())
    
    print("Loading HookedTransformer with fold_ln=False...")
    model = HookedTransformer.from_pretrained(
        "gpt2", 
        hf_model=lm_model, 
        tokenizer=None,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        device="cpu"
    )
    
    print(f"ln_final type: {type(model.ln_final)}")
    if hasattr(model.ln_final, 'w'):
        print(f"ln_final.w shape: {model.ln_final.w.shape}")
        print(f"ln_final.w sample: {model.ln_final.w[:5]}")
    else:
        print("ln_final.w STILL not found")
        
    if hasattr(model.ln_final, 'b'):
        print(f"ln_final.b shape: {model.ln_final.b.shape}")

if __name__ == "__main__":
    verify()
