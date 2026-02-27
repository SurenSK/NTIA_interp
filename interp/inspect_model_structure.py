
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
from transformer_lens import HookedTransformer

MODEL_PATH = "final_gpt2_5g_classifier"
TOKENIZER_PATH = "tokenizer_oran"

def inspect_model():
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
    
    print(f"Loading model from {MODEL_PATH}...")
    device = "cpu" # sufficient for inspection
    
    hf_model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)
    lm_model = GPT2LMHeadModel(hf_model.config)
    lm_model.transformer.load_state_dict(hf_model.transformer.state_dict())
    
    print("Loading HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        "gpt2", 
        hf_model=lm_model, 
        tokenizer=tokenizer,
        device=device
    )
    
    print("\n--- Model Inspection ---")
    print(f"Model type: {type(model)}")
    print(f"ln_final type: {type(model.ln_final)}")
    print(f"ln_final keys: {model.ln_final.state_dict().keys()}")
    
    if hasattr(model.ln_final, 'w'):
        print(f"ln_final.w shape: {model.ln_final.w.shape}")
    else:
        print("ln_final.w not found")
        
    if hasattr(model.ln_final, 'b'):
        print(f"ln_final.b shape: {model.ln_final.b.shape}")
    else:
        print("ln_final.b not found")

    # Check if weights are Identity or actual values
    if hasattr(model.ln_final, 'w'):
        print(f"ln_final.w sample: {model.ln_final.w[:5]}")
        
    print("\n--- Unembed / Score Inspection ---")
    # In HookedTransformer, the unembedding is usually model.unembed
    print(f"model.unembed type: {type(model.unembed)}")
    if hasattr(model.unembed, 'W_U'):
         print(f"model.unembed.W_U shape: {model.unembed.W_U.shape}")
    
    # We are using hf_model.score for classification in the original script.
    # Let's see if we can locate that weight in HookedTransformer or if we must use external.
    print(f"hf_model.score weight shape: {hf_model.score.weight.shape}")

if __name__ == "__main__":
    inspect_model()
