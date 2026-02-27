
import os
import torch
import json
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

def find_samples():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_oran")
    model = GPT2ForSequenceClassification.from_pretrained("final_gpt2_5g_classifier").to(device)
    
    targets = ["PROTOCOL", "TRANSPORT"]
    found = {}
    
    print(f"Scanning samples for {targets}...")
    
    for i in range(100): # Scan up to 100 samples
        if len(found) == len(targets): break
        
        path = f"dataset_output_5k/{i}/capture.txt"
        
        if not os.path.exists(path): continue
        
        # Read content
        with open(path, 'r') as f: text = f.read()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            
        pred_idx = torch.argmax(logits).item()
        pred_label = model.config.id2label[pred_idx]
        
        if pred_label in targets and pred_label not in found:
            print(f"FOUND: Sample {i} -> {pred_label} (Logit={logits[pred_idx]:.2f})")
            found[pred_label] = path

    print("\nResults:")
    for label, path in found.items():
        print(f"{label}: {path}")

if __name__ == "__main__":
    find_samples()
