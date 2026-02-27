import os
import torch
import json
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

def scan_predictions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("tokenizer_oran")
    model = GPT2ForSequenceClassification.from_pretrained("final_gpt2_5g_classifier").to(device)
    
    print(f"Scanning samples for INTEROPERATING prediction...")
    
    for i in range(50):
        path = f"dataset_output_5k/{i}/capture.txt"
        meta_path = f"dataset_output_5k/{i}/metadata.json"
        
        if not os.path.exists(path): continue
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        if meta.get("family") != "INTEROPING": continue
            
        # Parse (simplified)
        with open(path, 'r') as f: text = f.read()
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            
        pred_idx = torch.argmax(logits).item()
        pred_label = model.config.id2label[pred_idx]
        
        print(f"Sample {i}: GT={meta['family']} Pred={pred_label} (Logit={logits[pred_idx]:.2f})")
        
        if pred_label == "INTEROPERATING":
            print(f"FOUND! Sample {i} is INTEROPERATING.")
            break

if __name__ == "__main__":
    scan_predictions()
