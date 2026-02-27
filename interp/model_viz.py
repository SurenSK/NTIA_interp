import torch
import glob
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm

# --- CONFIG ---
MODEL_PATH = "model_nano_interop" 
TOKENIZER_PATH = "tokenizer_oran"
DATA_DIR = "dataset_output_5k"
NUM_SAMPLES = 5000  # Adjust for speed vs accuracy

def run_viz():
    # 1. Load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)

    # 2. Define Targets
    # Updated Viz List
    families = ["WHISPER", "STUTTER", "GHOST", "MUTE", "BINGE", "SQUEEZE", "INTEROPING"]
    # Map friendly name -> Token ID
    fam_map = {f: tokenizer.encode(f"[FAM_{f}]")[0] for f in families}
    # Reverse map for plotting
    id_to_name = {v: k for k, v in fam_map.items()}
    sorted_ids = [fam_map[f] for f in families]

    # 3. Initialize Matrix: [GT_Classes, Pred_Classes + 1 (Other)]
    # We accumulate probabilities here
    conf_matrix = np.zeros((len(families), len(families) + 1))
    class_counts = np.zeros(len(families))

    # 4. Gather Data
    files = glob.glob(os.path.join(DATA_DIR, "*"))[:NUM_SAMPLES]
    print(f"Processing {len(files)} samples...")

    for d in tqdm(files):
        try:
            # Read GT
            with open(os.path.join(d, "metadata.json"), "r") as f:
                meta = json.load(f)
            
            # Determine GT Index
            gt_fam = next((f for f in families if f in meta['family'].upper()), None)
            if not gt_fam: continue
            gt_idx = families.index(gt_fam)

            # Read PCAP & Prompt
            with open(os.path.join(d, "capture.txt"), "r") as f:
                pcap = f.read().strip()
            
            # Prompt ends with space to invite the token
            prompt = f"[START_PCAP]\n{pcap}\n[END_PCAP]\n[START_RESULT]"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Inference
            with torch.no_grad():
                logits = model(**inputs).logits
                # Get logits of the LAST token
                last_token_logits = logits[0, -1, :]
                probs = torch.softmax(last_token_logits, dim=0).cpu().numpy()

            # 5. Accumulate Probabilities
            row_probs = []
            total_target_prob = 0
            
            for fid in sorted_ids:
                p = probs[fid]
                row_probs.append(p)
                total_target_prob += p
            
            # The "Waste" Bin (Everything else)
            other_prob = 1.0 - total_target_prob
            row_probs.append(other_prob)

            # Add to Matrix
            conf_matrix[gt_idx] += np.array(row_probs)
            class_counts[gt_idx] += 1
            
        except Exception as e:
            continue

    # 6. Normalize (Average Probability per Class)
    # Avoid div by zero
    class_counts[class_counts == 0] = 1
    conf_matrix = conf_matrix / class_counts[:, None]

    # 7. Visualize
    plt.figure(figsize=(10, 8))
    x_labels = families + ["OTHER (Waste)"]
    y_labels = families
    
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=x_labels, 
        yticklabels=y_labels
    )
    plt.title(f"Soft Confusion Matrix (Avg Probabilities) - N={len(files)}")
    plt.ylabel("Ground Truth Family")
    plt.xlabel("Model Probability Distribution")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_viz()