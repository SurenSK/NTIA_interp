import torch
import pandas as pd
import numpy as np
import sys
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- CONFIG ---
MODEL_PATH = "./final_gpt2_5g_classifier"
DATA_FILE = "traffic_logs.csv"
BATCH_SIZE = 32

# --- DATASET CLASS ---
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def main():
    # 1. Hardware Check
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[*] using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] WARNING: CUDA not detected. Using CPU (will be slow).")
        device = torch.device("cpu")

    # 2. Load Model
    print(f"[*] Loading model from {MODEL_PATH}...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)

    # 3. Load Data
    print(f"[*] Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=['text', 'label'])
    
    # 4. Prepare Inference
    dataset = InferenceDataset(df['text'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[*] Running inference on {len(df)} samples...")
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    # 5. Map & Matrix
    id2label = model.config.id2label
    predicted_labels = [id2label[p] for p in predictions]
    true_labels = df['label'].tolist()

    # Get unique labels from both truth and predictions to ensure square matrix
    labels = sorted(list(set(true_labels + predicted_labels)))
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX (Rows=True, Cols=Predicted)")
    print("="*80)
    
    # Pandas for readable formatting
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(true_labels, predicted_labels, labels=labels, zero_division=0))

if __name__ == "__main__":
    main()