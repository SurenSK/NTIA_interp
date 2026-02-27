import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys

# --- CONFIG ---
MODEL_PATH = "./final_gpt2_5g_classifier"
DATA_FILE = "traffic_logs_final.csv"
BATCH_SIZE = 32
TARGET_CLASSES = ["INTEROPERATING", "PROTOCOL", "TRANSPORT"]

# --- DATASET ---
class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}

def main():
    # 1. Load Data & Filter for Target Classes Only
    print(f"[*] Loading data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE).dropna()
    
    # Filter: Keep only rows where label is in the Big 3
    df_filtered = df[df['label'].isin(TARGET_CLASSES)].copy()
    print(f"[*] Filtered dataset size: {len(df_filtered)} (Original: {len(df)})")

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[!] Failed to load model: {e}")
        sys.exit(1)

    # 3. Inference
    dataset = InferenceDataset(df_filtered['text'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions = []
    print("[*] Running Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    # 4. Map back to labels
    id2label = model.config.id2label
    pred_labels = [id2label[p] for p in predictions]
    true_labels = df_filtered['label'].tolist()

    # 5. Generate Confusion Matrix (3x3)
    cm = confusion_matrix(true_labels, pred_labels, labels=TARGET_CLASSES)
    
    # Normalize (Optional: allows you to see percentages vs raw counts)
    # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 6. Plot
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    heatmap = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', # 'd' for integer counts, '.2f' for percentages
        cmap='Blues', 
        xticklabels=TARGET_CLASSES, 
        yticklabels=TARGET_CLASSES,
        annot_kws={"size": 14}
    )
    
    plt.title('5G Failure Classification (GPT-2)', fontsize=16, pad=20)
    plt.ylabel('Actual Cause', fontsize=14)
    plt.xlabel('Predicted Cause', fontsize=14)
    
    output_img = "confusion_matrix_3x3.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\n[*] Matrix saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    main()