import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

# --- CONFIG ---
CSV_FILE = "traffic_logs_final.csv"
MODEL_NAME = "gpt2"
MAX_LEN = 1024  # Max safe size for GPT-2
EPOCHS = 3
BATCH_SIZE = 2  # 1024 tokens is heavy. Reduce batch size to fit VRAM/RAM.

# --- LOAD ---
print("[*] Loading Data...")
df = pd.read_csv(CSV_FILE).dropna()
labels = sorted(df['label'].unique().tolist())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}

# --- SPLIT (Stratified) ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].apply(lambda x: label2id[x]).tolist(), 
    test_size=0.15,
    stratify=df['label'] # CRITICAL for rare classes
)

# --- WEIGHTS ---
class_weights = compute_class_weight(
    class_weight="balanced", 
    classes=np.unique(train_labels), 
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)
print(f"[*] Class Weights: {class_weights}")

# --- TOKENIZER ---
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)

train_enc = tokenize_function(train_texts)
val_enc = tokenize_function(val_texts)

class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

train_dataset = TrafficDataset(train_enc, train_labels)
val_dataset = TrafficDataset(val_enc, val_labels)

# --- MODEL ---
class WeightedGPT2(GPT2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            weights = class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, logits) if loss is not None else logits

model = WeightedGPT2.from_pretrained(MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id)
model.config.pad_token_id = tokenizer.pad_token_id

# --- TRAIN ---
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available() # Use Mixed Precision if GPU available
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Output full report to logs so you can see per-class performance
    print("\n" + classification_report(labels, preds, target_names=[id2label[i] for i in range(len(id2label))], zero_division=0))
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("[*] Starting Training...")
trainer.train()
model.save_pretrained("./final_gpt2_5g_classifier")
tokenizer.save_pretrained("./final_gpt2_5g_classifier")