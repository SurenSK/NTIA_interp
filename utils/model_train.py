import os
import glob
import json
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# --- CONFIG ---
TOKENIZER_PATH = "tokenizer_oran"
DATA_DIR = "dataset_output_5k"
OUTPUT_DIR = "model_nano_interop" # New Nano Dir
MAX_LEN = 2048 
FAM_LOSS_WEIGHT = 50.0 

class ORANDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_len=2048):
        self.tokenizer = tokenizer
        self.samples = []
        files = glob.glob(os.path.join(data_dir, "*"))
        print(f"Loading {len(files)} samples...")
        
        for d in files:
            try:
                with open(os.path.join(d, "capture.txt"), "r") as f:
                    pcap = f.read().strip()
                with open(os.path.join(d, "metadata.json"), "r") as f:
                    meta = json.load(f)
                
                # Dynamic Token + Interop Support
                fam_key = meta['family'].upper()
                fam_token = f"[FAM_{fam_key}]"
                
                # Skip if hotpatch didn't work or token missing
                if fam_token not in tokenizer.get_vocab():
                    continue 

                text = (
                    f"[START_PCAP]\n{pcap}\n[END_PCAP]\n"
                    f"[START_RESULT]{fam_token} {meta['parameter_name']} {meta['parameter_value']} [END_RESULT]"
                )
                
                enc = tokenizer(text, truncation=True, max_length=max_len, padding="max_length")
                self.samples.append(enc)
            except:
                pass
        print(f"Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.samples[idx].items()}
        item["labels"] = item["input_ids"].clone()
        return item

class FamilyLossTrainer(Trainer):
    def __init__(self, start_result_token_id, fam_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_result_token_id = start_result_token_id
        self.fam_weight = fam_weight
        self.loss_fct = CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        start_result_mask = (inputs["input_ids"] == self.start_result_token_id)
        valid_indices = start_result_mask[:, :-1]
        
        if valid_indices.sum() > 0:
            fam_logits = shift_logits[valid_indices]
            fam_labels = shift_labels[valid_indices]
            fam_loss = self.loss_fct(fam_logits, fam_labels)
            total_loss = lm_loss + (self.fam_weight * fam_loss)
            self.log({"lm": lm_loss.item(), "fam": fam_loss.item()})
        else:
            total_loss = lm_loss

        return (total_loss, outputs) if return_outputs else total_loss

def train():
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = "[PAD]"
    start_result_id = tokenizer.convert_tokens_to_ids("[START_RESULT]")
    
    # --- HOTPATCH ---
    new_tokens = ["[FAM_INTEROPING]"]
    if tokenizer.add_tokens(new_tokens) > 0:
        print(f"Hotpatched tokenizer with: {new_tokens}")
    
    # --- NANO ARCHITECTURE (Fast) ---
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=2048,
        n_embd=256,       # Small
        n_layer=4,        # Shallow
        n_head=8,         # Parallel
        n_inner=1024,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )
    
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model Params: {model.num_parameters() / 1e6:.2f}M")

    dataset = ORANDataset(tokenizer, DATA_DIR, MAX_LEN)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=6,             # Reduced epochs
        per_device_train_batch_size=8,  # Higher batch size (fits in VRAM)
        per_device_eval_batch_size=16,
        logging_steps=10,
        learning_rate=3e-4,
        save_strategy="no",
        fp16=True if torch.cuda.is_available() else False,
    )

    trainer = FamilyLossTrainer(
        start_result_token_id=start_result_id,
        fam_weight=FAM_LOSS_WEIGHT,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✨ Nano Interop Model Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()