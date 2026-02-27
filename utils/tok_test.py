import os
import glob
import random
from transformers import PreTrainedTokenizerFast

# --- CONFIG ---
DATA_DIR = "dataset_output_5k"
TOKENIZER_PATH = "tokenizer_oran"

def test_tokenizer():
    # 1. Load Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: {TOKENIZER_PATH} not found. Did you run train_tokenizer_v3.py?")
        return
    
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
    
    # 2. Gather Files
    files = glob.glob(os.path.join(DATA_DIR, "*", "capture.txt"))
    if not files:
        print(f"No files found in {DATA_DIR}. Check your path.")
        return

    # 3. Pick 3 Random Victims
    samples = random.sample(files, 3)
    
    print(f"\n--- 🔍 RANDOM SAMPLE INSPECTION ---")
    
    for i, file_path in enumerate(samples):
        sample_id = file_path.split(os.sep)[-2] # Extract folder name (e.g. '1042')
        print(f"\n\033[1;36m=== SAMPLE {i+1} (ID: {sample_id}) ===\033[0m")
        
        try:
            with open(file_path, 'r') as f:
                # Get first 3 non-empty lines
                lines = [l.strip() for l in f.readlines() if l.strip()][:3]
            
            if not lines:
                print("\033[91m(Empty File - Likely 'Squeeze' or 'Whisper' Failure)\033[0m")
                continue
                
            for line in lines:
                tokens = tokenizer.tokenize(line)
                
                # Check compression ratio
                char_len = len(line)
                tok_len = len(tokens)
                ratio = char_len / tok_len if tok_len > 0 else 0
                
                print(f"Raw:    {line}")
                print(f"Tokens: {tokens}")
                print(f"Stats:  {char_len} chars -> {tok_len} tokens ({ratio:.1f}x compression)")
                print("-" * 40)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 4. The "Atom" Check
    print(f"\n--- 🧪 ATOM VERIFICATION ---")
    print("Checking if critical protocol concepts are single tokens...")
    
    targets = [
        "127.0.10.1", "127.0.10.11",  # IPs
        "[INIT]", "[SACK]", "[DATA]", "[ABORT]", # Chunk Types
        "init tag:", "cum ack", "TSN:" # Fields
    ]
    
    vocab = tokenizer.get_vocab()
    for t in targets:
        # Check for token 't' OR 'Ġt' (leading space version)
        is_there = t in vocab or f"Ġ{t}" in vocab
        status = "✅" if is_there else "❌"
        print(f"{status} {t}")

if __name__ == "__main__":
    test_tokenizer()