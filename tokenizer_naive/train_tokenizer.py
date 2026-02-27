import os
import glob
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

DATA_DIR = "dataset"
VOCAB_SIZE = 5000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
    "[START_PCAP]", "[END_PCAP]", 
    "[START_RESULT]", "[END_RESULT]",
    "[FAM_WHISPER]", "[FAM_STUTTER]", "[FAM_GHOST]", 
    "[FAM_MUTE]", "[FAM_BINGE]", "[FAM_SQUEEZE]",
    "[FAM_INTEROPING]",
    "127.0.10.1", "127.0.10.11", "127.0.10.12", "127.0.0.5", "127.0.10.99",
    "[INIT]", "[INIT ACK]", "[COOKIE ECHO]", "[COOKIE ACK]", 
    "[DATA]", "[SACK]", "[ABORT]", "[ERROR]", "[HB REQ]", "[HB ACK]", 
    "[SHUTDOWN]", "[SHUTDOWN ACK]",
    "init tag:", "rwnd:", "OS:", "MIS:", "init TSN:",
    "cum ack", "a_rwnd", "#gap acks", "#dup tsns",
    "TSN:", "SID:", "SSEQ", "PPID"
]

def train_tokenizer():
    files = glob.glob(os.path.join(DATA_DIR, "*", "capture.txt"))
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )
    output_path = "tokenizer_test"
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_model(output_path)
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=SPECIAL_TOKENS
    )
    hf_tokenizer.save_pretrained(output_path)
    print(f"Deliverable saved to: {output_path}")

if __name__ == "__main__":
    train_tokenizer()