
import os
import glob
import torch
import numpy as np
from scapy.all import rdpcap, SCTP, IP
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Config
VALIDATION_DIR = "validation"
MODEL_PATH = "final_gpt2_5g_classifier"
TOKENIZER_PATH = "tokenizer_oran"
ASSUMED_BITRATE = 8_000_000.0  # 8 Mbps

IP_MAP = {
    "1": "DU", "11": "CU", "5": "CORE",
    "100": "DU", "112": "CU", "132": "CORE"
}

def get_chunk_type(name):
    name = name.upper()
    if "INIT ACK" in name: return "INIT_ACK"
    if "COOKIE ECHO" in name: return "COOKIE_ECHO"
    if "COOKIE ACK" in name: return "COOKIE_ACK"
    if "HEARTBEAT" in name: return "HB"
    if "ABORT" in name: return "ABORT"
    if "ERROR" in name: return "ERROR"
    if "DATA" in name: return "DATA"
    if "SACK" in name: return "SACK"
    if "SHUTDOWN" in name: return "SHUTDOWN"
    if "INIT" in name: return "INIT"
    return "MSG" # Default fallback

def process_pcap(path):
    print(f"Processing {path}...")
    try:
        packets = rdpcap(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    if not packets:
        print("No packets found.")
        return None

    total_bits = 0
    token_sequence = []
    
    for pkt in packets:
        total_bits += len(pkt) * 8
        
        if not pkt.haslayer(SCTP) or not pkt.haslayer(IP):
            continue
            
        src = pkt[IP].src.split('.')[-1]
        dst = pkt[IP].dst.split('.')[-1]
        
        src_label = IP_MAP.get(src, src)
        dst_label = IP_MAP.get(dst, dst) # Only src is mapped in interp_logit_lens? 
        # interp: f"{ip_map.get(src_suf, src_suf)}>{dst_suf}" -> dst is NOT mapped in interp_logit_lens.py line 49!
        # Wait, check line 49 of interp_logit_lens.py: `ip_map.get(src_suf, src_suf)}>{dst_suf}`
        # But traffic_logs_final.csv shows `CU>5`, `CORE>11`. 
        # `11` is CU. `5` is CORE.
        # So `CU>5` means `src=11` (CU) `dst=5`. 5 is NOT mapped.
        # `CORE>11` means `src=5` (CORE) `dst=11`. 11 IS NOT mapped.
        # So `interp_logit_lens.py` ONLY maps SRC?
        # Let's check CSV again.
        # `CU>5:MSG` -> src mapped. dst not.
        # `CORE>11:MSG` -> src mapped. dst not.
        # `DU>11:MSG` -> src mapped. dst not.
        # `CU>1:MSG` -> src mapped. dst not.
        # `2>8:MSG` -> neither mapped.
        # Okay, I will follow interp_logit_lens logic: map ONLY SRC.
        
        direction = f"{src_label}>{dst}"

        chunks = []
        current = pkt.getlayer(SCTP).payload
        while current and current.name != "NoPayload":
            c_type = get_chunk_type(current.name)
            chunks.append(c_type)
            current = current.payload
            
        if not chunks:
            continue
            
        chunk_str = "+".join(chunks)
        token_sequence.append(f"{direction}:{chunk_str}")

    duration = total_bits / ASSUMED_BITRATE
    header = f"[LEN={len(token_sequence)}] [DUR={duration:.2f}s]"
    full_text = f"{header} {' '.join(token_sequence)}"
    return full_text

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_PATH} on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    
    pcap_files = glob.glob(os.path.join(VALIDATION_DIR, "*.pcapng"))
    
    for pcap_path in pcap_files:
        print("-" * 50)
        text = process_pcap(pcap_path)
        if not text: continue
        
        print(f"Input: {text[:100]}...")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            
        probs = torch.softmax(logits, dim=0)
        pred_idx = torch.argmax(logits).item()
        pred_label = model.config.id2label[pred_idx]
        
        print(f"\nPrediction for {os.path.basename(pcap_path)}:")
        print(f"Label: {pred_label}")
        print("Logits:")
        for i, val in enumerate(logits):
            label = model.config.id2label[i]
            print(f"  {label}: {val.item():.4f} (Prob: {probs[i].item():.4f})")

if __name__ == "__main__":
    main()
