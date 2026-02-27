import os
import re
import pandas as pd
import numpy as np
from scapy.all import rdpcap, SCTP
from scapy.layers.inet import IP
from transformers import GPT2Tokenizer
from tqdm import tqdm

# --- CONFIG ---
DATASET_ROOT = "./dataset2_5k/dataset"
OUTPUT_FILE = "traffic_logs_final.csv"
MODEL_NAME = "gpt2"

def get_config_class(run_id):
    rid = int(run_id)
    if 0 <= rid < 100: return "BASELINE"
    if 100 <= rid < 1600: return "TRANSPORT"
    if 1600 <= rid < 3100: return "PROTOCOL"
    if 3100 <= rid < 4100: return "THROUGHPUT"
    if 4100 <= rid < 5000: return "BUFFER"
    raise ValueError(f"Run ID {rid} out of range")

def get_bandwidth(path):
    log_path = os.path.join(path, "iperf-client.log")
    if not os.path.exists(log_path): raise FileNotFoundError(f"Missing {log_path}")
    
    with open(log_path, 'r', errors='ignore') as f: content = f.read()
    
    # Use 0.0 if sender stats are missing instead of crashing
    match = re.search(r'([\d\.]+)\s+Mbits/sec.*sender', content)
    if not match:
        return 0.0
    
    return float(match.group(1))

def get_label(run_id, bandwidth):
    if bandwidth >= 1.0: return "INTEROPERATING"
    return get_config_class(run_id)

def pcap_to_text(pcap_path):
    if not os.path.exists(pcap_path): raise FileNotFoundError(f"Missing PCAP: {pcap_path}")
    
    # Read full file
    packets = rdpcap(pcap_path)
    if not packets: raise ValueError(f"Empty PCAP: {pcap_path}")

    pkt_count = len(packets)
    duration = packets[-1].time - packets[0].time
    token_sequence = [f"[LEN={pkt_count}]", f"[DUR={float(duration):.2f}s]"]
    
    for i, pkt in enumerate(packets): 
        if not pkt.haslayer(SCTP): continue
        if not pkt.haslayer(IP): raise ValueError(f"Pkt {i} in {pcap_path} missing IP layer")
            
        src = pkt[IP].src.split('.')[-1]
        dst = pkt[IP].dst.split('.')[-1]
        
        # Mapping for 5G logical entities
        src_tok = {"11": "CU", "1": "DU", "5": "CORE"}.get(src, src)

        chunks = []
        current = pkt.getlayer(SCTP).payload
        while current and current.name != "NoPayload":
            name = current.name
            if "INIT ACK" in name: c = "INIT_ACK"
            elif "COOKIE ECHO" in name: c = "COOKIE_ECHO"
            elif "COOKIE ACK" in name: c = "COOKIE_ACK"
            elif "HEARTBEAT" in name: c = "HB"
            elif "ABORT" in name: c = "ABORT"
            elif "ERROR" in name: c = "ERROR"
            elif "DATA" in name: c = "DATA"
            elif "SACK" in name: c = "SACK"
            elif "SHUTDOWN" in name: c = "SHUTDOWN"
            elif "INIT" in name: c = "INIT"
            else: c = "MSG"
            chunks.append(c)
            current = current.payload
            
        if not chunks: continue
        token_sequence.append(f"{src_tok}>{dst}:{'+'.join(chunks)}")

    if len(token_sequence) <= 2: raise ValueError(f"No SCTP chunks found in {pcap_path}")
    return " ".join(token_sequence)

def main():
    data = []
    run_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if d.isdigit()], key=int)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    token_lengths = []

    print(f"[*] Processing {len(run_dirs)} runs...")

    for run_id_str in tqdm(run_dirs):
        path = os.path.join(DATASET_ROOT, run_id_str)
        
        # Only catches PCAP structural errors or missing files now
        bw = get_bandwidth(path)
        label = get_label(int(run_id_str), bw)
        text = pcap_to_text(os.path.join(path, "wire_trace.pcap"))
        
        tokens = tokenizer.encode(text)
        token_lengths.append(len(tokens))
        data.append({"text": text, "label": label, "len": len(tokens)})

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*50 + "\nDATASET AUDIT REPORT\n" + "="*50)
    print(f"Total Samples: {len(df)}")
    print(f"Avg Tokens: {np.mean(token_lengths):.1f} | Max: {np.max(token_lengths)}")
    print(f"95th Percentile: {np.percentile(token_lengths, 95):.1f}")
    print(f"Truncation (>1024): {len([x for x in token_lengths if x > 1024])} samples")
    print("-" * 30 + "\nClass Distribution:\n", df['label'].value_counts())

if __name__ == "__main__":
    main()