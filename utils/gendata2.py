import os
import re
import glob
import pandas as pd
from scapy.all import rdpcap, SCTP
from scapy.layers.inet import IP

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset2_5k/dataset"
OUTPUT_FILE = "traffic_logs.csv"

# 1. Map Run ID to the Config Class
def get_config_class(run_id):
    rid = int(run_id)
    if 0 <= rid < 100: return "BASELINE"
    if 100 <= rid < 1600: return "TRANSPORT"
    if 1600 <= rid < 3100: return "PROTOCOL"
    if 3100 <= rid < 4100: return "THROUGHPUT"
    if 4100 <= rid < 5000: return "BUFFER"
    raise ValueError(f"Run ID {rid} is out of expected range (0-5000)")

# 2. Extract Bandwidth (STRICT)
def get_bandwidth(path):
    log_path = os.path.join(path, "iperf-client.log")
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing iperf log: {log_path}")

    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
        
    # STRICT: We only accept the sender summary. 
    # If the client finished, this line MUST exist.
    match = re.search(r'([\d\.]+)\s+Mbits/sec.*sender', content)
    
    if not match:
        return 0.0
        raise ValueError(f"CRITICAL: 'sender' statistics missing in {log_path}. Run likely crashed or was killed early.")

    return float(match.group(1))

# 3. Determine Label
def get_label(run_id, bandwidth):
    if bandwidth >= 1.0:
        return "INTEROPERATING"
    return get_config_class(run_id)

# 4. PCAP Serializer (STRICT)
def pcap_to_text(pcap_path):
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"Missing PCAP: {pcap_path}")
    
    # Let Scapy crash if the file is malformed. No try/except.
    packets = rdpcap(pcap_path, count=40) 
    
    if not packets:
        raise ValueError(f"PCAP is 0 bytes or empty: {pcap_path}")

    token_sequence = []
    
    for i, pkt in enumerate(packets): 
        # Skip non-SCTP noise if any (though filter should prevent it)
        if not pkt.haslayer(SCTP): 
            continue
        
        # STRICT: Every SCTP packet must have an IP layer.
        if not pkt.haslayer(IP):
            raise ValueError(f"Malformed Packet {i} in {pcap_path}: SCTP layer present without IPv4 header.")
            
        src = pkt[IP].src.split('.')[-1]
        dst = pkt[IP].dst.split('.')[-1]
        direction = f"{src}>{dst}"

        # Chunk Analysis
        chunks = []
        current = pkt.getlayer(SCTP).payload
        
        # Iterate chunks
        while current and current.name != "NoPayload":
            name = current.name
            
            # Map Scapy names to Tokens
            if "INIT ACK" in name: c_type = "INIT_ACK"
            elif "COOKIE ECHO" in name: c_type = "COOKIE_ECHO"
            elif "COOKIE ACK" in name: c_type = "COOKIE_ACK"
            elif "HEARTBEAT" in name: c_type = "HB"
            elif "ABORT" in name: c_type = "ABORT"
            elif "ERROR" in name: c_type = "ERROR"
            elif "DATA" in name: c_type = "DATA"
            elif "SACK" in name: c_type = "SACK"
            elif "SHUTDOWN" in name: c_type = "SHUTDOWN"
            elif "INIT" in name: c_type = "INIT"
            else: c_type = name.replace(" ", "_")
            
            chunks.append(c_type)
            current = current.payload

        # STRICT: An SCTP packet MUST have at least one chunk or valid payload.
        if not chunks:
             raise ValueError(f"Packet {i} in {pcap_path} has SCTP header but no Chunks/Payload.")

        chunk_str = "+".join(chunks)
        token_sequence.append(f"{direction}:{chunk_str}")

    # STRICT: If we parsed the file but found 0 valid SCTP packets, the run is useless.
    if not token_sequence:
        raise ValueError(f"No valid SCTP packets found in {pcap_path} (Filter failure or empty capture).")

    return " ".join(token_sequence)

def main():
    data = []
    run_dirs = sorted([d for d in os.listdir(DATASET_ROOT) if d.isdigit()], key=int)
    total = len(run_dirs)
    
    print(f"[*] Processing {total} runs in RUTHLESS STRICT MODE...")
    
    for i, run_id_str in enumerate(run_dirs):
        run_id = int(run_id_str)
        path = os.path.join(DATASET_ROOT, run_id_str)
        
        # 1. Get Bandwidth (Crashes if missing)
        bw = get_bandwidth(path)
        
        # 2. Get Features (Crashes if missing/malformed)
        pcap_file = os.path.join(path, "wire_trace.pcap")
        text_seq = pcap_to_text(pcap_file)
        
        # 3. Label
        final_label = get_label(run_id, bw)

        data.append({
            "run_id": run_id,
            "text": text_seq,
            "label": final_label,
            "bandwidth": bw
        })
        
        if i % 100 == 0:
            print(f"    Processed {i}/{total}...", end="\r")

    print(f"\n[*] Saving {len(data)} rows to {OUTPUT_FILE}...")
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n[*] Class Distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()