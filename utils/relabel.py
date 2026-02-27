import os
import glob
import json
import re
from tqdm import tqdm

DATA_DIR = "dataset_output_5k"

def assess_pcap(pcap_text):
    """
    Returns True if the PCAP shows a successful, surviving connection.
    Returns False if it looks like a failure (Abort, Hang, etc).
    """
    # 1. Critical Failure Flags
    if "ABORT" in pcap_text:
        return False # Dead.
    
    # 2. Handshake Check
    # We look for the 4-way SCTP handshake
    has_init = "[INIT]" in pcap_text
    has_init_ack = "[INIT ACK]" in pcap_text
    has_cookie = "[COOKIE ECHO]" in pcap_text
    has_cookie_ack = "[COOKIE ACK]" in pcap_text
    
    if not (has_init and has_init_ack and has_cookie and has_cookie_ack):
        # If handshake didn't finish, it's a failure (or Mute start)
        return False
        
    # 3. Sustained Traffic Check
    # Count number of data packets
    data_count = pcap_text.count("[DATA]")
    
    # Heuristic: A healthy session should have some data exchange. 
    # Mute usually dies after ~4-5 packets. Healthy streams have 10+.
    if data_count < 6:
        return False
        
    return True

def run_relabeling():
    files = glob.glob(os.path.join(DATA_DIR, "*"))
    print(f"Scanning {len(files)} samples for survivors...")
    
    relabel_count = 0
    original_counts = {}
    
    for d in tqdm(files):
        pcap_path = os.path.join(d, "capture.txt")
        meta_path = os.path.join(d, "metadata.json")
        
        try:
            with open(pcap_path, "r") as f:
                pcap = f.read()
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            original_fam = meta.get('family', 'unknown').upper()
            original_counts[original_fam] = original_counts.get(original_fam, 0) + 1
            
            # JUDGEMENT DAY
            is_healthy = assess_pcap(pcap)
            
            if is_healthy:
                # Override the family label
                meta['original_family'] = meta['family'] # Save backup
                meta['family'] = "INTEROPING"
                
                # Write it back
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=4)
                
                relabel_count += 1
                
        except Exception as e:
            continue

    print(f"\n--- RELABELING COMPLETE ---")
    print(f"Total Rescued Samples: {relabel_count} / {len(files)}")
    print(f"Success Rate: {relabel_count/len(files)*100:.1f}%")
    print("\nIf this number is high (>50%), your previous model was trying to classify 'Success' as 'Failure'.")

if __name__ == "__main__":
    run_relabeling()