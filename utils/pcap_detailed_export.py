
import glob
import datetime
from scapy.all import rdpcap, IP, SCTP
from scapy.layers.sctp import *

def format_packet(pkt):
    try:
        if not pkt.haslayer(IP) or not pkt.haslayer(SCTP):
            return None
            
        # Timestamp
        ts = datetime.datetime.fromtimestamp(float(pkt.time)).strftime('%H:%M:%S.%f')
        
        # IP and Ports
        ip_src = pkt[IP].src
        ip_dst = pkt[IP].dst
        sport = pkt[SCTP].sport
        dport = pkt[SCTP].dport
        
        # Chunks
        chunks_str = ""
        current = pkt[SCTP].payload
        
        chunk_details = []
        
        while current and current.name != "NoPayload":
            c_name = current.name
            c_info = f"[{c_name}]"
            
            # Add specific details if possible (simplified for now)
            if c_name == "INIT":
                if hasattr(current, 'init_tag'): c_info += f" [init tag: {current.init_tag}]"
            elif c_name == "DATA":
                if hasattr(current, 'tsn'): c_info += f" [TSN: {current.tsn}]"
                if hasattr(current, 'stream_id'): c_info += f" [SID: {current.stream_id}]"
                if hasattr(current, 'stream_seq'): c_info += f" [SSEQ {current.stream_seq}]"
            elif c_name == "SACK":
                if hasattr(current, 'cum_tsn_ack'): c_info += f" [cum ack {current.cum_tsn_ack}]"
                
            chunk_details.append(c_info)
            current = current.payload
            
        if not chunk_details:
            return None
            
        chunks_str = " ".join(chunk_details)
        
        # Construct line
        # 02:09:16.875602 IP 127.0.10.1.57427 > 127.0.10.11.38472: sctp (1) [INIT] ...
        line = f"{ts} IP {ip_src}.{sport} > {ip_dst}.{dport}: sctp (1) {chunks_str}"
        return line
        
    except Exception as e:
        return None

def process_file(pcap_path):
    print(f"Reading {pcap_path}...")
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Failed to read {pcap_path}: {e}")
        return

    out_path = pcap_path + ".txt"
    lines = []
    for pkt in packets:
        line = format_packet(pkt)
        if line:
            lines.append(line)
            
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {len(lines)} packets to {out_path}")

def main():
    # Validation files
    val_files = glob.glob('validation/*.pcapng')
    for f in val_files:
        process_file(f)
        
    # Dataset 8 files
    ds_files = glob.glob('dataset2_5k/dataset/8/*.pcap') 
    for f in ds_files:
        process_file(f)

if __name__ == "__main__":
    main()
