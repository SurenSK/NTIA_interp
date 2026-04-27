import os
import subprocess
import argparse
from pathlib import Path

def get_tshark_path():
    # Common Windows paths for Wireshark
    paths = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def convert_pcaps(directory, detail_level="summary"):
    tshark_path = get_tshark_path()
    if not tshark_path:
        print("Error: Could not find tshark.exe. Please ensure Wireshark is installed.")
        return

    dir_path = Path(directory)
    pcap_files = list(dir_path.glob("**/*.pcap")) # Search recursively just in case
    
    if not pcap_files:
        print(f"No .pcap files found in '{directory}'.")
        return

    print(f"Found {len(pcap_files)} PCAP files. Starting conversion...")
    
    for pcap in pcap_files:
        out_txt = pcap.with_name(f"{pcap.stem}_{detail_level}.txt")
        
        print(f"\nProcessing: {pcap.name}")
        
        cmd = [tshark_path, "-r", str(pcap)]
        
        if detail_level == "full":
            cmd.append("-V")
        elif detail_level == "json":
            cmd.extend(["-T", "json"])
            out_txt = pcap.with_name(f"{pcap.stem}.json")
            
        print(f"  Running command: {' '.join(cmd)}")
        print(f"  Outputting to : {out_txt.name}")
        
        try:
            with open(out_txt, "w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL, check=True)
            print(f"  Done. Wrote to {out_txt.name}")
        except subprocess.CalledProcessError as e:
            print(f"  Failed: tshark exited with code {e.returncode}")
        except Exception as e:
            print(f"  An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PCAPs to text using TShark.")
    parser.add_argument("directory", help="The directory containing the .pcap files", default=".", nargs="?")
    parser.add_argument("--detail", "-d", choices=["summary", "full", "json"], default="summary", 
                        help="Level of output detail. 'summary' is 1-line-per-packet view. 'full' is full protocol tree.")
    
    args = parser.parse_args()
    convert_pcaps(args.directory, args.detail)
