
import re
import statistics
import glob
import os

def analyze_html(filename):
    print(f"Analyzing {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Regex to capture token and score
    # <div class="tooltip token" ...>
    #     TOKEN (or maybe multiple lines)
    #     <span class="tooltiptext">Score: 0.1234</span>
    # </div>
    pattern = re.compile(r'<div class="tooltip token"[^>]*>\s*(.*?)\s*<span class="tooltiptext">Score:\s*([0-9.]+)</span>', re.DOTALL)
    
    matches = pattern.findall(content)
    
    scores = []
    token_data = []

    for text, score_str in matches:
        try:
            score = float(score_str)
            scores.append(score)
            token_data.append((text.strip(), score))
        except ValueError:
            continue
    
    if not scores:
        print("No scores found.")
        return

    max_score = max(scores)
    mean_score = statistics.mean(scores)
    non_zero = [s for s in scores if s > 0.0]
    
    # Calculate sparsity
    total_tokens = len(scores)
    sparsity = (total_tokens - len(non_zero)) / total_tokens if total_tokens > 0 else 0

    print(f"  Total tokens: {total_tokens}")
    print(f"  Non-zero scores: {len(non_zero)} ({len(non_zero)/total_tokens:.2%})")
    print(f"  Max score: {max_score:.6f}")
    print(f"  Mean score: {mean_score:.6f}")
    
    # Top 5 tokens
    token_data.sort(key=lambda x: x[1], reverse=True)
    print("  Top 5 tokens:")
    for text, score in token_data[:5]:
        # Truncate text if too long
        display_text = (text[:20] + '..') if len(text) > 20 else text
        print(f"    '{display_text}': {score:.6f}")
    print("-" * 30)

files = glob.glob("token_importance*.html")
for f in files:
    analyze_html(f)
