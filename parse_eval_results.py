#!/usr/bin/env python3
"""
Parse evaluation results from raw log files and convert to clean JSON.
Handles numpy types properly.
"""

import json
import re
import sys
from pathlib import Path

def parse_raw_results(raw_path):
    """Parse evaluation results from raw log file."""
    text = Path(raw_path).read_text(errors='ignore')
    
    # Find the results dictionary (last occurrence)
    lines = text.splitlines()
    result_line = None
    
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith('{') and 'arc_challenge' in stripped:
            result_line = stripped
            break
    
    if result_line is None:
        print(f"ERROR: No results found in {raw_path}")
        return None
    
    # Replace numpy types with regular Python types
    result_line = re.sub(r'np\.float64\(([\d.e+-]+)\)', r'\1', result_line)
    result_line = re.sub(r'np\.int64\((\d+)\)', r'\1', result_line)
    
    # Use eval() safely (we trust our own output)
    try:
        results = eval(result_line)
        return results
    except Exception as e:
        print(f"ERROR parsing {raw_path}: {e}")
        return None

def extract_summary(results):
    """Extract summary metrics from full results."""
    summary = {}
    
    # Main benchmarks
    benchmarks = ['lambada_openai', 'mmlu', 'arc_challenge', 'arc_easy', 'hellaswag', 'piqa', 'winogrande']
    
    for bench in benchmarks:
        if bench in results:
            bench_data = results[bench]
            if 'acc,none' in bench_data:
                summary[bench] = bench_data['acc,none']
            elif 'acc_norm,none' in bench_data:
                summary[bench] = bench_data['acc_norm,none']
            elif 'perplexity,none' in bench_data:
                summary[bench] = bench_data['perplexity,none']
    
    return summary

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_eval_results.py <raw_file> [output_json]")
        sys.exit(1)
    
    raw_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = raw_path.replace('.raw', '.json')
    
    print(f"Parsing: {raw_path}")
    results = parse_raw_results(raw_path)
    
    if results is None:
        sys.exit(1)
    
    # Extract summary
    summary = extract_summary(results)
    
    # Save full results
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results saved to: {out_path}")
    print(f"\n📊 Summary:")
    for bench, score in summary.items():
        if isinstance(score, float) and score < 10:  # Accuracy (0-1)
            print(f"  {bench:20s}: {score:.4f}")
        else:  # Perplexity
            print(f"  {bench:20s}: {score:.2f}")

if __name__ == '__main__':
    main()
