#!/usr/bin/env python3
"""
Create all_results.json from individual result files for experiment_runner.py analysis
"""

import json
import glob
from pathlib import Path

def create_all_results():
    results_dir = Path("results")
    
    # Find all individual result files
    result_files = glob.glob(str(results_dir / "result_*.json"))
    print(f"Found {len(result_files)} individual result files")
    
    all_results = []
    
    for file_path in sorted(result_files):
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
                all_results.append(result_data)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Save as all_results.json
    output_file = results_dir / "all_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Created {output_file} with {len(all_results)} results")

if __name__ == "__main__":
    create_all_results() 