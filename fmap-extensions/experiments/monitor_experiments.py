#!/usr/bin/env python3

import os
import subprocess
import time
import glob

def check_experiment_progress():
    print("=== FMAP Experiment Monitor ===")
    
    # Check running FMAP processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        fmap_processes = [line for line in result.stdout.split('\n') if 'FMAP.jar' in line and 'grep' not in line]
        
        if fmap_processes:
            print(f"\n🔄 RUNNING EXPERIMENTS: {len(fmap_processes)}")
            for i, proc in enumerate(fmap_processes, 1):
                parts = proc.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                
                # Get detailed timing info
                timing_result = subprocess.run(['ps', '-p', pid, '-o', 'pid,pcpu,pmem,etime,time'], 
                                             capture_output=True, text=True)
                if timing_result.returncode == 0:
                    timing_lines = timing_result.stdout.strip().split('\n')
                    if len(timing_lines) > 1:
                        timing_data = timing_lines[1].split()
                        elapsed = timing_data[3] if len(timing_data) > 3 else "unknown"
                        cpu_time = timing_data[4] if len(timing_data) > 4 else "unknown"
                        
                        print(f"   Process {i}: PID {pid}, CPU {cpu}%, MEM {mem}%, Elapsed: {elapsed}, CPU Time: {cpu_time}")
                        
                        # Try to extract heuristic from command line
                        if '-h' in proc:
                            h_index = proc.find('-h') + 3
                            heuristic_num = proc[h_index:h_index+2].strip()
                            heuristics = {
                                '1': 'DTG',
                                '2': 'DTG+Landmarks', 
                                '3': 'Inc_DTG+Landmarks',
                                '4': 'Centroids',
                                '5': 'MCS'
                            }
                            heuristic_name = heuristics.get(heuristic_num, f"Unknown({heuristic_num})")
                            print(f"        Heuristic: {heuristic_name}")
        else:
            print("\nNO EXPERIMENTS CURRENTLY RUNNING")
    
    except Exception as e:
        print(f"Error checking processes: {e}")
    
    # Check completed experiments
    results_dir = "experiments/results"
    if os.path.exists(results_dir):
        completed = len([f for f in os.listdir(results_dir) if f.startswith('result_') and f.endswith('.json')])
        print(f"\nCOMPLETED EXPERIMENTS: {completed}/65")
        
        # Show latest results
        result_files = glob.glob(os.path.join(results_dir, "result_*.json"))
        if result_files:
            latest_file = max(result_files, key=os.path.getmtime)
            mod_time = time.ctime(os.path.getmtime(latest_file))
            print(f"   Latest result: {os.path.basename(latest_file)} ({mod_time})")
    
    # Calculate estimated time remaining
    if fmap_processes and completed > 0:
        # Simple estimation based on average time per experiment
        avg_time_per_exp = 60  # Rough estimate in seconds
        remaining_experiments = 65 - completed
        estimated_minutes = (remaining_experiments * avg_time_per_exp) / 60
        print(f"\n ESTIMATED TIME REMAINING: ~{estimated_minutes:.1f} minutes")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        print("Starting continuous monitoring (Ctrl+C to stop)...")
        try:
            while True:
                check_experiment_progress()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        check_experiment_progress()
        print("\nTip: Run with --continuous for live monitoring") 