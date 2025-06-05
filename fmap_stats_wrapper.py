#!/usr/bin/env python3
"""
FMAP Statistics Wrapper
Captures detailed performance statistics from FMAP command line execution
similar to what's available in GUI mode.
"""

import subprocess
import time
import psutil
import json
import re
import sys
import argparse
from pathlib import Path
import threading
import os

class FMAPStatsCollector:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory_mb = 0
        self.memory_samples = []
        self.monitoring = False
        self.process = None
        
    def monitor_memory(self, process):
        """Monitor memory usage in a separate thread"""
        self.monitoring = True
        while self.monitoring:
            try:
                if process and process.poll() is None:
                    # Get memory info for the process and its children
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    for child in process.children(recursive=True):
                        try:
                            memory_mb += child.memory_info().rss / 1024 / 1024
                        except:
                            pass
                    
                    self.memory_samples.append(memory_mb)
                    self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
                
    def parse_fmap_output(self, stdout, stderr):
        """Parse FMAP output to extract available statistics"""
        stats = {
            'plan_length': 0,
            'planning_time_sec': 0.0,
            'total_time_sec': 0.0,
            'num_messages': 0,
            'solution_found': False,
            'grounding_time_ms': 0,
            'heuristic_evaluations': 0,
            'plan_actions': [],
            'agents': []
        }
        
        lines = (stdout + '\n' + stderr).split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract plan length
            match = re.search(r'; Plan length: (\d+)', line)
            if match:
                stats['plan_length'] = int(match.group(1))
                stats['solution_found'] = True
                
            # Extract planning time
            match = re.search(r'; Planning time: ([\d.]+) sec\.', line)
            if match:
                stats['planning_time_sec'] = float(match.group(1))
                
            # Extract total time
            match = re.search(r'Total time: ([\d.]+) sec\.', line)
            if match:
                stats['total_time_sec'] = float(match.group(1))
                
            # Extract number of messages
            match = re.search(r'Number of messages: (\d+)', line)
            if match:
                stats['num_messages'] = int(match.group(1))
                
            # Extract grounding time
            match = re.search(r'Grounding.*?(\d+)ms', line)
            if match:
                stats['grounding_time_ms'] = int(match.group(1))
                
            # Count heuristic evaluations (look for Hdtg patterns)
            if re.search(r'Hdtg\s*=\s*\d+', line) or re.search(r'H=\s*\d+', line):
                stats['heuristic_evaluations'] += 1
                
            # Extract plan actions (lines with timestamp: action format)
            if ':' in line and not line.startswith(';'):
                try:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        timestamp = float(parts[0].strip())
                        action = parts[1].strip()
                        stats['plan_actions'].append({
                            'timestamp': timestamp,
                            'action': action
                        })
                except:
                    pass
                    
            # Look for no solution
            if 'No plan found' in line:
                stats['solution_found'] = False
        
        return stats
        
    def calculate_plan_metrics(self, plan_actions):
        """Calculate plan quality metrics from actions"""
        if not plan_actions:
            return {
                'makespan': 0.0,
                'concurrency_index': 0.0,
                'parallel_actions': 0
            }
            
        timestamps = [action['timestamp'] for action in plan_actions]
        makespan = max(timestamps) if timestamps else 0.0
        
        # Calculate concurrency (actions at same timestamp)
        unique_timestamps = set(timestamps)
        parallel_actions = len(timestamps) - len(unique_timestamps)
        concurrency_index = parallel_actions / len(timestamps) if timestamps else 0.0
        
        return {
            'makespan': makespan,
            'concurrency_index': concurrency_index,
            'parallel_actions': parallel_actions
        }
        
    def run_fmap_with_stats(self, fmap_args):
        """Run FMAP and collect comprehensive statistics"""
        print("🚀 Running FMAP with statistics collection...")
        print(f"Command: java -jar FMAP.jar {' '.join(fmap_args)}")
        
        # Start timing
        self.start_time = time.time()
        
        # Prepare command
        cmd = ['java', '-jar', 'FMAP.jar'] + fmap_args
        
        # Start process
        try:
            self.process = psutil.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Start memory monitoring thread
            memory_thread = threading.Thread(
                target=self.monitor_memory, 
                args=(self.process,)
            )
            memory_thread.start()
            
            # Wait for completion
            stdout, stderr = self.process.communicate()
            self.end_time = time.time()
            
            # Stop memory monitoring
            self.monitoring = False
            memory_thread.join()
            
        except Exception as e:
            self.end_time = time.time()
            self.monitoring = False
            return {
                'error': str(e),
                'wall_clock_time': self.end_time - self.start_time
            }
        
        # Parse FMAP output
        fmap_stats = self.parse_fmap_output(stdout, stderr)
        
        # Calculate derived metrics
        wall_clock_time = self.end_time - self.start_time
        plan_metrics = self.calculate_plan_metrics(fmap_stats['plan_actions'])
        
        # Estimate branching factor (simplified)
        branching_factor = 0.0
        if fmap_stats['heuristic_evaluations'] > 0 and fmap_stats['plan_length'] > 0:
            # Rough estimate based on evaluations vs plan length
            branching_factor = fmap_stats['heuristic_evaluations'] / fmap_stats['plan_length']
        
        # Compile comprehensive statistics
        comprehensive_stats = {
            # Performance metrics
            'wall_clock_time_sec': wall_clock_time,
            'planning_time_sec': fmap_stats['planning_time_sec'],
            'total_time_sec': fmap_stats['total_time_sec'],
            'grounding_time_ms': fmap_stats['grounding_time_ms'],
            
            # Memory metrics
            'peak_memory_mb': self.peak_memory_mb,
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'memory_samples_count': len(self.memory_samples),
            
            # Search metrics
            'heuristic_evaluations': fmap_stats['heuristic_evaluations'],
            'estimated_branching_factor': branching_factor,
            'solution_found': fmap_stats['solution_found'],
            
            # Plan quality metrics
            'plan_length': fmap_stats['plan_length'],
            'makespan': plan_metrics['makespan'],
            'concurrency_index': plan_metrics['concurrency_index'],
            'parallel_actions': plan_metrics['parallel_actions'],
            
            # Communication metrics
            'num_messages': fmap_stats['num_messages'],
            
            # Raw data
            'plan_actions': fmap_stats['plan_actions'],
            'stdout': stdout,
            'stderr': stderr,
            'exit_code': self.process.returncode if self.process else -1
        }
        
        return comprehensive_stats

def print_statistics_report(stats):
    """Print a comprehensive statistics report similar to GUI mode"""
    print("\n" + "="*60)
    print("📊 FMAP EXECUTION STATISTICS")
    print("="*60)
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        print(f"⏱️  Wall clock time: {stats['wall_clock_time']:.3f} sec.")
        return
    
    # Solution status
    status = "✅ SOLUTION FOUND" if stats['solution_found'] else "❌ NO SOLUTION"
    print(f"Status: {status}")
    print()
    
    # Timing statistics
    print("⏱️  TIMING STATISTICS:")
    print(f"   Wall clock time:      {stats['wall_clock_time_sec']:.3f} sec.")
    print(f"   Planning time:        {stats['planning_time_sec']:.3f} sec.")
    print(f"   Total FMAP time:      {stats['total_time_sec']:.3f} sec.")
    print(f"   Grounding time:       {stats['grounding_time_ms']:.0f} ms.")
    print()
    
    # Memory statistics
    print("💾 MEMORY STATISTICS:")
    print(f"   Peak memory usage:    {stats['peak_memory_mb']:.1f} MB")
    print(f"   Average memory:       {stats['avg_memory_mb']:.1f} MB")
    print(f"   Memory samples:       {stats['memory_samples_count']}")
    print()
    
    # Search statistics
    print("🔍 SEARCH STATISTICS:")
    print(f"   Heuristic evaluations: {stats['heuristic_evaluations']}")
    print(f"   Est. branching factor: {stats['estimated_branching_factor']:.3f}")
    print()
    
    # Plan quality (if solution found)
    if stats['solution_found']:
        print("📋 PLAN QUALITY:")
        print(f"   Plan length:          {stats['plan_length']} actions")
        print(f"   Makespan:             {stats['makespan']:.1f}")
        print(f"   Parallel actions:     {stats['parallel_actions']}")
        print(f"   Concurrency index:    {stats['concurrency_index']:.3f}")
        print()
    
    # Communication statistics
    print("📡 COMMUNICATION:")
    print(f"   Messages exchanged:   {stats['num_messages']}")
    print()
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='FMAP Statistics Wrapper')
    parser.add_argument('fmap_args', nargs='+', help='FMAP command line arguments')
    parser.add_argument('--output', '-o', help='Save statistics to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    
    args = parser.parse_args()
    
    # Check if FMAP.jar exists
    if not Path('FMAP.jar').exists():
        print("❌ Error: FMAP.jar not found in current directory")
        sys.exit(1)
    
    # Run FMAP with statistics collection
    collector = FMAPStatsCollector()
    stats = collector.run_fmap_with_stats(args.fmap_args)
    
    # Print report
    print_statistics_report(stats)
    
    # Show verbose output if requested
    if args.verbose and 'stdout' in stats:
        print("\n" + "="*60)
        print("📝 VERBOSE OUTPUT:")
        print("="*60)
        print("STDOUT:")
        print(stats['stdout'])
        if stats['stderr']:
            print("\nSTDERR:")
            print(stats['stderr'])
    
    # Save to JSON if requested
    if args.output:
        # Remove large text fields for JSON output
        json_stats = stats.copy()
        if 'stdout' in json_stats:
            del json_stats['stdout']
        if 'stderr' in json_stats:
            del json_stats['stderr']
            
        with open(args.output, 'w') as f:
            json.dump(json_stats, f, indent=2)
        print(f"\n💾 Statistics saved to: {args.output}")

if __name__ == "__main__":
    main() 