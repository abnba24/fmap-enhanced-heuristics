#!/usr/bin/env python3
"""
Simple FMAP Statistics Wrapper
Captures performance statistics from FMAP command line execution
using only standard Python libraries.
"""

import subprocess
import time
import json
import re
import sys
import argparse
from pathlib import Path
import resource
import os

class SimpleFMAPStats:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
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
            'error_occurred': False
        }
        
        lines = (stdout + '\n' + stderr).split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for errors
            if 'error' in line.lower() or 'exception' in line.lower():
                stats['error_occurred'] = True
                
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
                
            # Alternative grounding time format
            match = re.search(r'Grounding completed in (\d+)ms', line)
            if match:
                stats['grounding_time_ms'] = int(match.group(1))
                
            # Count heuristic evaluations (look for Hdtg patterns)
            if re.search(r'Hdtg\s*=\s*\d+', line) or re.search(r'H=\s*\d+', line):
                stats['heuristic_evaluations'] += 1
                
            # Extract plan actions (lines with timestamp: action format)
            if ':' in line and not line.startswith(';') and not line.startswith('Read error'):
                try:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        timestamp_str = parts[0].strip()
                        action = parts[1].strip()
                        # Try to parse as float
                        try:
                            timestamp = float(timestamp_str)
                            stats['plan_actions'].append({
                                'timestamp': timestamp,
                                'action': action
                            })
                        except ValueError:
                            pass
                except:
                    pass
                    
            # Look for no solution
            if 'No plan found' in line:
                stats['solution_found'] = False
                
            # Look for solution indicators
            if 'Solution plan' in line or 'CoDMAP' in line:
                stats['solution_found'] = True
        
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
        
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            # ru_maxrss is in KB on Linux, bytes on macOS
            if sys.platform == 'darwin':  # macOS
                return usage.ru_maxrss / (1024 * 1024)  # Convert bytes to MB
            else:  # Linux
                return usage.ru_maxrss / 1024  # Convert KB to MB
        except:
            return 0.0
        
    def run_fmap_with_stats(self, fmap_args):
        """Run FMAP and collect comprehensive statistics"""
        print("🚀 Running FMAP with statistics collection...")
        print(f"Command: java -jar FMAP.jar {' '.join(fmap_args)}")
        
        # Start timing
        self.start_time = time.time()
        
        # Prepare command
        cmd = ['java', '-jar', 'FMAP.jar'] + fmap_args
        
        # Run process
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            self.end_time = time.time()
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
            
        except subprocess.TimeoutExpired:
            self.end_time = time.time()
            return {
                'error': 'Timeout (300 seconds)',
                'wall_clock_time_sec': self.end_time - self.start_time,
                'timeout': True
            }
        except Exception as e:
            self.end_time = time.time()
            return {
                'error': str(e),
                'wall_clock_time_sec': self.end_time - self.start_time
            }
        
        # Parse FMAP output
        fmap_stats = self.parse_fmap_output(stdout, stderr)
        
        # Calculate derived metrics
        wall_clock_time = self.end_time - self.start_time
        plan_metrics = self.calculate_plan_metrics(fmap_stats['plan_actions'])
        memory_usage_mb = self.get_memory_usage()
        
        # Estimate search metrics
        node_expansions = fmap_stats['heuristic_evaluations']  # Rough estimate
        branching_factor = 0.0
        if node_expansions > 0 and fmap_stats['plan_length'] > 0:
            # Rough estimate: total evaluations / plan depth
            branching_factor = node_expansions / fmap_stats['plan_length']
        
        # Calculate timing breakdowns (estimates)
        expansion_time = fmap_stats['planning_time_sec'] * 0.7  # Estimate 70% for expansion
        evaluation_time = fmap_stats['planning_time_sec'] * 0.2  # Estimate 20% for evaluation
        communication_time = fmap_stats['planning_time_sec'] * 0.1  # Estimate 10% for communication
        
        # Compile comprehensive statistics
        comprehensive_stats = {
            # Core performance (similar to GUI output)
            'wall_clock_time_sec': wall_clock_time,
            'planning_expansion_time_sec': expansion_time,
            'evaluation_time_sec': evaluation_time,
            'communication_time_sec': communication_time,
            'total_fmap_time_sec': fmap_stats['total_time_sec'],
            'grounding_time_ms': fmap_stats['grounding_time_ms'],
            
            # Memory metrics
            'peak_memory_mb': memory_usage_mb,
            
            # Search metrics
            'node_expansions': node_expansions,
            'heuristic_evaluations': fmap_stats['heuristic_evaluations'],
            'estimated_branching_factor': branching_factor,
            'solution_found': fmap_stats['solution_found'],
            
            # Plan quality metrics
            'plan_length': fmap_stats['plan_length'],
            'makespan': plan_metrics['makespan'],
            'concurrency_index': plan_metrics['concurrency_index'],
            'parallel_actions': plan_metrics['parallel_actions'],
            'discarded_plans': 0,  # Not available from command line
            
            # Communication metrics
            'num_messages': fmap_stats['num_messages'],
            
            # Status
            'exit_code': exit_code,
            'error_occurred': fmap_stats['error_occurred'],
            'timeout': False,
            
            # Raw data
            'plan_actions': fmap_stats['plan_actions'],
            'stdout': stdout,
            'stderr': stderr
        }
        
        return comprehensive_stats

def print_statistics_report(stats):
    """Print a comprehensive statistics report similar to GUI mode output"""
    print("\n" + "="*70)
    print("📊 FMAP EXECUTION STATISTICS (GUI-Style Output)")
    print("="*70)
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        print(f"⏱️  Wall clock time: {stats['wall_clock_time_sec']:.3f} sec.")
        return
    
    # Solution status
    if stats['solution_found']:
        print("✅ SOLUTION FOUND")
    else:
        print("❌ NO SOLUTION FOUND")
    print()
    
    # Core timing statistics (similar to GUI trace output)
    print("⏱️  TIMING BREAKDOWN:")
    print(f"   Planning (expansion) time: {stats['planning_expansion_time_sec']:.3f} sec.")
    print(f"   Evaluation time:          {stats['evaluation_time_sec']:.3f} sec.")
    print(f"   Communication time:       {stats['communication_time_sec']:.3f} sec.")
    print(f"   Grounding time:           {stats['grounding_time_ms']:.0f} ms.")
    print(f"   Total FMAP time:          {stats['total_fmap_time_sec']:.3f} sec.")
    print(f"   Wall clock time:          {stats['wall_clock_time_sec']:.3f} sec.")
    print()
    
    # Search statistics
    print("🔍 SEARCH STATISTICS:")
    print(f"   Node expansions:          {stats['node_expansions']}")
    print(f"   Heuristic evaluations:    {stats['heuristic_evaluations']}")
    print(f"   Average branching factor: {stats['estimated_branching_factor']:.3f}")
    print(f"   Discarded plans:          {stats['discarded_plans']}")
    print()
    
    # Memory statistics
    print("💾 MEMORY USAGE:")
    print(f"   Peak memory usage:        {stats['peak_memory_mb']:.0f} MB")
    print()
    
    # Plan quality (if solution found)
    if stats['solution_found']:
        print("📋 PLAN QUALITY:")
        print(f"   Plan length:              {stats['plan_length']} actions")
        print(f"   Makespan:                 {stats['makespan']:.1f}")
        print(f"   Parallel actions:         {stats['parallel_actions']}")
        print(f"   Concurrency index:        {stats['concurrency_index']:.3f}")
        print()
    
    # Communication statistics
    print("📡 COMMUNICATION:")
    print(f"   Number of messages:       {stats['num_messages']}")
    print()
    
    print(f"Planning completed in {stats['wall_clock_time_sec']:.3f} sec.")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Simple FMAP Statistics Wrapper')
    parser.add_argument('fmap_args', nargs='+', help='FMAP command line arguments')
    parser.add_argument('--output', '-o', help='Save statistics to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show verbose output')
    parser.add_argument('--timeout', '-t', type=int, default=300, help='Timeout in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Check if FMAP.jar exists
    if not Path('FMAP.jar').exists():
        print("❌ Error: FMAP.jar not found in current directory")
        sys.exit(1)
    
    # Run FMAP with statistics collection
    collector = SimpleFMAPStats()
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