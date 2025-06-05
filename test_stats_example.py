#!/usr/bin/env python3
"""
Demo of FMAP Statistics Output
Shows what the statistics wrapper produces with sample successful FMAP output
"""

def demo_successful_run():
    """Simulate a successful FMAP run with realistic output"""
    
    # Sample FMAP output from a successful run
    sample_stdout = """
; Waiting to start
; Grounding completed in 45ms. (78 messages, 23 actions)
; Planning step: 1
; Hdtg = 4, Hlan = 0
; Planning step: 2
; Hdtg = 3, Hlan = 0
; Planning step: 3
; Hdtg = 2, Hlan = 0
; Planning step: 4
; Hdtg = 1, Hlan = 0
; Planning step: 5
; Hdtg = 0, Hlan = 0

; Solution plan - CoDMAP Distributed format
; -----------------------------------------
0.0: (drive-truck truck1 s2 s0 driver1)
1.0: (load-truck obj11 truck1 s0)
2.0: (drive-truck truck1 s0 s1 driver1)
3.0: (unload-truck obj11 truck1 s1)
0.0: (drive-truck truck2 s0 s1 driver2)
1.0: (load-truck obj21 truck2 s1)
2.0: (drive-truck truck2 s1 s2 driver2)
3.0: (unload-truck obj21 truck2 s2)

; Planning time: 0.654 sec.
; Plan length: 8
Number of messages: 89
Total time: 0.721 sec.
"""

    # Simulate statistics calculation
    from simple_fmap_stats import SimpleFMAPStats
    
    collector = SimpleFMAPStats()
    stats = collector.parse_fmap_output(sample_stdout, "")
    
    # Simulate timing
    stats_full = {
        'wall_clock_time_sec': 0.721,
        'planning_expansion_time_sec': 0.654 * 0.7,  # 70% for expansion
        'evaluation_time_sec': 0.654 * 0.2,  # 20% for evaluation  
        'communication_time_sec': 0.654 * 0.1,  # 10% for communication
        'total_fmap_time_sec': 0.721,
        'grounding_time_ms': 45,
        'peak_memory_mb': 128,
        'node_expansions': stats['heuristic_evaluations'],
        'heuristic_evaluations': stats['heuristic_evaluations'],
        'estimated_branching_factor': stats['heuristic_evaluations'] / max(1, stats['plan_length']),
        'solution_found': stats['solution_found'],
        'plan_length': stats['plan_length'],
        'makespan': 3.0,  # From plan actions
        'concurrency_index': 0.0,  # No parallel actions in this example
        'parallel_actions': 0,
        'discarded_plans': 0,
        'num_messages': stats['num_messages'],
        'exit_code': 0,
        'error_occurred': False,
        'timeout': False
    }
    
    return stats_full

def print_gui_style_output(stats):
    """Print output in the style similar to GUI trace messages"""
    print("="*70)
    print("📊 FMAP GUI-STYLE STATISTICS OUTPUT")
    print("="*70)
    print()
    print("✅ Solution found!")
    print()
    print(f"Planning (expansion) time: {stats['planning_expansion_time_sec']:.3f} sec.")
    print(f"Evaluation time: {stats['evaluation_time_sec']:.3f} sec.")
    print(f"Communication time: {stats['communication_time_sec']:.3f} sec.")
    print(f"Average branching factor: {stats['estimated_branching_factor']:.3f}")
    print(f"Discarded plans: {stats['discarded_plans']}")
    print(f"Planning completed in {stats['wall_clock_time_sec']:.3f} sec.")
    print(f"Used memory: {stats['peak_memory_mb']:.0f}kb.")
    print(f"Plan length: {stats['plan_length']}")
    print(f"Number of messages: {stats['num_messages']}")
    print(f"Total time: {stats['total_fmap_time_sec']:.3f} sec.")
    print()
    print("="*70)

if __name__ == "__main__":
    print("🎯 DEMONSTRATION: GUI-Style Statistics from Command Line")
    print("\nThis shows what you would get from a successful FMAP run:")
    
    stats = demo_successful_run()
    print_gui_style_output(stats)
    
    print("\n💡 HOW TO USE:")
    print("python3 simple_fmap_stats.py -- [your FMAP arguments here]")
    print("\nExample:")
    print("python3 simple_fmap_stats.py -- driver1 domain.pddl problem1.pddl driver2 domain.pddl problem2.pddl agents.txt -h 1") 