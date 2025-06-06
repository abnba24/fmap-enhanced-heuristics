#!/usr/bin/env python3

import json
import glob
from collections import defaultdict

# Correct heuristic mapping from experiment_runner.py
heuristic_names = {1: 'DTG', 2: 'DTG+Landmarks', 3: 'Inc_DTG+Landmarks', 4: 'Centroids', 5: 'MCS'}

results = defaultdict(list)
for file in glob.glob('results/result_*.json'):
    with open(file) as f:
        data = json.load(f)
        h_id = data['config']['heuristic']
        h_name = heuristic_names.get(h_id, f'H{h_id}')
        success = data['search']['coverage']
        time = data['search']['wall_clock_time'] if success else 0
        memory = data['search']['peak_memory_mb'] if success else 0
        plan_length = data['plan']['plan_length'] if success else 0
        
        results[h_name].append({
            'success': success,
            'time': time,
            'memory': memory,
            'plan_length': plan_length
        })

print('Heuristic Performance Summary:')
print('-' * 80)
for h_name, data_list in results.items():
    total = len(data_list)
    successes = sum(1 for d in data_list if d['success'])
    success_rate = successes / total if total > 0 else 0
    
    if successes > 0:
        avg_time = sum(d['time'] for d in data_list if d['success']) / successes
        avg_memory = sum(d['memory'] for d in data_list if d['success']) / successes  
        avg_plan = sum(d['plan_length'] for d in data_list if d['success']) / successes
    else:
        avg_time = avg_memory = avg_plan = 0
    
    print(f'{h_name:20}: {success_rate:6.1%} success, {avg_time:8.2f}s avg time, {avg_memory:8.1f}MB avg memory, {avg_plan:6.1f} avg plan length') 