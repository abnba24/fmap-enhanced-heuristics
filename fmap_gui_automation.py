#!/usr/bin/env python3
"""FMAP GUI Automation Script
Automates FMAP.jar GUI experiments using pyautogui to compare heuristic performance.
Generated from AUTOMATION_AGENT_PROMPT.md instructions.
"""

import os
import re
import time
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyautogui

# Configure pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1.0

# Detect screen resolution
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
print(f"Screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Heuristic mapping
HEURISTICS = {
    0: "FF",
    1: "DTG",
    2: "DTG+Landmarks",
    3: "Inc.DTG+Landmarks",
    4: "Centroids",
    5: "MCS",
}

# Experiment matrix based on AUTOMATION_AGENT_PROMPT.md
EXPERIMENTS = [
    {
        "domain": "driverlog",
        "problem": "Pfile1",
        "agents": 2,
        "complexity": "EASY",
        "cmd": "java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h {H} -gui",
    },
    {
        "domain": "driverlog",
        "problem": "Pfile2",
        "agents": 2,
        "complexity": "MEDIUM",
        "cmd": "java -jar FMAP.jar driver1 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile2/agent-list.txt -h {H} -gui",
    },
    {
        "domain": "driverlog",
        "problem": "Pfile5",
        "agents": 3,
        "complexity": "HARD",
        "cmd": "java -jar FMAP.jar driver1 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver2.pddl driver3 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver3.pddl Domains/driverlog/Pfile5/agent-list.txt -h {H} -gui",
    },
    {
        "domain": "ma-blocksworld",
        "problem": "Pfile6-2",
        "agents": 4,
        "complexity": "MEDIUM",
        "cmd": "java -jar FMAP.jar r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h {H} -gui",
    },
    {
        "domain": "elevators",
        "problem": "Pfile1",
        "agents": 3,
        "complexity": "EASY",
        "cmd": "java -jar FMAP.jar fast0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsfast0.pddl slow0-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow0-0.pddl slow1-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow1-0.pddl Domains/elevators/Pfile1/agent-list.txt -h {H} -gui",
    },
]

# Output data columns
COLUMNS = [
    'domain', 'problem', 'agent_count', 'complexity',
    'heuristic_id', 'heuristic_name', 'timestamp',
    'success', 'timeout', 'error_message',
    'planning_time_sec', 'evaluation_time_sec', 'communication_time_sec',
    'total_time_sec', 'memory_mb', 'plan_length', 'messages',
    'branching_factor', 'discarded_plans',
    'search_efficiency', 'memory_per_agent', 'messages_per_second'
]

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

# Regex patterns for metric extraction
def extract_metrics_from_trace(trace_text: str) -> dict:
    patterns = {
        'planning_time_sec': r'Planning \(expansion\) time: ([\d.]+) sec\.',
        'evaluation_time_sec': r'Evaluation time: ([\d.]+) sec\.',
        'communication_time_sec': r'Communication time: ([\d.]+) sec\.',
        'branching_factor': r'Average branching factor: ([\d.]+)',
        'memory_mb': r'Used memory: (\d+)kb',
        'plan_length': r'Plan length: (\d+)',
        'messages': r'Number of messages: (\d+)',
        'total_time_sec': r'Total time: ([\d.]+) sec\.',
        'discarded_plans': r'Discarded plans: (\d+)',
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, trace_text)
        if match:
            value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
            metrics[key] = value
        else:
            metrics[key] = None
    return metrics

# Derived metrics

def calculate_derived_metrics(row: pd.Series) -> pd.Series:
    row['search_efficiency'] = row['plan_length'] / row['total_time_sec'] if row.get('total_time_sec') else 0
    row['memory_per_agent'] = row['memory_mb'] / row['agent_count'] if row.get('memory_mb') else None
    row['messages_per_second'] = row['messages'] / row['total_time_sec'] if row.get('total_time_sec') else 0
    return row

# Error handling

def handle_experiment_error(domain, problem, heuristic, error):
    return {
        'domain': domain,
        'problem': problem,
        'heuristic_id': heuristic,
        'heuristic_name': HEURISTICS.get(heuristic, str(heuristic)),
        'agent_count': None,
        'complexity': None,
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'timeout': False,
        'error_message': str(error),
    }

# Core experiment runner

def run_experiment(exp):
    results = []
    for heuristic_id, heuristic_name in HEURISTICS.items():
        cmd = exp['cmd'].format(H=heuristic_id)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Launching {exp['domain']} {exp['problem']} with {heuristic_name}...")
        try:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(3)

            # Activate FMAP window if possible
            windows = pyautogui.getWindowsWithTitle("FMAP")
            if windows:
                windows[0].activate()

            start_time = time.time()
            trace_text = ''
            success = False
            timeout = False

            # Wait up to 300 seconds for completion
            while time.time() - start_time < 300:
                out = process.stdout.readline().decode('utf-8', errors='ignore')
                trace_text += out
                if 'Planning completed' in out or 'Solution found' in out:
                    success = True
                    break
                if out == '' and process.poll() is not None:
                    break

            if process.poll() is None:
                timeout = True
                process.kill()

            metrics = extract_metrics_from_trace(trace_text)
            row = {
                'domain': exp['domain'],
                'problem': exp['problem'],
                'agent_count': exp['agents'],
                'complexity': exp['complexity'],
                'heuristic_id': heuristic_id,
                'heuristic_name': heuristic_name,
                'timestamp': timestamp,
                'success': success,
                'timeout': timeout,
                'error_message': None,
            }
            row.update(metrics)
            row = calculate_derived_metrics(pd.Series(row)).to_dict()

            screenshot_path = RESULTS_DIR / f"experiment_{exp['domain']}_{exp['problem']}_{heuristic_id}_{timestamp}.png"
            pyautogui.screenshot(str(screenshot_path))

            pyautogui.hotkey('alt', 'f4')
            time.sleep(2)
            subprocess.run(["pkill", "-f", "FMAP.jar"], capture_output=True)

            results.append(row)
        except Exception as e:
            subprocess.run(["pkill", "-f", "FMAP.jar"], capture_output=True)
            results.append(handle_experiment_error(exp['domain'], exp['problem'], heuristic_id, e))
    return results

# Visualization functions

def generate_visualizations(df: pd.DataFrame):
    import seaborn as sns
    import matplotlib.pyplot as plt

    success_matrix = df.pivot_table(values='success', index='heuristic_name', columns='domain', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(success_matrix, annot=True, cmap='RdYlGn', fmt='.2f', cbar_kws={'label': 'Success Rate'})
    plt.title('Heuristic Success Rate by Domain')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'success_rate_heatmap.png', dpi=300)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[df['success']==True], x='heuristic_name', y='total_time_sec')
    plt.yscale('log')
    plt.title('Execution Time Distribution by Heuristic')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'execution_time_comparison.png', dpi=300)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df[df['success']==True], x='heuristic_name', y='memory_mb', ci=95)
    plt.title('Memory Usage by Heuristic')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'memory_usage_comparison.png', dpi=300)

    plt.figure(figsize=(10, 6))
    for heuristic in df['heuristic_name'].unique():
        heuristic_data = df[df['heuristic_name'] == heuristic]
        plt.plot(heuristic_data['agent_count'], heuristic_data['total_time_sec'], marker='o', label=heuristic)
    plt.xlabel('Number of Agents')
    plt.ylabel('Total Time (sec)')
    plt.title('Scalability Analysis: Agents vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / 'scalability_analysis.png', dpi=300)

# Report generation

def generate_experiment_report(df: pd.DataFrame) -> str:
    report = f"""# FMAP Heuristic Performance Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total experiments: {len(df)}
- Success rate: {df['success'].mean():.1%}
- Domains tested: {df['domain'].nunique()}
- Heuristics compared: {df['heuristic_name'].nunique()}

## Key Findings
### Best Overall Heuristic: {df.groupby('heuristic_name')['success'].mean().idxmax()}
### Fastest Heuristic: {df[df['success']==True].groupby('heuristic_name')['total_time_sec'].mean().idxmin()}
### Most Memory Efficient: {df[df['success']==True].groupby('heuristic_name')['memory_mb'].mean().idxmin()}

## Centroids Heuristic Analysis
- Success rate: {df[df['heuristic_id']==4]['success'].mean():.1%}
- Average time: {df[(df['heuristic_id']==4) & (df['success']==True)]['total_time_sec'].mean():.3f} sec
- Status: {'WORKING CORRECTLY' if df[df['heuristic_id']==4]['success'].sum() > 0 else 'ISSUES DETECTED'}
"""
    return report

# Main automation flow

def main():
    all_results = []
    for exp in EXPERIMENTS:
        all_results.extend(run_experiment(exp))

    df = pd.DataFrame(all_results)
    df.to_csv('experiment_results.csv', index=False)

    summary_table = df.groupby('heuristic_name').agg({
        'success': 'mean',
        'total_time_sec': 'mean',
        'memory_mb': 'mean',
        'plan_length': 'mean',
        'messages': 'mean',
    }).round(3)
    summary_table.to_csv('performance_summary.csv')

    generate_visualizations(df)

    report = generate_experiment_report(df)
    with open('experiment_report.md', 'w') as f:
        f.write(report)

    print('Automation complete.')

if __name__ == '__main__':
    main()
