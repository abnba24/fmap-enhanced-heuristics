#!/usr/bin/env python3
"""
FMAP Heuristic Performance Analysis
Comprehensive evaluation of heuristics across multiple domains and problems.
"""

import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from collections import defaultdict
import numpy as np

class HeuristicExperimentRunner:
    def __init__(self):
        self.heuristics = {
            0: "FF",
            1: "DTG", 
            2: "DTG+Landmarks",
            3: "Inc.DTG+Landmarks", 
            4: "Centroids",
            5: "MCS"
        }
        
        self.domains = [
            "driverlog", "logistics", "rovers", "satellite", 
            "depots", "elevators", "openstacks", "woodworking", 
            "zenotravel", "ma-blocksworld"
        ]
        
        self.results = []
        self.timeout = 120  # 2 minutes per experiment
        
    def find_domain_problems(self, domain):
        """Find all available problems for a domain"""
        domain_path = f"Domains/{domain}"
        problems = []
        
        if not os.path.exists(domain_path):
            return problems
            
        for item in os.listdir(domain_path):
            item_path = os.path.join(domain_path, item)
            if os.path.isdir(item_path) and item.startswith("Pfile"):
                problems.append(item)
        
        return sorted(problems)[:5]  # Limit to first 5 problems per domain
    
    def detect_problem_structure(self, domain, problem):
        """Analyze problem structure to determine complexity and agent count"""
        problem_path = f"Domains/{domain}/{problem}"
        
        # Default structure
        structure = {
            "agents": 1,
            "domain_file": None,
            "problem_files": [],
            "agent_file": None,
            "complexity": "low"
        }
        
        try:
            files = os.listdir(problem_path)
            
            # Find domain file
            domain_files = [f for f in files if f.endswith('.pddl') and 'domain' in f.lower()]
            if domain_files:
                structure["domain_file"] = os.path.join(problem_path, domain_files[0])
            
            # Find problem files
            problem_files = [f for f in files if f.endswith('.pddl') and 'problem' in f.lower()]
            structure["problem_files"] = [os.path.join(problem_path, f) for f in problem_files]
            
            # Find agent configuration
            agent_files = [f for f in files if f in ['agents.txt', 'agent-list.txt']]
            if agent_files:
                structure["agent_file"] = os.path.join(problem_path, agent_files[0])
                structure["agents"] = len(structure["problem_files"])
            
            # Estimate complexity based on problem files
            if structure["problem_files"]:
                total_size = sum(os.path.getsize(f) for f in structure["problem_files"])
                if total_size > 3000:
                    structure["complexity"] = "high"
                elif total_size > 1500:
                    structure["complexity"] = "medium"
                    
        except Exception as e:
            print(f"Error analyzing {domain}/{problem}: {e}")
            
        return structure
    
    def run_single_experiment(self, domain, problem, heuristic_id, structure):
        """Run a single experiment and parse results"""
        print(f"Running {domain}/{problem} with {self.heuristics[heuristic_id]} heuristic...")
        
        # Prepare command based on domain structure
        if structure["agents"] > 1 and structure["agent_file"]:
            # Multi-agent setup
            cmd = ["java", "-jar", "FMAP.jar"]
            
            # Add agent configurations
            for i, pfile in enumerate(structure["problem_files"]):
                agent_name = f"agent{i+1}"
                cmd.extend([agent_name, structure["domain_file"], pfile])
            
            cmd.extend([structure["agent_file"], "-h", str(heuristic_id)])
        else:
            # Single-agent setup (most domains)
            if not structure["domain_file"] or not structure["problem_files"]:
                return None
                
            cmd = ["java", "-jar", "FMAP.jar", 
                   "-domain", structure["domain_file"],
                   "-problem", structure["problem_files"][0],
                   "-h", str(heuristic_id)]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.timeout,
                cwd=os.getcwd()
            )
            execution_time = time.time() - start_time
            
            # Parse output
            metrics = self.parse_experiment_output(result.stdout + result.stderr, execution_time)
            metrics.update({
                "domain": domain,
                "problem": problem,
                "heuristic_id": heuristic_id,
                "heuristic_name": self.heuristics[heuristic_id],
                "agents": structure["agents"],
                "complexity": structure["complexity"],
                "success": result.returncode == 0
            })
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {
                "domain": domain,
                "problem": problem, 
                "heuristic_id": heuristic_id,
                "heuristic_name": self.heuristics[heuristic_id],
                "agents": structure["agents"],
                "complexity": structure["complexity"],
                "success": False,
                "timeout": True,
                "execution_time": self.timeout
            }
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None
    
    def parse_experiment_output(self, output, execution_time):
        """Parse FMAP output to extract metrics"""
        metrics = {
            "execution_time": execution_time,
            "plan_length": 0,
            "nodes_expanded": 0,
            "heuristic_evaluations": 0,
            "solution_found": False,
            "plan_cost": 0,
            "makespan": 0,
            "timeout": False
        }
        
        lines = output.split('\n')
        
        # Look for solution plan
        solution_lines = []
        in_solution = False
        
        for line in lines:
            line = line.strip()
            
            # Check for solution start
            if "Solution plan" in line or "CoDMAP" in line:
                in_solution = True
                continue
                
            # Count plan steps
            if in_solution and re.match(r'^\d+:', line):
                solution_lines.append(line)
                metrics["plan_length"] += 1
                
            # Check for stopping condition
            if "Stopping" in line:
                in_solution = False
                
            # Count heuristic evaluations (look for Hdtg patterns)
            if re.search(r'Hdtg\s*=\s*\d+', line):
                metrics["heuristic_evaluations"] += 1
                
            # Look for expanded nodes or search statistics
            if "nodes" in line.lower() and "expanded" in line.lower():
                numbers = re.findall(r'\d+', line)
                if numbers:
                    metrics["nodes_expanded"] = int(numbers[0])
        
        # Determine if solution was found
        if metrics["plan_length"] > 0:
            metrics["solution_found"] = True
            metrics["plan_cost"] = metrics["plan_length"]  # Assuming unit cost actions
            
        return metrics
    
    def run_all_experiments(self):
        """Run experiments across all domains and heuristics"""
        total_experiments = 0
        completed_experiments = 0
        
        for domain in self.domains:
            problems = self.find_domain_problems(domain)
            print(f"\n=== Domain: {domain} ({len(problems)} problems) ===")
            
            for problem in problems:
                structure = self.detect_problem_structure(domain, problem)
                print(f"Problem {problem}: {structure['agents']} agents, {structure['complexity']} complexity")
                
                for heuristic_id in self.heuristics.keys():
                    total_experiments += 1
                    result = self.run_single_experiment(domain, problem, heuristic_id, structure)
                    
                    if result:
                        self.results.append(result)
                        completed_experiments += 1
                        
                        if result.get("success", False):
                            print(f"  ✓ {self.heuristics[heuristic_id]}: {result.get('plan_length', 0)} steps")
                        else:
                            print(f"  ✗ {self.heuristics[heuristic_id]}: Failed")
        
        print(f"\n=== Experiments Completed: {completed_experiments}/{total_experiments} ===")
        return self.results
    
    def save_results(self, filename="heuristic_experiment_results.json"):
        """Save results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def load_results(self, filename="heuristic_experiment_results.json"):
        """Load results from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                self.results = json.load(f)
            print(f"Results loaded from {filename}")
            return True
        return False

def create_visualizations(results):
    """Create comprehensive visualizations and analysis"""
    df = pd.DataFrame(results)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Success Rate by Heuristic
    plt.subplot(4, 3, 1)
    success_rates = df.groupby('heuristic_name')['solution_found'].mean().sort_values(ascending=False)
    success_rates.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Success Rate by Heuristic')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Average Execution Time by Heuristic
    plt.subplot(4, 3, 2)
    avg_times = df[df['solution_found'] == True].groupby('heuristic_name')['execution_time'].mean().sort_values()
    avg_times.plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Average Execution Time (Successful Solutions)')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 3. Average Plan Length by Heuristic
    plt.subplot(4, 3, 3)
    avg_lengths = df[df['solution_found'] == True].groupby('heuristic_name')['plan_length'].mean().sort_values()
    avg_lengths.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Average Plan Length (Successful Solutions)')
    plt.ylabel('Plan Length')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Performance by Domain (Heatmap)
    plt.subplot(4, 3, 4)
    domain_heuristic = df.pivot_table(values='solution_found', index='domain', columns='heuristic_name', aggfunc='mean')
    sns.heatmap(domain_heuristic, annot=True, cmap='YlOrRd', cbar_kws={'label': 'Success Rate'})
    plt.title('Success Rate by Domain and Heuristic')
    plt.xlabel('Heuristic')
    plt.ylabel('Domain')
    
    # 5. Execution Time by Complexity
    plt.subplot(4, 3, 5)
    complexity_data = df[df['solution_found'] == True]
    sns.boxplot(data=complexity_data, x='complexity', y='execution_time', hue='heuristic_name')
    plt.title('Execution Time by Problem Complexity')
    plt.ylabel('Time (seconds)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Plan Quality vs Speed Trade-off
    plt.subplot(4, 3, 6)
    successful_df = df[df['solution_found'] == True]
    for heuristic in successful_df['heuristic_name'].unique():
        heur_data = successful_df[successful_df['heuristic_name'] == heuristic]
        plt.scatter(heur_data['execution_time'], heur_data['plan_length'], 
                   label=heuristic, alpha=0.7, s=50)
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Plan Length')
    plt.title('Plan Quality vs Speed Trade-off')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 7. Performance by Number of Agents
    plt.subplot(4, 3, 7)
    agent_perf = df.groupby(['agents', 'heuristic_name'])['solution_found'].mean().unstack()
    agent_perf.plot(kind='bar', width=0.8)
    plt.title('Success Rate by Number of Agents')
    plt.xlabel('Number of Agents')
    plt.ylabel('Success Rate')
    plt.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # 8. Heuristic Evaluations vs Solution Quality
    plt.subplot(4, 3, 8)
    eval_data = df[df['solution_found'] == True]
    sns.scatterplot(data=eval_data, x='heuristic_evaluations', y='plan_length', 
                    hue='heuristic_name', style='complexity', s=100)
    plt.title('Heuristic Evaluations vs Plan Length')
    plt.xlabel('Number of Heuristic Evaluations')
    plt.ylabel('Plan Length')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 9. Domain Difficulty Analysis
    plt.subplot(4, 3, 9)
    domain_difficulty = df.groupby('domain')['solution_found'].mean().sort_values()
    domain_difficulty.plot(kind='barh', color='orange', edgecolor='black')
    plt.title('Domain Difficulty (Overall Success Rate)')
    plt.xlabel('Success Rate')
    plt.grid(axis='x', alpha=0.3)
    
    # 10. Timeout Rate by Heuristic
    plt.subplot(4, 3, 10)
    timeout_rates = df.groupby('heuristic_name')['timeout'].mean().sort_values(ascending=False)
    timeout_rates.plot(kind='bar', color='red', alpha=0.7, edgecolor='black')
    plt.title('Timeout Rate by Heuristic')
    plt.ylabel('Timeout Rate')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 11. Performance Distribution
    plt.subplot(4, 3, 11)
    successful_df = df[df['solution_found'] == True]
    plt.hist([successful_df[successful_df['heuristic_name'] == h]['execution_time'] 
              for h in successful_df['heuristic_name'].unique()], 
             label=successful_df['heuristic_name'].unique(), alpha=0.7, bins=15)
    plt.title('Execution Time Distribution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 12. Summary Statistics Table
    plt.subplot(4, 3, 12)
    plt.axis('off')
    summary_stats = df.groupby('heuristic_name').agg({
        'solution_found': ['mean', 'count'],
        'execution_time': 'mean',
        'plan_length': 'mean'
    }).round(3)
    
    table_data = []
    for heuristic in summary_stats.index:
        row = [
            heuristic,
            f"{summary_stats.loc[heuristic, ('solution_found', 'mean')]:.3f}",
            f"{summary_stats.loc[heuristic, ('solution_found', 'count')]}",
            f"{summary_stats.loc[heuristic, ('execution_time', 'mean')]:.2f}s",
            f"{summary_stats.loc[heuristic, ('plan_length', 'mean')]:.1f}"
        ]
        table_data.append(row)
    
    table = plt.table(cellText=table_data, 
                     colLabels=['Heuristic', 'Success Rate', 'Total Tests', 'Avg Time', 'Avg Length'],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    plt.title('Summary Statistics', y=0.9)
    
    plt.tight_layout()
    plt.savefig('heuristic_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_matrix(results):
    """Create a comprehensive performance matrix"""
    df = pd.DataFrame(results)
    
    # Create detailed performance matrix
    metrics = ['solution_found', 'execution_time', 'plan_length', 'heuristic_evaluations']
    
    # Performance by Domain and Heuristic
    performance_matrix = {}
    
    for metric in metrics:
        matrix_data = df.pivot_table(
            values=metric, 
            index='domain', 
            columns='heuristic_name', 
            aggfunc='mean' if metric != 'solution_found' else 'mean'
        )
        performance_matrix[metric] = matrix_data
    
    # Save matrices to CSV
    for metric, matrix in performance_matrix.items():
        matrix.to_csv(f'performance_matrix_{metric}.csv')
        print(f"Saved {metric} matrix to performance_matrix_{metric}.csv")
    
    return performance_matrix

def generate_report(results):
    """Generate a comprehensive analysis report"""
    df = pd.DataFrame(results)
    
    report = f"""
# FMAP Heuristic Performance Analysis Report
## Experimental Setup
- **Total Experiments**: {len(df)}
- **Domains Tested**: {df['domain'].nunique()}
- **Heuristics Evaluated**: {df['heuristic_name'].nunique()}
- **Timeout Limit**: 120 seconds per experiment

## Overall Performance Summary

### Success Rates by Heuristic:
"""
    
    success_rates = df.groupby('heuristic_name')['solution_found'].agg(['mean', 'count'])
    for heuristic, stats in success_rates.iterrows():
        report += f"- **{heuristic}**: {stats['mean']:.1%} ({int(stats['mean'] * stats['count'])}/{stats['count']} solved)\n"
    
    report += f"""
### Key Findings:

**Best Performing Heuristics:**
"""
    
    # Best heuristics analysis
    best_success = success_rates['mean'].idxmax()
    best_time = df[df['solution_found'] == True].groupby('heuristic_name')['execution_time'].mean().idxmin()
    best_quality = df[df['solution_found'] == True].groupby('heuristic_name')['plan_length'].mean().idxmin()
    
    report += f"""
1. **Highest Success Rate**: {best_success} ({success_rates.loc[best_success, 'mean']:.1%})
2. **Fastest Execution**: {best_time} 
3. **Best Plan Quality**: {best_quality}

### Domain Analysis:
"""
    
    domain_analysis = df.groupby('domain').agg({
        'solution_found': 'mean',
        'execution_time': 'mean',
        'plan_length': 'mean'
    }).round(3)
    
    hardest_domain = domain_analysis['solution_found'].idxmin()
    easiest_domain = domain_analysis['solution_found'].idxmax()
    
    report += f"""
- **Easiest Domain**: {easiest_domain} ({domain_analysis.loc[easiest_domain, 'solution_found']:.1%} success rate)
- **Hardest Domain**: {hardest_domain} ({domain_analysis.loc[hardest_domain, 'solution_found']:.1%} success rate)

### Multi-Agent Performance:
"""
    
    agent_analysis = df.groupby('agents')['solution_found'].mean()
    report += f"""
- **Single-Agent Problems**: {agent_analysis.get(1, 0):.1%} success rate
- **Multi-Agent Problems**: {agent_analysis.get(2, 0):.1%} success rate

## Detailed Recommendations:

1. **For Maximum Success Rate**: Use {best_success}
2. **For Speed-Critical Applications**: Use {best_time}  
3. **For Optimal Plan Quality**: Use {best_quality}
4. **For Complex Domains**: Consider domain-specific analysis

## Statistical Significance:
- All results based on {len(df)} total experiments
- Timeout rate: {df['timeout'].mean():.1%}
- Average execution time: {df[df['solution_found'] == True]['execution_time'].mean():.2f}s
"""
    
    # Save report
    with open('heuristic_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("Report saved to heuristic_analysis_report.md")
    return report

if __name__ == "__main__":
    print("🚀 Starting FMAP Heuristic Performance Analysis...")
    
    runner = HeuristicExperimentRunner()
    
    # Try to load existing results first
    if not runner.load_results():
        print("No existing results found. Running new experiments...")
        results = runner.run_all_experiments()
        runner.save_results()
    else:
        print("Using existing results...")
        results = runner.results
    
    if results:
        print(f"\n📊 Analyzing {len(results)} experimental results...")
        
        # Create visualizations
        create_visualizations(results)
        
        # Create performance matrices  
        matrices = create_performance_matrix(results)
        
        # Generate comprehensive report
        report = generate_report(results)
        
        print("\n✅ Analysis complete! Check the following files:")
        print("- heuristic_performance_analysis.png (comprehensive visualizations)")
        print("- performance_matrix_*.csv (detailed performance matrices)")
        print("- heuristic_analysis_report.md (comprehensive report)")
        print("- heuristic_experiment_results.json (raw results)")
    else:
        print("❌ No results to analyze!") 