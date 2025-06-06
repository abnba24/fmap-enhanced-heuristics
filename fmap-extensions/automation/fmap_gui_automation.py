#!/usr/bin/env python3
"""
FMAP GUI Automation System
Comprehensive automation for testing heuristic performance via GUI interface
"""

import os
import subprocess
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyautogui
import re
import json
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple
import psutil
import threading
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fmap_gui_automation.log'),
        logging.StreamHandler()
    ]
)

class FMAPCommandLineAutomation:
    """FMAP Command Line Automation for Heuristic Performance Testing"""
    
    def __init__(self, results_dir="gui_automation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize data tracking
        self.experiment_data = []
        self.current_experiment = 0
        self.total_experiments = 0
        
        # Heuristic mapping
        self.heuristics = {
            0: "FF",
            1: "DTG", 
            2: "DTG + Landmarks",
            3: "Inc. DTG + Landmarks",
            4: "Centroids",
            5: "MCS"
        }
        
        # Experimental setup - using command line execution
        self.experiment_matrix = [
            # Format: (domain, problem, agents, complexity, command_template)
            ("driverlog", "Pfile1", 2, "EASY", 
             "java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h {H}"),
            
            ("driverlog", "Pfile2", 2, "MEDIUM",
             "java -jar FMAP.jar driver1 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile2/agents.txt -h {H}"),
            
            ("elevators", "Pfile1", 3, "EASY",
             "java -jar FMAP.jar fast0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsfast0.pddl slow0-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow0-0.pddl slow1-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow1-0.pddl Domains/elevators/Pfile1/agents.txt -h {H}"),
            
            ("ma-blocksworld", "Pfile4-0", 2, "MEDIUM",
             "java -jar FMAP.jar arm1 Domains/ma-blocksworld/Pfile4-0/Domainma-blocksworld.pddl Domains/ma-blocksworld/Pfile4-0/Problemma-blocksworldarm1.pddl arm2 Domains/ma-blocksworld/Pfile4-0/Domainma-blocksworld.pddl Domains/ma-blocksworld/Pfile4-0/Problemma-blocksworldarm2.pddl Domains/ma-blocksworld/Pfile4-0/agents.txt -h {H}")
        ]
        
        self.total_experiments = len(self.experiment_matrix) * len(self.heuristics)
        
        print(f"FMAP Command Line Automation Initialized!")
        print(f"Total Experiments: {self.total_experiments}")
        print(f"🗂️  Results Directory: {self.results_dir}")

    def execute_fmap_command(self, command: str, timeout: int = 60) -> Dict:
        """Execute FMAP command and capture output with timeout"""
        start_time = time.time()
        
        try:
            # Run command with timeout
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout} seconds',
                'execution_time': timeout,
                'return_code': -1,
                'timeout': True
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'execution_time': 0,
                'return_code': -1,
                'error': str(e)
            }

    def extract_metrics_from_output(self, output: str) -> Dict:
        """Extract metrics from FMAP command output"""
        metrics = {
            'plan_found': False,
            'plan_length': 0,
            'search_nodes': 0,
            'heuristic_calls': 0,
            'solution_plan': [],
            'heuristic_values': []
        }
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for solution plan
            if 'Solution plan' in line:
                metrics['plan_found'] = True
            
            # Extract plan steps
            if re.match(r'^\d+:', line):
                metrics['solution_plan'].append(line)
                metrics['plan_length'] += 1
            
            # Extract heuristic values
            if 'Hdtg' in line or 'Hlan' in line:
                metrics['heuristic_calls'] += 1
                metrics['heuristic_values'].append(line)
        
        return metrics

    def run_single_experiment(self, domain: str, problem: str, agents: int, 
                            complexity: str, command_template: str, heuristic_id: int) -> Dict:
        """Run a single experiment with specified parameters"""
        
        print(f"\n🔬 Experiment {self.current_experiment + 1}/{self.total_experiments}")
        print(f"   Domain: {domain}")
        print(f"   Problem: {problem}")  
        print(f"   Heuristic: {self.heuristics[heuristic_id]} (h={heuristic_id})")
        print(f"   Complexity: {complexity}")
        
        # Prepare command
        command = command_template.format(H=heuristic_id)
        print(f"   Command: {command}")
        
        # Execute experiment
        start_time = time.time()
        result = self.execute_fmap_command(command, timeout=120)
        end_time = time.time()
        
        # Extract metrics
        metrics = self.extract_metrics_from_output(result.get('stdout', ''))
        
        # Prepare experiment record
        experiment_record = {
            'experiment_id': self.current_experiment + 1,
            'timestamp': datetime.now().isoformat(),
            'domain': domain,
            'problem': problem,
            'agents': agents,
            'complexity': complexity,
            'heuristic_id': heuristic_id,
            'heuristic_name': self.heuristics[heuristic_id],
            'command': command,
            'execution_time': result.get('execution_time', 0),
            'success': result.get('success', False),
            'return_code': result.get('return_code', -1),
            'plan_found': metrics['plan_found'],
            'plan_length': metrics['plan_length'],
            'heuristic_calls': metrics['heuristic_calls'],
            'stdout': result.get('stdout', ''),
            'stderr': result.get('stderr', ''),
            'timeout': result.get('timeout', False)
        }
        
        # Save individual experiment result
        experiment_file = self.results_dir / f"experiment_{self.current_experiment + 1:03d}.json"
        with open(experiment_file, 'w') as f:
            json.dump(experiment_record, f, indent=2)
        
        print(f"    Execution Time: {result.get('execution_time', 0):.2f}s")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Plan Found: {metrics['plan_found']}")
        print(f"   📏 Plan Length: {metrics['plan_length']}")
        
        self.experiment_data.append(experiment_record)
        self.current_experiment += 1
        
        return experiment_record

    def run_comprehensive_experiments(self):
        """Run all experiments in the matrix across all heuristics"""
        
        print(f"\nStarting Comprehensive FMAP Heuristic Experiments")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run experiments for each domain/problem combination
        for domain, problem, agents, complexity, command_template in self.experiment_matrix:
            
            print(f"\n📂 Testing Domain: {domain} - {problem}")
            print(f"   Agents: {agents}, Complexity: {complexity}")
            
            # Test each heuristic
            for heuristic_id in self.heuristics.keys():
                try:
                    self.run_single_experiment(
                        domain, problem, agents, complexity, 
                        command_template, heuristic_id
                    )
                    
                    # Brief pause between experiments
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"   Error in experiment: {e}")
                    continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nAll Experiments Completed!")
        print(f" Total Time: {total_time:.2f} seconds")
        print(f"Total Experiments: {len(self.experiment_data)}")
        
        # Save comprehensive results
        self.save_comprehensive_results()
        self.generate_analysis_report()

    def save_comprehensive_results(self):
        """Save all experiment results to CSV and JSON"""
        
        # Create DataFrame
        df = pd.DataFrame(self.experiment_data)
        
        # Save to CSV
        csv_file = self.results_dir / "comprehensive_results.csv"
        df.to_csv(csv_file, index=False)
        
        # Save to JSON
        json_file = self.results_dir / "comprehensive_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2)
        
        print(f"Results saved to:")
        print(f"   {csv_file}")
        print(f"   {json_file}")

    def generate_analysis_report(self):
        """Generate comprehensive analysis and visualizations"""
        
        if not self.experiment_data:
            print("No experiment data to analyze")
            return
        
        df = pd.DataFrame(self.experiment_data)
        
        print(f"\nGENERATING ANALYSIS REPORT")
        print(f"{'='*50}")
        
        # 1. Success Rate Analysis
        self.analyze_success_rates(df)
        
        # 2. Performance Analysis
        self.analyze_performance(df)
        
        # 3. Centroids Heuristic Focus
        self.analyze_centroids_heuristic(df)
        
        # 4. Generate Visualizations
        self.generate_visualizations(df)

    def analyze_success_rates(self, df: pd.DataFrame):
        """Analyze success rates by heuristic"""
        
        print(f"\nSUCCESS RATE ANALYSIS")
        print(f"-" * 30)
        
        success_by_heuristic = df.groupby('heuristic_name')['success'].agg(['count', 'sum', 'mean'])
        success_by_heuristic.columns = ['Total', 'Successful', 'Success_Rate']
        success_by_heuristic['Success_Rate'] = success_by_heuristic['Success_Rate'] * 100
        
        print(success_by_heuristic)
        
        # Save analysis
        analysis_file = self.results_dir / "success_rate_analysis.csv"
        success_by_heuristic.to_csv(analysis_file)

    def analyze_performance(self, df: pd.DataFrame):
        """Analyze performance metrics"""
        
        print(f"\nPERFORMANCE ANALYSIS")
        print(f"-" * 30)
        
        # Filter successful experiments
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("No successful experiments to analyze")
            return
        
        performance_stats = successful_df.groupby('heuristic_name').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'plan_length': ['mean', 'std', 'min', 'max'],
            'heuristic_calls': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        print(performance_stats)
        
        # Save analysis
        performance_file = self.results_dir / "performance_analysis.csv"
        performance_stats.to_csv(performance_file)

    def analyze_centroids_heuristic(self, df: pd.DataFrame):
        """Special analysis for Centroids heuristic (h=4)"""
        
        print(f"\nCENTROIDS HEURISTIC ANALYSIS (h=4)")
        print(f"-" * 40)
        
        centroids_df = df[df['heuristic_id'] == 4]
        
        if len(centroids_df) == 0:
            print("No Centroids experiments found")
            return
        
        print(f"Total Centroids Experiments: {len(centroids_df)}")
        print(f"Successful: {centroids_df['success'].sum()}")
        print(f"Success Rate: {centroids_df['success'].mean() * 100:.1f}%")
        
        if centroids_df['success'].any():
            successful_centroids = centroids_df[centroids_df['success'] == True]
            print(f" Average Execution Time: {successful_centroids['execution_time'].mean():.2f}s")
            print(f"📏 Average Plan Length: {successful_centroids['plan_length'].mean():.1f}")
            print(f"Average Heuristic Calls: {successful_centroids['heuristic_calls'].mean():.1f}")
        
        # Save Centroids analysis
        centroids_file = self.results_dir / "centroids_analysis.csv"
        centroids_df.to_csv(centroids_file, index=False)

    def generate_visualizations(self, df: pd.DataFrame):
        """Generate comprehensive visualizations"""
        
        print(f"\nGENERATING VISUALIZATIONS")
        print(f"-" * 30)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Success Rate Heatmap
        self.plot_success_rate_heatmap(df)
        
        # 2. Performance Comparison
        self.plot_performance_comparison(df)
        
        # 3. Centroids Focus
        self.plot_centroids_analysis(df)

    def plot_success_rate_heatmap(self, df: pd.DataFrame):
        """Generate success rate heatmap"""
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='success', 
            index='domain', 
            columns='heuristic_name', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Success Rate'})
        plt.title('Success Rate by Domain and Heuristic')
        plt.tight_layout()
        
        heatmap_file = self.results_dir / "success_rate_heatmap.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Success Rate Heatmap: {heatmap_file}")

    def plot_performance_comparison(self, df: pd.DataFrame):
        """Generate performance comparison plots"""
        
        successful_df = df[df['success'] == True]
        
        if len(successful_df) == 0:
            print("   ⚠️  No successful experiments for performance comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Execution Time
        sns.boxplot(data=successful_df, x='heuristic_name', y='execution_time', ax=axes[0,0])
        axes[0,0].set_title('Execution Time by Heuristic')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plan Length
        sns.boxplot(data=successful_df, x='heuristic_name', y='plan_length', ax=axes[0,1])
        axes[0,1].set_title('Plan Length by Heuristic')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Heuristic Calls
        sns.boxplot(data=successful_df, x='heuristic_name', y='heuristic_calls', ax=axes[1,0])
        axes[1,0].set_title('Heuristic Calls by Heuristic')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Success Rate by Domain
        success_by_domain = df.groupby(['domain', 'heuristic_name'])['success'].mean().reset_index()
        sns.barplot(data=success_by_domain, x='domain', y='success', hue='heuristic_name', ax=axes[1,1])
        axes[1,1].set_title('Success Rate by Domain')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        performance_file = self.results_dir / "performance_comparison.png"
        plt.savefig(performance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Performance Comparison: {performance_file}")

    def plot_centroids_analysis(self, df: pd.DataFrame):
        """Generate Centroids-specific analysis"""
        
        centroids_df = df[df['heuristic_id'] == 4]
        
        if len(centroids_df) == 0:
            print("   ⚠️  No Centroids experiments for analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Centroids success by domain
        centroids_success = centroids_df.groupby('domain')['success'].agg(['count', 'sum', 'mean'])
        axes[0].bar(centroids_success.index, centroids_success['mean'])
        axes[0].set_title('Centroids Heuristic Success Rate by Domain')
        axes[0].set_ylabel('Success Rate')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Centroids vs other heuristics
        comparison_data = df.groupby('heuristic_name')['success'].mean()
        axes[1].bar(comparison_data.index, comparison_data.values)
        axes[1].set_title('Success Rate Comparison: All Heuristics')
        axes[1].set_ylabel('Success Rate')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Highlight Centroids
        centroids_idx = list(comparison_data.index).index('Centroids')
        axes[1].patches[centroids_idx].set_color('red')
        axes[1].patches[centroids_idx].set_alpha(0.8)
        
        plt.tight_layout()
        
        centroids_file = self.results_dir / "centroids_analysis.png"
        plt.savefig(centroids_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Centroids Analysis: {centroids_file}")

def main():
    """Main execution function"""
    
    print("🤖 FMAP Command Line Automation System")
    print("="*50)
    print("Testing heuristic performance across multiple domains")
    print("Special focus on Centroids heuristic validation")
    print("="*50)
    
    # Initialize automation system
    automation = FMAPCommandLineAutomation()
    
    # Run comprehensive experiments
    automation.run_comprehensive_experiments()
    
    print(f"\nEXPERIMENT SUITE COMPLETED!")
    print(f"📁 Results saved in: {automation.results_dir}")
    
    return automation

if __name__ == "__main__":
    automation = main() 