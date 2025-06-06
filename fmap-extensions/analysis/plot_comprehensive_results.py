#!/usr/bin/env python3
"""
Comprehensive Results Plotting Script for FMAP Experiments

This script generates various plots and analysis charts from the comprehensive
experiment results, suitable for research reports and presentations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveResultsPlotter:
    """Generates comprehensive plots from experiment results"""
    
    def __init__(self, results_file, output_dir="experiments/plots"):
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.df = pd.read_csv(results_file)
        print(f"Loaded {len(self.df)} experiment results")
        
        # Add derived columns
        self.df['complexity'] = self.df.apply(self._categorize_complexity, axis=1)
        self.df['success_rate'] = self.df['plan_found'].astype(int)
        
    def _categorize_complexity(self, row):
        """Categorize problem complexity based on agent count"""
        agent_count = row['agent_count']
        if agent_count == 2:
            return "VERY_SMALL"
        elif agent_count == 3:
            return "SMALL"
        elif agent_count == 5:
            return "MEDIUM"
        elif agent_count >= 8:
            return "LARGE"
        else:
            return "OTHER"
    
    def plot_success_rates_by_heuristic(self):
        """Plot success rates for each heuristic"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall success rates
        success_by_heuristic = self.df.groupby('heuristic_name')['success_rate'].mean()
        
        ax1.bar(success_by_heuristic.index, success_by_heuristic.values)
        ax1.set_title('Overall Success Rate by Heuristic')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(success_by_heuristic.values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        
        # Success rates by complexity
        success_by_complexity = self.df.groupby(['complexity', 'heuristic_name'])['success_rate'].mean().unstack()
        success_by_complexity.plot(kind='bar', ax=ax2)
        ax2.set_title('Success Rate by Problem Complexity')
        ax2.set_ylabel('Success Rate')
        ax2.set_xlabel('Problem Complexity')
        ax2.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_execution_times(self):
        """Plot execution time analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Filter successful runs for timing analysis
        successful = self.df[self.df['plan_found'] == True]
        
        if len(successful) == 0:
            print("No successful runs found for timing analysis")
            return
        
        # Box plot by heuristic
        sns.boxplot(data=successful, x='heuristic_name', y='execution_time', ax=ax1)
        ax1.set_title('Execution Time Distribution by Heuristic')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Box plot by domain
        sns.boxplot(data=successful, x='domain', y='execution_time', ax=ax2)
        ax2.set_title('Execution Time Distribution by Domain')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Scatter plot: Agent count vs execution time
        for heuristic in successful['heuristic_name'].unique():
            heur_data = successful[successful['heuristic_name'] == heuristic]
            ax3.scatter(heur_data['agent_count'], heur_data['execution_time'], 
                       label=heuristic, alpha=0.7)
        
        ax3.set_xlabel('Agent Count')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Scalability: Agent Count vs Execution Time')
        ax3.legend()
        ax3.set_yscale('log')
        
        # Heatmap: Domain vs Heuristic (average time)
        pivot_time = successful.pivot_table(values='execution_time', 
                                           index='domain', 
                                           columns='heuristic_name', 
                                           aggfunc='mean')
        sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Average Execution Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'execution_times.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_plan_quality(self):
        """Plot plan quality metrics"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        successful = self.df[self.df['plan_found'] == True]
        
        if len(successful) == 0:
            print("No successful runs found for plan quality analysis")
            return
        
        # Plan length by heuristic
        sns.boxplot(data=successful, x='heuristic_name', y='plan_length', ax=ax1)
        ax1.set_title('Plan Length Distribution by Heuristic')
        ax1.set_ylabel('Plan Length (actions)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plan length by domain
        sns.boxplot(data=successful, x='domain', y='plan_length', ax=ax2)
        ax2.set_title('Plan Length Distribution by Domain')
        ax2.set_ylabel('Plan Length (actions)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plan length vs execution time
        for heuristic in successful['heuristic_name'].unique():
            heur_data = successful[successful['heuristic_name'] == heuristic]
            ax3.scatter(heur_data['plan_length'], heur_data['execution_time'], 
                       label=heuristic, alpha=0.7)
        
        ax3.set_xlabel('Plan Length (actions)')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_title('Plan Quality vs Performance')
        ax3.legend()
        ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plan_quality.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_domain_analysis(self):
        """Analyze performance by domain"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Success rate by domain
        domain_success = self.df.groupby('domain')['success_rate'].mean()
        domain_success.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Success Rate by Domain')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(domain_success.values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        
        # Agent count distribution by domain
        agent_counts = self.df.groupby('domain')['agent_count'].unique()
        domain_labels = []
        agent_count_values = []
        
        for domain, counts in agent_counts.items():
            for count in counts:
                domain_labels.append(f"{domain}\n({count} agents)")
                agent_count_values.append(count)
        
        domain_order = self.df.groupby('domain')['agent_count'].mean().sort_values().index
        
        sns.boxplot(data=self.df, x='domain', y='agent_count', order=domain_order, ax=ax2)
        ax2.set_title('Agent Count Distribution by Domain')
        ax2.set_ylabel('Agent Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Problem count by domain
        problem_counts = self.df.groupby('domain').size()
        problem_counts.plot(kind='bar', ax=ax3, color='lightgreen')
        ax3.set_title('Number of Problems Tested by Domain')
        ax3.set_ylabel('Problem Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # Success rate heatmap by domain and heuristic
        pivot_success = self.df.pivot_table(values='success_rate', 
                                           index='domain', 
                                           columns='heuristic_name', 
                                           aggfunc='mean')
        sns.heatmap(pivot_success, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Success Rate by Domain and Heuristic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_heuristic_comparison(self):
        """Comprehensive heuristic comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Performance radar chart data preparation
        heuristics = self.df['heuristic_name'].unique()
        metrics = {}
        
        for heuristic in heuristics:
            heur_data = self.df[self.df['heuristic_name'] == heuristic]
            successful_data = heur_data[heur_data['plan_found'] == True]
            
            metrics[heuristic] = {
                'Success Rate': heur_data['success_rate'].mean(),
                'Avg Time': successful_data['execution_time'].mean() if len(successful_data) > 0 else 0,
                'Avg Plan Length': successful_data['plan_length'].mean() if len(successful_data) > 0 else 0
            }
        
        # Bar chart comparison
        metrics_df = pd.DataFrame(metrics).T
        metrics_df['Success Rate'].plot(kind='bar', ax=ax1, color='lightblue')
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        
        # Execution time comparison (successful runs only)
        successful = self.df[self.df['plan_found'] == True]
        if len(successful) > 0:
            metrics_df['Avg Time'].plot(kind='bar', ax=ax2, color='orange')
            ax2.set_title('Average Execution Time (Successful Runs)')
            ax2.set_ylabel('Time (seconds)')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plan length comparison
        if len(successful) > 0:
            metrics_df['Avg Plan Length'].plot(kind='bar', ax=ax3, color='green')
            ax3.set_title('Average Plan Length')
            ax3.set_ylabel('Plan Length (actions)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Combined performance score
        if len(successful) > 0:
            # Normalize metrics (higher is better for success rate, lower is better for time and plan length)
            normalized_metrics = metrics_df.copy()
            normalized_metrics['Success Rate'] = normalized_metrics['Success Rate']  # Keep as is (0-1)
            normalized_metrics['Time Score'] = 1 / (1 + normalized_metrics['Avg Time'] / normalized_metrics['Avg Time'].max())
            normalized_metrics['Plan Score'] = 1 / (1 + normalized_metrics['Avg Plan Length'] / normalized_metrics['Avg Plan Length'].max())
            
            # Combined score (equal weights)
            normalized_metrics['Combined Score'] = (
                normalized_metrics['Success Rate'] * 0.5 + 
                normalized_metrics['Time Score'] * 0.25 + 
                normalized_metrics['Plan Score'] * 0.25
            )
            
            normalized_metrics['Combined Score'].plot(kind='bar', ax=ax4, color='purple')
            ax4.set_title('Combined Performance Score')
            ax4.set_ylabel('Score (higher is better)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heuristic_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_statistics(self):
        """Generate and save summary statistics"""
        summary = {
            'Total Experiments': len(self.df),
            'Successful Experiments': len(self.df[self.df['plan_found'] == True]),
            'Overall Success Rate': self.df['success_rate'].mean(),
            'Domains Tested': len(self.df['domain'].unique()),
            'Heuristics Tested': len(self.df['heuristic_name'].unique()),
            'Agent Count Range': f"{self.df['agent_count'].min()}-{self.df['agent_count'].max()}",
        }
        
        # By heuristic stats
        heuristic_stats = self.df.groupby('heuristic_name').agg({
            'success_rate': ['mean', 'count'],
            'execution_time': 'mean',
            'plan_length': 'mean'
        }).round(3)
        
        # By domain stats  
        domain_stats = self.df.groupby('domain').agg({
            'success_rate': ['mean', 'count'],
            'execution_time': 'mean',
            'agent_count': ['min', 'max']
        }).round(3)
        
        # Save to file
        stats_file = self.output_dir / 'summary_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("FMAP Comprehensive Experiment Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\n\nHEURISTIC PERFORMANCE:\n")
            f.write(heuristic_stats.to_string())
            
            f.write(f"\n\nDOMAIN PERFORMANCE:\n")
            f.write(domain_stats.to_string())
        
        print(f"Summary statistics saved to: {stats_file}")
        
        # Print key findings
        print("\n" + "="*50)
        print("KEY EXPERIMENT FINDINGS:")
        print("="*50)
        
        best_heuristic = self.df.groupby('heuristic_name')['success_rate'].mean().idxmax()
        best_success_rate = self.df.groupby('heuristic_name')['success_rate'].mean().max()
        
        print(f"🏆 Best Heuristic: {best_heuristic} (Success Rate: {best_success_rate:.2%})")
        
        successful = self.df[self.df['plan_found'] == True]
        if len(successful) > 0:
            fastest_heuristic = successful.groupby('heuristic_name')['execution_time'].mean().idxmin()
            fastest_time = successful.groupby('heuristic_name')['execution_time'].mean().min()
            print(f"Fastest Heuristic: {fastest_heuristic} (Avg Time: {fastest_time:.2f}s)")
        
        best_domain = self.df.groupby('domain')['success_rate'].mean().idxmax()
        best_domain_rate = self.df.groupby('domain')['success_rate'].mean().max()
        print(f"Most Reliable Domain: {best_domain} (Success Rate: {best_domain_rate:.2%})")
        
        print(f"Total Problems Tested: {len(self.df['problem'].unique())}")
        print(f"Agent Count Range: {self.df['agent_count'].min()}-{self.df['agent_count'].max()}")
        
    def generate_all_plots(self):
        """Generate all plots and analysis"""
        print("Generating comprehensive plots...")
        
        self.plot_success_rates_by_heuristic()
        self.plot_execution_times()
        self.plot_plan_quality()
        self.plot_domain_analysis()
        self.plot_heuristic_comparison()
        self.generate_summary_statistics()
        
        print(f"\nAll plots saved to: {self.output_dir}")
        print("Analysis complete! Ready for your report.")

def main():
    """Main plotting function"""
    parser = argparse.ArgumentParser(description='Generate comprehensive plots from FMAP experiment results')
    parser.add_argument('results_file', help='CSV file with experiment results')
    parser.add_argument('--output-dir', default='experiments/plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file {args.results_file} not found")
        sys.exit(1)
    
    plotter = ComprehensiveResultsPlotter(args.results_file, args.output_dir)
    plotter.generate_all_plots()

if __name__ == "__main__":
    main() 