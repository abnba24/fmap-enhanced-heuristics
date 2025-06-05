#!/usr/bin/env python3
"""
FMAP Experiment Results Visualizer

This module creates comprehensive visualizations for experimental results,
including performance comparisons, statistical analyses, and publication-quality plots.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentVisualizer:
    """Creates visualizations for FMAP experimental results"""
    
    def __init__(self, results_file: str = "results/all_results.json", output_dir: str = "plots"):
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.heuristic_colors = {
            'DTG': '#1f77b4',
            'DTG+Landmarks': '#ff7f0e', 
            'Inc_DTG+Landmarks': '#2ca02c',
            'Centroids': '#d62728',
            'MCS': '#9467bd'
        }
        
        self.results_df = None
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare experimental data"""
        from data_analyzer import ExperimentAnalyzer
        
        analyzer = ExperimentAnalyzer(self.results_file)
        self.results_df = analyzer.load_results()
        
        return self.results_df
    
    def plot_coverage_comparison(self) -> None:
        """Create coverage comparison plots"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        # Overall coverage by heuristic
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Coverage rates
        coverage_by_heuristic = self.results_df.groupby('heuristic_name')['coverage'].agg(['count', 'sum', 'mean'])
        coverage_by_heuristic['success_rate'] = coverage_by_heuristic['sum'] / coverage_by_heuristic['count']
        
        bars = ax1.bar(coverage_by_heuristic.index, coverage_by_heuristic['success_rate'],
                      color=[self.heuristic_colors.get(h, 'gray') for h in coverage_by_heuristic.index])
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Coverage Comparison Across Heuristics')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, coverage_by_heuristic['success_rate']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Coverage by domain
        coverage_by_domain = self.results_df.groupby(['domain', 'heuristic_name'])['coverage'].mean().unstack()
        coverage_by_domain.plot(kind='bar', ax=ax2, color=[self.heuristic_colors.get(h, 'gray') for h in coverage_by_domain.columns])
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Coverage by Domain')
        ax2.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'coverage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created coverage comparison plot")
    
    def plot_performance_comparison(self) -> None:
        """Create performance comparison plots"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        # Filter to solved instances only
        solved_df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(solved_df) == 0:
            logger.warning("No solved instances for performance plotting")
            return
        
        # Performance metrics plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Wall clock time
        if solved_df['wall_time'].sum() > 0:
            sns.boxplot(data=solved_df, x='heuristic_name', y='wall_time', ax=axes[0,0])
            axes[0,0].set_yscale('log')
            axes[0,0].set_ylabel('Wall Clock Time (seconds)')
            axes[0,0].set_title('Wall Clock Time Distribution')
            plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Peak memory
        if solved_df['peak_memory'].sum() > 0:
            sns.boxplot(data=solved_df, x='heuristic_name', y='peak_memory', ax=axes[0,1])
            axes[0,1].set_ylabel('Peak Memory (MB)')
            axes[0,1].set_title('Peak Memory Usage Distribution')
            plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Node expansions
        if solved_df['node_expansions'].sum() > 0:
            sns.boxplot(data=solved_df, x='heuristic_name', y='node_expansions', ax=axes[1,0])
            axes[1,0].set_yscale('log')
            axes[1,0].set_ylabel('Node Expansions')
            axes[1,0].set_title('Search Effort Distribution')
            plt.setp(axes[1,0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plan cost
        if solved_df['plan_cost'].sum() > 0:
            sns.boxplot(data=solved_df, x='heuristic_name', y='plan_cost', ax=axes[1,1])
            axes[1,1].set_ylabel('Plan Cost')
            axes[1,1].set_title('Plan Cost Distribution')
            plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created performance comparison plots")
    
    def plot_time_vs_quality_scatter(self) -> None:
        """Create time vs quality scatter plots (Pareto frontier analysis)"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        solved_df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(solved_df) == 0 or solved_df['wall_time'].sum() == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Time vs Plan Cost
        for heuristic in solved_df['heuristic_name'].unique():
            heur_data = solved_df[solved_df['heuristic_name'] == heuristic]
            if len(heur_data) > 0:
                ax1.scatter(heur_data['wall_time'], heur_data['plan_cost'], 
                           label=heuristic, alpha=0.7, s=50,
                           color=self.heuristic_colors.get(heuristic, 'gray'))
        
        ax1.set_xlabel('Wall Clock Time (seconds)')
        ax1.set_ylabel('Plan Cost')
        ax1.set_title('Time vs Plan Cost Trade-off')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time vs Memory
        for heuristic in solved_df['heuristic_name'].unique():
            heur_data = solved_df[solved_df['heuristic_name'] == heuristic]
            if len(heur_data) > 0 and heur_data['peak_memory'].sum() > 0:
                ax2.scatter(heur_data['wall_time'], heur_data['peak_memory'], 
                           label=heuristic, alpha=0.7, s=50,
                           color=self.heuristic_colors.get(heuristic, 'gray'))
        
        ax2.set_xlabel('Wall Clock Time (seconds)')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Time vs Memory Trade-off')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_quality_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created time vs quality scatter plots")
    
    def plot_speedup_analysis(self, baseline: str = 'DTG') -> None:
        """Create speedup analysis plots"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        solved_df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(solved_df) == 0:
            return
        
        # Calculate speedups relative to baseline
        baseline_data = solved_df[solved_df['heuristic_name'] == baseline].set_index(['domain', 'problem'])['wall_time']
        
        speedup_data = []
        
        for heuristic in solved_df['heuristic_name'].unique():
            if heuristic == baseline:
                continue
                
            heur_data = solved_df[solved_df['heuristic_name'] == heuristic].set_index(['domain', 'problem'])['wall_time']
            
            # Find common instances
            common_instances = baseline_data.index.intersection(heur_data.index)
            
            if len(common_instances) > 0:
                baseline_times = baseline_data.loc[common_instances]
                heur_times = heur_data.loc[common_instances]
                
                speedups = baseline_times / heur_times
                
                for instance, speedup in speedups.items():
                    if 0.01 <= speedup <= 100:  # Filter outliers
                        speedup_data.append({
                            'heuristic': heuristic,
                            'domain': instance[0],
                            'problem': instance[1],
                            'speedup': speedup,
                            'log_speedup': np.log2(speedup)
                        })
        
        if not speedup_data:
            logger.warning("No speedup data available")
            return
        
        speedup_df = pd.DataFrame(speedup_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speedup distribution
        sns.boxplot(data=speedup_df, x='heuristic', y='speedup', ax=ax1)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax1.set_ylabel(f'Speedup vs {baseline}')
        ax1.set_title('Speedup Distribution')
        ax1.legend()
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Log speedup for better visualization
        sns.boxplot(data=speedup_df, x='heuristic', y='log_speedup', ax=ax2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No speedup')
        ax2.set_ylabel(f'Log₂ Speedup vs {baseline}')
        ax2.set_title('Log Speedup Distribution')
        ax2.legend()
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'speedup_analysis_vs_{baseline}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created speedup analysis plots vs {baseline}")
    
    def plot_scaling_analysis(self) -> None:
        """Create scaling analysis by number of agents"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        # Group by number of agents
        scaling_data = self.results_df.groupby(['num_agents', 'heuristic_name']).agg({
            'coverage': 'mean',
            'wall_time': 'median',
            'peak_memory': 'median',
            'plan_cost': 'median'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Coverage vs number of agents
        for heuristic in scaling_data['heuristic_name'].unique():
            heur_data = scaling_data[scaling_data['heuristic_name'] == heuristic]
            axes[0,0].plot(heur_data['num_agents'], heur_data['coverage'], 
                          marker='o', label=heuristic, linewidth=2,
                          color=self.heuristic_colors.get(heuristic, 'gray'))
        
        axes[0,0].set_xlabel('Number of Agents')
        axes[0,0].set_ylabel('Coverage Rate')
        axes[0,0].set_title('Coverage Scaling')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Time vs number of agents
        solved_scaling = scaling_data[scaling_data['coverage'] > 0]
        for heuristic in solved_scaling['heuristic_name'].unique():
            heur_data = solved_scaling[solved_scaling['heuristic_name'] == heuristic]
            if len(heur_data) > 0 and heur_data['wall_time'].sum() > 0:
                axes[0,1].plot(heur_data['num_agents'], heur_data['wall_time'], 
                              marker='o', label=heuristic, linewidth=2,
                              color=self.heuristic_colors.get(heuristic, 'gray'))
        
        axes[0,1].set_xlabel('Number of Agents')
        axes[0,1].set_ylabel('Median Wall Time (seconds)')
        axes[0,1].set_title('Time Scaling')
        axes[0,1].set_yscale('log')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Memory vs number of agents
        for heuristic in solved_scaling['heuristic_name'].unique():
            heur_data = solved_scaling[solved_scaling['heuristic_name'] == heuristic]
            if len(heur_data) > 0 and heur_data['peak_memory'].sum() > 0:
                axes[1,0].plot(heur_data['num_agents'], heur_data['peak_memory'], 
                              marker='o', label=heuristic, linewidth=2,
                              color=self.heuristic_colors.get(heuristic, 'gray'))
        
        axes[1,0].set_xlabel('Number of Agents')
        axes[1,0].set_ylabel('Median Peak Memory (MB)')
        axes[1,0].set_title('Memory Scaling')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plan cost vs number of agents
        for heuristic in solved_scaling['heuristic_name'].unique():
            heur_data = solved_scaling[solved_scaling['heuristic_name'] == heuristic]
            if len(heur_data) > 0 and heur_data['plan_cost'].sum() > 0:
                axes[1,1].plot(heur_data['num_agents'], heur_data['plan_cost'], 
                              marker='o', label=heuristic, linewidth=2,
                              color=self.heuristic_colors.get(heuristic, 'gray'))
        
        axes[1,1].set_xlabel('Number of Agents')
        axes[1,1].set_ylabel('Median Plan Cost')
        axes[1,1].set_title('Plan Quality Scaling')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created scaling analysis plots")
    
    def plot_domain_specific_analysis(self) -> None:
        """Create domain-specific performance analysis"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        solved_df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(solved_df) == 0:
            return
        
        domains = solved_df['domain'].unique()
        n_domains = len(domains)
        
        if n_domains == 0:
            return
        
        # Create subplots for each domain
        fig, axes = plt.subplots(n_domains, 1, figsize=(12, 4*n_domains), squeeze=False)
        
        for i, domain in enumerate(domains):
            domain_data = solved_df[solved_df['domain'] == domain]
            
            if len(domain_data) > 0 and domain_data['wall_time'].sum() > 0:
                # Time comparison for this domain
                sns.boxplot(data=domain_data, x='heuristic_name', y='wall_time', ax=axes[i,0])
                axes[i,0].set_yscale('log')
                axes[i,0].set_ylabel('Wall Time (seconds)')
                axes[i,0].set_title(f'Performance in {domain.upper()} Domain')
                plt.setp(axes[i,0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_specific_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created domain-specific analysis plots")
    
    def plot_statistical_significance_heatmap(self) -> None:
        """Create heatmap showing statistical significance between heuristics"""
        if self.results_df is None:
            self.load_and_prepare_data()
        
        from data_analyzer import ExperimentAnalyzer
        
        analyzer = ExperimentAnalyzer(self.results_file)
        analyzer.results_df = self.results_df
        
        # Test key metrics
        metrics = ['wall_time', 'plan_cost', 'node_expansions']
        heuristics = self.results_df['heuristic_name'].unique()
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            comparisons = analyzer.perform_pairwise_comparisons(metric)
            
            # Create significance matrix
            sig_matrix = np.zeros((len(heuristics), len(heuristics)))
            effect_matrix = np.zeros((len(heuristics), len(heuristics)))
            
            heur_to_idx = {h: i for i, h in enumerate(heuristics)}
            
            for comp in comparisons:
                idx_a = heur_to_idx[comp.heuristic_a]
                idx_b = heur_to_idx[comp.heuristic_b]
                
                # Significance (1 if significant, 0 if not)
                sig_value = 1 if comp.significant else 0
                sig_matrix[idx_a, idx_b] = sig_value
                sig_matrix[idx_b, idx_a] = sig_value
                
                # Effect size
                effect_matrix[idx_a, idx_b] = comp.effect_size
                effect_matrix[idx_b, idx_a] = comp.effect_size
            
            # Plot significance heatmap
            im = axes[i].imshow(sig_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
            
            # Add effect size annotations
            for row in range(len(heuristics)):
                for col in range(len(heuristics)):
                    if row != col:
                        text = f'{effect_matrix[row, col]:.2f}' if effect_matrix[row, col] > 0 else ''
                        axes[i].text(col, row, text, ha='center', va='center', 
                                   color='white' if sig_matrix[row, col] > 0.5 else 'black')
            
            axes[i].set_xticks(range(len(heuristics)))
            axes[i].set_yticks(range(len(heuristics)))
            axes[i].set_xticklabels(heuristics, rotation=45, ha='right')
            axes[i].set_yticklabels(heuristics)
            axes[i].set_title(f'Statistical Significance\n{metric}')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_significance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created statistical significance heatmap")
    
    def create_all_visualizations(self) -> None:
        """Create all visualizations"""
        logger.info("Creating all visualizations...")
        
        try:
            self.plot_coverage_comparison()
            self.plot_performance_comparison()
            self.plot_time_vs_quality_scatter()
            self.plot_speedup_analysis()
            self.plot_scaling_analysis()
            self.plot_domain_specific_analysis()
            self.plot_statistical_significance_heatmap()
            
            logger.info(f"All visualizations saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            raise

def main():
    """Main entry point for visualization"""
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else "results/all_results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    
    visualizer = ExperimentVisualizer(results_file, output_dir)
    
    try:
        visualizer.create_all_visualizations()
        print(f"Visualizations created successfully in {output_dir}/")
        
    except FileNotFoundError as e:
        logger.error(f"Results file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 