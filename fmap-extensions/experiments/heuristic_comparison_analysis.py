#!/usr/bin/env python3
"""
FMAP Heuristic Comparison Analysis

This script loads all individual experiment result files, performs comprehensive
heuristic comparison analysis, and generates plots and tables comparing the
performance of different heuristics across domains and problems.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HeuristicComparisonAnalyzer:
    """Comprehensive heuristic comparison analyzer"""
    
    def __init__(self, results_dir="results", plots_dir="plots"):
        self.results_dir = Path(results_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Heuristic name mapping (corrected to match Java source)
        self.heuristic_names = {
            1: "DTG",
            2: "DTG+Landmarks",
            3: "Inc_DTG+Landmarks",
            4: "Centroids",
            5: "MCS"
        }
        
        self.df = None
        
    def load_all_results(self):
        """Load all individual result JSON files"""
        print("Loading all experiment results...")
        
        result_files = glob.glob(str(self.results_dir / "result_*.json"))
        print(f"Found {len(result_files)} result files")
        
        results = []
        failed_files = []
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract relevant information
                result = self._extract_result_data(data, file_path)
                if result:
                    results.append(result)
                    
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"Failed to load {file_path}: {e}")
        
        if failed_files:
            print(f"Warning: Failed to load {len(failed_files)} files")
            
        if not results:
            print("No valid results found!")
            return pd.DataFrame()
            
        self.df = pd.DataFrame(results)
        print(f"Successfully loaded {len(self.df)} experiments")
        
        # Add derived columns
        self._add_derived_columns()
        
        return self.df
    
    def _extract_result_data(self, data, file_path):
        """Extract relevant data from a single result JSON"""
        try:
            # Parse filename to get experiment info
            filename = Path(file_path).stem
            parts = filename.split('_')
            
            config = data.get('config', {})
            search = data.get('search', {})
            plan = data.get('plan', {})
            
            result = {
                'filename': filename,
                'domain': config.get('domain'),
                'problem': config.get('problem'),
                'heuristic_id': config.get('heuristic'),
                'heuristic_name': self.heuristic_names.get(config.get('heuristic', 0), f"Unknown_{config.get('heuristic')}"),
                'agent_count': len(config.get('agents', [])),
                'agents': config.get('agents', []),
                
                # Search performance
                'coverage': search.get('coverage', False),
                'wall_clock_time': search.get('wall_clock_time', None),
                'cpu_time': search.get('cpu_time', None),
                'peak_memory_mb': search.get('peak_memory_mb', None),
                'search_nodes': search.get('search_nodes', None),
                
                # Plan quality
                'plan_found': plan.get('plan_found', False),
                'plan_length': plan.get('plan_length', None),
                'makespan': plan.get('makespan', None),
                
                # Heuristic values (if available)
                'dtg_heuristic_values': search.get('dtg_heuristic_values'),
                'landmark_heuristic_values': search.get('landmark_heuristic_values'),
                
                # Error information
                'error_message': data.get('error_message'),
                'has_error': data.get('error_message') is not None
            }
            
            return result
            
        except Exception as e:
            print(f"Error extracting data from {file_path}: {e}")
            return None
    
    def _add_derived_columns(self):
        """Add derived columns for analysis"""
        # Problem complexity categories
        self.df['complexity'] = self.df['agent_count'].apply(self._categorize_complexity)
        
        # Success indicator
        self.df['success'] = self.df['plan_found'].astype(int)
        
        # Performance categories
        self.df['time_category'] = pd.cut(self.df['wall_clock_time'], 
                                         bins=[0, 1, 10, 60, 1800, float('inf')],
                                         labels=['Very Fast (<1s)', 'Fast (1-10s)', 'Medium (10-60s)', 'Slow (1-30min)', 'Very Slow (>30min)'])
        
        # Memory categories
        self.df['memory_category'] = pd.cut(self.df['peak_memory_mb'],
                                           bins=[0, 100, 500, 1000, 5000, float('inf')],
                                           labels=['Low (<100MB)', 'Medium (100-500MB)', 'High (500MB-1GB)', 'Very High (1-5GB)', 'Extreme (>5GB)'])
    
    def _categorize_complexity(self, agent_count):
        """Categorize problem complexity based on agent count"""
        if agent_count == 2:
            return "Very Small (2 agents)"
        elif agent_count == 3:
            return "Small (3 agents)"
        elif agent_count <= 5:
            return "Medium (4-5 agents)"
        elif agent_count <= 8:
            return "Large (6-8 agents)"
        else:
            return "Very Large (>8 agents)"
    
    def generate_summary_tables(self):
        """Generate comprehensive summary tables"""
        print("\nGenerating summary tables...")
        
        # Overall summary by heuristic
        print("\n" + "="*80)
        print("HEURISTIC PERFORMANCE SUMMARY")
        print("="*80)
        
        summary_stats = []
        
        for heuristic in self.df['heuristic_name'].unique():
            heur_data = self.df[self.df['heuristic_name'] == heuristic]
            successful = heur_data[heur_data['plan_found'] == True]
            
            stats = {
                'Heuristic': heuristic,
                'Total_Experiments': len(heur_data),
                'Successful': len(successful),
                'Success_Rate': len(successful) / len(heur_data) if len(heur_data) > 0 else 0,
                'Avg_Time': successful['wall_clock_time'].mean() if len(successful) > 0 else None,
                'Median_Time': successful['wall_clock_time'].median() if len(successful) > 0 else None,
                'Avg_Memory': successful['peak_memory_mb'].mean() if len(successful) > 0 else None,
                'Avg_Plan_Length': successful['plan_length'].mean() if len(successful) > 0 else None,
                'Avg_Search_Nodes': successful['search_nodes'].mean() if len(successful) > 0 else None,
                'Domains_Tested': heur_data['domain'].nunique(),
                'Problems_Tested': heur_data['problem'].nunique()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.sort_values('Success_Rate', ascending=False)
        
        print(summary_df.to_string(index=False, float_format='%.3f'))
        
        # Save summary table
        summary_df.to_csv(self.plots_dir / 'heuristic_summary.csv', index=False)
        
        # Domain-specific analysis
        print("\n" + "="*80)
        print("DOMAIN-SPECIFIC PERFORMANCE")
        print("="*80)
        
        domain_analysis = []
        for domain in self.df['domain'].unique():
            domain_data = self.df[self.df['domain'] == domain]
            
            for heuristic in domain_data['heuristic_name'].unique():
                heur_domain_data = domain_data[domain_data['heuristic_name'] == heuristic]
                successful = heur_domain_data[heur_domain_data['plan_found'] == True]
                
                analysis = {
                    'Domain': domain,
                    'Heuristic': heuristic,
                    'Experiments': len(heur_domain_data),
                    'Successful': len(successful),
                    'Success_Rate': len(successful) / len(heur_domain_data) if len(heur_domain_data) > 0 else 0,
                    'Avg_Time': successful['wall_clock_time'].mean() if len(successful) > 0 else None,
                    'Avg_Plan_Length': successful['plan_length'].mean() if len(successful) > 0 else None,
                    'Avg_Agents': heur_domain_data['agent_count'].mean()
                }
                domain_analysis.append(analysis)
        
        domain_df = pd.DataFrame(domain_analysis)
        domain_df = domain_df.sort_values(['Domain', 'Success_Rate'], ascending=[True, False])
        
        print(domain_df.to_string(index=False, float_format='%.3f'))
        domain_df.to_csv(self.plots_dir / 'domain_analysis.csv', index=False)
        
        return summary_df, domain_df
    
    def plot_heuristic_comparison(self):
        """Generate comprehensive heuristic comparison plots"""
        print("\nGenerating heuristic comparison plots...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Success Rate Comparison
        ax1 = plt.subplot(3, 3, 1)
        success_by_heuristic = self.df.groupby('heuristic_name')['success'].mean()
        bars = ax1.bar(range(len(success_by_heuristic)), success_by_heuristic.values, 
                       color=sns.color_palette("husl", len(success_by_heuristic)))
        ax1.set_title('Success Rate by Heuristic', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(success_by_heuristic)))
        ax1.set_xticklabels(success_by_heuristic.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(success_by_heuristic.values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Execution Time Comparison (successful runs only)
        ax2 = plt.subplot(3, 3, 2)
        successful_df = self.df[self.df['plan_found'] == True]
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='wall_clock_time', ax=ax2)
            ax2.set_title('Execution Time Distribution (Successful Runs)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Wall Clock Time (seconds)')
            ax2.set_yscale('log')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Memory Usage Comparison
        ax3 = plt.subplot(3, 3, 3)
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='peak_memory_mb', ax=ax3)
            ax3.set_title('Memory Usage Distribution', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Peak Memory (MB)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Plan Length Comparison
        ax4 = plt.subplot(3, 3, 4)
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='plan_length', ax=ax4)
            ax4.set_title('Plan Length Distribution', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Plan Length (actions)')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Search Nodes Comparison
        ax5 = plt.subplot(3, 3, 5)
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='search_nodes', ax=ax5)
            ax5.set_title('Search Nodes Distribution', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Search Nodes')
            ax5.set_yscale('log')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Success Rate by Domain
        ax6 = plt.subplot(3, 3, 6)
        domain_heuristic_success = self.df.groupby(['domain', 'heuristic_name'])['success'].mean().unstack()
        domain_heuristic_success.plot(kind='bar', ax=ax6, width=0.8)
        ax6.set_title('Success Rate by Domain and Heuristic', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Success Rate')
        ax6.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. Performance vs Problem Complexity
        ax7 = plt.subplot(3, 3, 7)
        complexity_performance = successful_df.groupby(['complexity', 'heuristic_name'])['wall_clock_time'].mean().unstack()
        complexity_performance.plot(kind='bar', ax=ax7, width=0.8)
        ax7.set_title('Average Time by Complexity', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Average Wall Clock Time (s)')
        ax7.set_yscale('log')
        ax7.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Scatter: Time vs Plan Length
        ax8 = plt.subplot(3, 3, 8)
        for heuristic in successful_df['heuristic_name'].unique():
            heur_data = successful_df[successful_df['heuristic_name'] == heuristic]
            if len(heur_data) > 0:
                ax8.scatter(heur_data['plan_length'], heur_data['wall_clock_time'], 
                           label=heuristic, alpha=0.7, s=30)
        ax8.set_xlabel('Plan Length (actions)')
        ax8.set_ylabel('Wall Clock Time (seconds)')
        ax8.set_title('Time vs Plan Quality', fontsize=12, fontweight='bold')
        ax8.set_yscale('log')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Heuristic Performance Ranking
        ax9 = plt.subplot(3, 3, 9)
        # Create a ranking based on multiple criteria
        ranking_data = []
        for heuristic in self.df['heuristic_name'].unique():
            heur_data = self.df[self.df['heuristic_name'] == heuristic]
            successful = heur_data[heur_data['plan_found'] == True]
            
            if len(successful) > 0:
                score = (len(successful) / len(heur_data)) * 0.4  # Success rate weight
                score += (1 / (successful['wall_clock_time'].mean() + 1e-6)) * 0.3  # Speed weight (inverse)
                score += (1 / (successful['plan_length'].mean() + 1e-6)) * 0.3  # Plan quality weight (inverse)
                
                ranking_data.append({
                    'Heuristic': heuristic,
                    'Score': score,
                    'Experiments': len(heur_data),
                    'Success_Rate': len(successful) / len(heur_data)
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Score', ascending=False)
        
        bars = ax9.barh(range(len(ranking_df)), ranking_df['Score'], 
                        color=sns.color_palette("viridis", len(ranking_df)))
        ax9.set_yticks(range(len(ranking_df)))
        ax9.set_yticklabels(ranking_df['Heuristic'])
        ax9.set_xlabel('Composite Performance Score')
        ax9.set_title('Heuristic Performance Ranking', fontsize=12, fontweight='bold')
        
        # Add score labels
        for i, v in enumerate(ranking_df['Score']):
            ax9.text(v + max(ranking_df['Score']) * 0.01, i, f'{v:.3f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heuristic_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return ranking_df
    
    def plot_domain_analysis(self):
        """Generate domain-specific analysis plots"""
        print("\nGenerating domain analysis plots...")
        
        domains = self.df['domain'].unique()
        n_domains = len(domains)
        
        if n_domains == 0:
            print("No domains found!")
            return
        
        # Create subplots for each domain
        fig, axes = plt.subplots(2, (n_domains + 1) // 2, figsize=(5 * ((n_domains + 1) // 2), 10))
        if n_domains == 1:
            axes = [axes]
        elif n_domains <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, domain in enumerate(domains):
            if i >= len(axes):
                break
                
            domain_data = self.df[self.df['domain'] == domain]
            
            # Success rate by heuristic for this domain
            domain_success = domain_data.groupby('heuristic_name')['success'].mean()
            
            bars = axes[i].bar(range(len(domain_success)), domain_success.values,
                              color=sns.color_palette("Set2", len(domain_success)))
            axes[i].set_title(f'{domain.title()} Domain\nSuccess Rate by Heuristic', 
                             fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Success Rate')
            axes[i].set_ylim(0, 1)
            axes[i].set_xticks(range(len(domain_success)))
            axes[i].set_xticklabels(domain_success.index, rotation=45, ha='right')
            
            # Add value labels
            for j, v in enumerate(domain_success.values):
                axes[i].text(j, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(domains), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'domain_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """Generate a detailed text report"""
        print("\nGenerating detailed analysis report...")
        
        report_lines = []
        report_lines.append("FMAP HEURISTIC COMPARISON ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL EXPERIMENT STATISTICS")
        report_lines.append("-" * 30)
        report_lines.append(f"Total experiments: {len(self.df)}")
        report_lines.append(f"Successful experiments: {len(self.df[self.df['plan_found'] == True])}")
        report_lines.append(f"Overall success rate: {self.df['success'].mean():.3f}")
        report_lines.append(f"Domains tested: {self.df['domain'].nunique()}")
        report_lines.append(f"Problems tested: {self.df['problem'].nunique()}")
        report_lines.append(f"Heuristics tested: {self.df['heuristic_name'].nunique()}")
        report_lines.append(f"Agent count range: {self.df['agent_count'].min()} - {self.df['agent_count'].max()}")
        report_lines.append("")
        
        # Heuristic performance summary
        report_lines.append("HEURISTIC PERFORMANCE RANKING")
        report_lines.append("-" * 30)
        
        for heuristic in self.df['heuristic_name'].unique():
            heur_data = self.df[self.df['heuristic_name'] == heuristic]
            successful = heur_data[heur_data['plan_found'] == True]
            
            report_lines.append(f"\n{heuristic}:")
            report_lines.append(f"  Experiments: {len(heur_data)}")
            report_lines.append(f"  Success rate: {len(successful) / len(heur_data):.3f}")
            
            if len(successful) > 0:
                report_lines.append(f"  Avg execution time: {successful['wall_clock_time'].mean():.3f} seconds")
                report_lines.append(f"  Avg memory usage: {successful['peak_memory_mb'].mean():.1f} MB")
                report_lines.append(f"  Avg plan length: {successful['plan_length'].mean():.1f} actions")
                report_lines.append(f"  Domains covered: {heur_data['domain'].nunique()}")
        
        # Save report
        with open(self.plots_dir / 'analysis_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
    
    def run_complete_analysis(self):
        """Run the complete heuristic comparison analysis"""
        print("Starting FMAP Heuristic Comparison Analysis...")
        print("=" * 60)
        
        # Load data
        df = self.load_all_results()
        if df.empty:
            print("No results found! Please ensure result files exist in the results directory.")
            return
        
        # Generate summary tables
        summary_df, domain_df = self.generate_summary_tables()
        
        # Generate plots
        ranking_df = self.plot_heuristic_comparison()
        self.plot_domain_analysis()
        
        # Generate detailed report
        self.generate_detailed_report()
        
        print(f"\nAnalysis complete! Files saved to: {self.plots_dir}")
        print(f"- heuristic_comparison.png: Main comparison plots")
        print(f"- domain_analysis.png: Domain-specific analysis")
        print(f"- heuristic_summary.csv: Summary statistics table")
        print(f"- domain_analysis.csv: Domain-specific results table")
        print(f"- analysis_report.txt: Detailed text report")
        
        return {
            'summary_df': summary_df,
            'domain_df': domain_df,
            'ranking_df': ranking_df,
            'main_df': self.df
        }

def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FMAP Heuristic Comparison Analysis')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing result JSON files')
    parser.add_argument('--plots-dir', default='plots',
                       help='Directory to save plots and tables')
    
    args = parser.parse_args()
    
    analyzer = HeuristicComparisonAnalyzer(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir
    )
    
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main()