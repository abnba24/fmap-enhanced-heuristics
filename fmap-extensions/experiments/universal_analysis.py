#!/usr/bin/env python3
"""
Universal FMAP Experiment Analysis Tool
Automatically detects and handles both data formats:
1. Individual result_*.json files (from experiment_runner_resume.py)
2. all_results.json file (from experiment_runner.py)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalFMAPAnalyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Heuristic mapping
        self.heuristic_names = {
            1: "DTG", 
            2: "DTG+Landmarks", 
            3: "Inc_DTG+Landmarks", 
            4: "Centroids", 
            5: "MCS"
        }
        
    def auto_detect_and_load_data(self):
        """Automatically detect data format and load accordingly"""
        all_results_file = self.results_dir / "all_results.json"
        individual_files = list(self.results_dir.glob("result_*.json"))
        
        logger.info("Auto-detecting data format...")
        
        if all_results_file.exists():
            logger.info(f"Found all_results.json - using original experiment_runner.py format")
            return self._load_all_results_format()
        elif individual_files:
            logger.info(f"Found {len(individual_files)} individual result files - using resume runner format")
            return self._load_individual_results_format()
        else:
            logger.error("No experiment results found! Expected either:")
            logger.error("  - all_results.json (from experiment_runner.py)")
            logger.error("  - result_*.json files (from experiment_runner_resume.py)")
            return pd.DataFrame()
    
    def _load_all_results_format(self):
        """Load data from all_results.json (original experiment_runner.py format)"""
        all_results_file = self.results_dir / "all_results.json"
        
        try:
            with open(all_results_file, 'r') as f:
                all_results = json.load(f)
            
            logger.info(f"Loaded {len(all_results)} experiments from all_results.json")
            return self._convert_to_dataframe(all_results)
            
        except Exception as e:
            logger.error(f"Error loading all_results.json: {e}")
            return pd.DataFrame()
    
    def _load_individual_results_format(self):
        """Load data from individual result_*.json files (resume runner format)"""
        result_files = list(self.results_dir.glob("result_*.json"))
        results = []
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error loading {result_file}: {e}")
        
        logger.info(f"Loaded {len(results)} experiments from individual files")
        return self._convert_to_dataframe(results)
    
    def _convert_to_dataframe(self, results):
        """Convert results list to pandas DataFrame"""
        if not results:
            return pd.DataFrame()
        
        df_data = []
        for result in results:
            config = result.get('config', {})
            search = result.get('search', {})
            plan = result.get('plan', {})
            
            row = {
                'domain': config.get('domain', 'unknown'),
                'problem': config.get('problem', 'unknown'),
                'heuristic_id': config.get('heuristic', 0),
                'agent_count': len(config.get('agents', [])),
                'coverage': search.get('coverage', False),
                'wall_clock_time': search.get('wall_clock_time', 0),
                'cpu_time': search.get('cpu_time', 0),
                'peak_memory_mb': search.get('peak_memory_mb', 0),
                'search_nodes': search.get('search_nodes', 0),
                'plan_found': plan.get('plan_found', False),
                'plan_length': plan.get('plan_length', 0),
                'makespan': plan.get('makespan', 0.0),
                'error_message': result.get('error_message', ''),
                'agents': ', '.join(config.get('agents', [])),
            }
            
            # Add heuristic name
            row['heuristic_name'] = self.heuristic_names.get(row['heuristic_id'], f"H{row['heuristic_id']}")
            
            # Add complexity category
            row['complexity'] = self._categorize_complexity(row['problem'], row['agent_count'])
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        logger.info(f"Converted to DataFrame: {len(df)} experiments")
        return df
    
    def _categorize_complexity(self, problem_name, agent_count):
        """Categorize problem complexity based on problem number and agent count"""
        import re
        numbers = re.findall(r'\\d+', problem_name)
        problem_num = int(numbers[-1]) if numbers else 0
        
        if agent_count <= 2:
            if problem_num <= 5:
                return "SMALL"
            elif problem_num <= 15:
                return "MEDIUM"
            else:
                return "LARGE"
        elif agent_count <= 5:
            if problem_num <= 3:
                return "SMALL" 
            elif problem_num <= 10:
                return "MEDIUM"
            else:
                return "LARGE"
        else:
            if problem_num <= 2:
                return "SMALL"
            elif problem_num <= 5:
                return "MEDIUM"
            else:
                return "LARGE"
    
    def generate_compatibility_report(self, df):
        """Generate a report about data compatibility and sources"""
        report_file = self.results_dir / "compatibility_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("FMAP EXPERIMENT DATA COMPATIBILITY REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Data source detection
            all_results_exists = (self.results_dir / "all_results.json").exists()
            individual_files = list(self.results_dir.glob("result_*.json"))
            
            f.write("DATA SOURCE DETECTION\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"all_results.json found: {'YES' if all_results_exists else 'NO'}\\n")
            f.write(f"Individual result files: {'YES' if individual_files else 'NO'} ({len(individual_files)} files)\\n")
            f.write(f"Data format used: {'Original experiment_runner.py' if all_results_exists else 'Resume runner format'}\\n\\n")
            
            # Experiment overview  
            f.write("EXPERIMENT OVERVIEW\\n")
            f.write("-" * 18 + "\\n")
            f.write(f"Total experiments loaded: {len(df)}\\n")
            
            if len(df) > 0:
                successful_df = df[df['coverage'] == True]
                f.write(f"Successful experiments: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)\\n")
                f.write(f"Domains: {', '.join(sorted(df['domain'].unique()))}\\n")
                f.write(f"Heuristics: {', '.join(sorted(df['heuristic_name'].unique()))}\\n")
                f.write(f"Agent count range: {df['agent_count'].min()} - {df['agent_count'].max()}\\n")
                
                # Data quality check
                f.write("\\nDATA QUALITY CHECK\\n")
                f.write("-" * 18 + "\\n")
                f.write(f"Records with missing domains: {df['domain'].isna().sum()}\\n")
                f.write(f"Records with missing heuristics: {df['heuristic_id'].isna().sum()}\\n")
                f.write(f"Records with zero execution time: {(df['wall_clock_time'] == 0).sum()}\\n")
                f.write(f"Records with errors: {df['error_message'].str.len().gt(0).sum()}\\n")
            
            f.write("\\nAll data formats are compatible with this universal analyzer!\\n")
        
        logger.info(f"Compatibility report saved to {report_file}")
    
    def run_comprehensive_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("Starting Universal FMAP Analysis...")
        
        # Auto-detect and load data
        df = self.auto_detect_and_load_data()
        
        if df.empty:
            logger.error("No data loaded - analysis cannot proceed")
            return
        
        # Generate compatibility report
        self.generate_compatibility_report(df)
        
        # Generate all analysis components
        logger.info("Generating comprehensive analysis...")
        self.generate_summary_report(df)
        self.plot_heuristic_comparison(df)
        self.plot_domain_analysis(df)
        self.plot_performance_trends(df)
        self.generate_correlation_analysis(df)
        
        logger.info("Universal analysis complete!")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        logger.info(f"Plots saved to: {self.plots_dir}")
    
    def generate_summary_report(self, df):
        """Generate comprehensive summary report"""
        report_file = self.results_dir / "universal_analysis_summary.txt"
        successful_df = df[df['coverage'] == True]
        
        with open(report_file, 'w') as f:
            f.write("UNIVERSAL FMAP HEURISTIC ANALYSIS SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Total experiments: {len(df)}\\n")
            f.write(f"Successful experiments: {len(successful_df)} ({len(successful_df)/len(df)*100:.1f}%)\\n")
            f.write(f"Domains tested: {df['domain'].nunique()}\\n")
            f.write(f"Problems tested: {df['problem'].nunique()}\\n")
            f.write(f"Heuristics tested: {df['heuristic_name'].nunique()}\\n")
            f.write(f"Agent count range: {df['agent_count'].min()} - {df['agent_count'].max()}\\n\\n")
            
            if len(successful_df) > 0:
                # Performance by heuristic
                f.write("PERFORMANCE BY HEURISTIC\\n")
                f.write("-" * 25 + "\\n")
                heuristic_perf = successful_df.groupby('heuristic_name').agg({
                    'wall_clock_time': ['count', 'mean', 'std', 'min', 'max'],
                    'peak_memory_mb': ['mean', 'std'],
                    'plan_length': ['mean', 'std']
                }).round(3)
                f.write(str(heuristic_perf))
                f.write("\\n\\n")
                
                # Best performing heuristic
                best_heuristic = successful_df.groupby('heuristic_name')['wall_clock_time'].mean().idxmin()
                best_time = successful_df.groupby('heuristic_name')['wall_clock_time'].mean().min()
                f.write(f"BEST PERFORMING HEURISTIC: {best_heuristic}\\n")
                f.write(f"Average execution time: {best_time:.3f} seconds\\n\\n")
        
        logger.info(f"Universal summary report saved to {report_file}")
    
    def plot_heuristic_comparison(self, df):
        """Plot heuristic performance comparison"""
        successful_df = df[df['coverage'] == True]
        
        if len(successful_df) == 0:
            logger.warning("No successful experiments to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Universal FMAP Heuristic Performance Analysis', fontsize=16)
        
        # 1. Success rate by heuristic
        total_by_heuristic = df.groupby('heuristic_name').size()
        success_by_heuristic = successful_df.groupby('heuristic_name').size()
        success_rate = (success_by_heuristic / total_by_heuristic * 100).fillna(0)
        
        success_rate.plot(kind='bar', ax=axes[0,0], color='lightblue')
        axes[0,0].set_title('Success Rate by Heuristic')
        axes[0,0].set_ylabel('Success Rate (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Average execution time
        time_by_heuristic = successful_df.groupby('heuristic_name')['wall_clock_time'].agg(['mean', 'std'])
        time_by_heuristic['mean'].plot(kind='bar', yerr=time_by_heuristic['std'], ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Average Execution Time by Heuristic')
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Memory usage
        memory_by_heuristic = successful_df.groupby('heuristic_name')['peak_memory_mb'].agg(['mean', 'std'])
        memory_by_heuristic['mean'].plot(kind='bar', yerr=memory_by_heuristic['std'], ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Average Memory Usage by Heuristic')
        axes[1,0].set_ylabel('Memory (MB)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Plan quality
        quality_by_heuristic = successful_df.groupby('heuristic_name')['plan_length'].agg(['mean', 'std'])
        quality_by_heuristic['mean'].plot(kind='bar', yerr=quality_by_heuristic['std'], ax=axes[1,1], color='lightyellow')
        axes[1,1].set_title('Average Plan Length by Heuristic')
        axes[1,1].set_ylabel('Plan Length')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'universal_heuristic_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Heuristic comparison plot saved")
    
    def plot_domain_analysis(self, df):
        """Plot domain-specific analysis"""
        successful_df = df[df['coverage'] == True]
        
        if len(successful_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Universal FMAP Domain Analysis', fontsize=16)
        
        # 1. Success rate by domain
        total_by_domain = df.groupby('domain').size()
        success_by_domain = successful_df.groupby('domain').size()
        success_rate_domain = (success_by_domain / total_by_domain * 100).fillna(0)
        
        success_rate_domain.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Success Rate by Domain')
        axes[0,0].set_ylabel('Success Rate (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Performance heatmap
        if len(successful_df) > 5:
            pivot_time = successful_df.pivot_table(values='wall_clock_time', 
                                                 index='domain', 
                                                 columns='heuristic_name', 
                                                 aggfunc='mean')
            sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='viridis_r', ax=axes[0,1])
            axes[0,1].set_title('Execution Time by Domain and Heuristic')
        
        # 3. Agent count distribution
        agent_dist = df.groupby('domain')['agent_count'].agg(['min', 'max', 'mean'])
        agent_dist['mean'].plot(kind='bar', ax=axes[1,0], color='orange')
        axes[1,0].set_title('Average Agent Count by Domain')
        axes[1,0].set_ylabel('Agent Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Complexity distribution
        complexity_counts = df.groupby(['domain', 'complexity']).size().unstack(fill_value=0)
        complexity_counts.plot(kind='bar', stacked=True, ax=axes[1,1])
        axes[1,1].set_title('Problem Complexity Distribution by Domain')
        axes[1,1].set_ylabel('Number of Problems')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'universal_domain_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Domain analysis plot saved")
    
    def plot_performance_trends(self, df):
        """Plot performance trends and scaling analysis"""
        successful_df = df[df['coverage'] == True]
        
        if len(successful_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Universal FMAP Performance Trends', fontsize=16)
        
        # 1. Time vs Agent Count
        for heuristic in successful_df['heuristic_name'].unique():
            heuristic_data = successful_df[successful_df['heuristic_name'] == heuristic]
            if len(heuristic_data) > 1:
                axes[0,0].scatter(heuristic_data['agent_count'], heuristic_data['wall_clock_time'], 
                                 label=heuristic, alpha=0.7)
        axes[0,0].set_xlabel('Agent Count')
        axes[0,0].set_ylabel('Execution Time (seconds)')
        axes[0,0].set_title('Scaling: Time vs Agent Count')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # 2. Memory vs Agent Count
        for heuristic in successful_df['heuristic_name'].unique():
            heuristic_data = successful_df[successful_df['heuristic_name'] == heuristic]
            if len(heuristic_data) > 1:
                axes[0,1].scatter(heuristic_data['agent_count'], heuristic_data['peak_memory_mb'], 
                                 label=heuristic, alpha=0.7)
        axes[0,1].set_xlabel('Agent Count')
        axes[0,1].set_ylabel('Peak Memory (MB)')
        axes[0,1].set_title('Scaling: Memory vs Agent Count')
        axes[0,1].legend()
        
        # 3. Search Nodes vs Time
        axes[1,0].scatter(successful_df['search_nodes'], successful_df['wall_clock_time'], alpha=0.6)
        axes[1,0].set_xlabel('Search Nodes')
        axes[1,0].set_ylabel('Execution Time (seconds)')
        axes[1,0].set_title('Search Efficiency: Nodes vs Time')
        axes[1,0].set_xscale('log')
        axes[1,0].set_yscale('log')
        
        # 4. Plan Quality vs Time
        plan_df = successful_df[successful_df['plan_length'] > 0]
        if len(plan_df) > 0:
            axes[1,1].scatter(plan_df['wall_clock_time'], plan_df['plan_length'], alpha=0.6)
            axes[1,1].set_xlabel('Execution Time (seconds)')
            axes[1,1].set_ylabel('Plan Length')
            axes[1,1].set_title('Quality vs Speed Trade-off')
            axes[1,1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'universal_performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Performance trends plot saved")
    
    def generate_correlation_analysis(self, df):
        """Generate correlation analysis between metrics"""
        successful_df = df[df['coverage'] == True]
        
        if len(successful_df) < 5:
            return
        
        # Select numeric columns for correlation
        numeric_cols = ['agent_count', 'wall_clock_time', 'cpu_time', 'peak_memory_mb', 
                       'search_nodes', 'plan_length', 'makespan']
        correlation_data = successful_df[numeric_cols].dropna()
        
        if len(correlation_data) < 5:
            return
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Universal FMAP Performance Metrics Correlation')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'universal_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Correlation analysis plot saved")

def main():
    """Main execution function"""
    analyzer = UniversalFMAPAnalyzer("results")
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()