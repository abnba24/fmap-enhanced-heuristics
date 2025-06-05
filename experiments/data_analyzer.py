#!/usr/bin/env python3
"""
FMAP Experiment Data Analyzer

This module analyzes experimental results, computes statistical metrics,
and performs significance testing for heuristic comparisons.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
import logging
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Results of statistical comparison between two heuristics"""
    heuristic_a: str
    heuristic_b: str
    metric: str
    p_value: float
    statistic: float
    effect_size: float
    significant: bool
    a_better: bool  # True if heuristic_a is significantly better
    median_a: float
    median_b: float
    improvement_ratio: float

class ExperimentAnalyzer:
    """Analyzes experimental results and computes statistical metrics"""
    
    def __init__(self, results_file: str = "results/all_results.json"):
        self.results_file = results_file
        self.results_df = None
        self.heuristic_names = {
            1: "DTG",
            2: "DTG+Landmarks", 
            3: "Inc_DTG+Landmarks",
            4: "Centroids",
            5: "MCS"
        }
        
    def load_results(self) -> pd.DataFrame:
        """Load experimental results into a pandas DataFrame"""
        if not Path(self.results_file).exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            results_data = json.load(f)
        
        # Flatten the nested structure
        flattened_data = []
        for result in results_data:
            row = {}
            
            # Basic info
            row['domain'] = result['config']['domain']
            row['problem'] = result['config']['problem']
            row['heuristic_id'] = result['config']['heuristic']
            row['heuristic_name'] = self.heuristic_names.get(result['config']['heuristic'], 'Unknown')
            row['num_agents'] = len(result['config']['agents'])
            
            # Performance metrics
            perf = result['performance']
            row['coverage'] = perf['coverage']
            row['wall_time'] = perf['wall_clock_time']
            row['cpu_time'] = perf['cpu_time']
            row['peak_memory'] = perf['peak_memory_mb']
            row['node_expansions'] = perf['node_expansions']
            row['node_generations'] = perf['node_generations']
            row['effective_branching_factor'] = perf['effective_branching_factor']
            row['solution_depth'] = perf['solution_depth']
            
            # Plan quality metrics
            plan = result['plan_quality']
            row['plan_cost'] = plan['plan_cost']
            row['makespan'] = plan['makespan']
            row['concurrency_index'] = plan['concurrency_index']
            row['goal_distance_mean'] = plan['goal_distance_mean']
            row['goal_distance_max'] = plan['goal_distance_max']
            row['num_actions'] = plan['num_actions']
            row['parallel_actions'] = plan['parallel_actions']
            
            # Heuristic quality metrics
            heur = result['heuristic_quality']
            row['mean_absolute_error'] = heur['mean_absolute_error']
            row['rmse_error'] = heur['rmse_error']
            row['informedness_ratio'] = heur['informedness_ratio']
            row['correlation_with_true_cost'] = heur['correlation_with_true_cost']
            row['avg_computation_time_ms'] = heur['avg_computation_time_ms']
            row['guidance_reduction_factor'] = heur['guidance_reduction_factor']
            
            # Coordination metrics
            coord = result['coordination']
            row['messages_exchanged'] = coord['messages_exchanged']
            row['total_data_volume_bytes'] = coord['total_data_volume_bytes']
            row['synchronization_rounds'] = coord['synchronization_rounds']
            row['coordination_latency_ms'] = coord['coordination_latency_ms']
            row['privacy_leakage_score'] = coord['privacy_leakage_score']
            
            # Error info
            row['error_message'] = result.get('error_message')
            row['has_error'] = result.get('error_message') is not None
            
            flattened_data.append(row)
        
        self.results_df = pd.DataFrame(flattened_data)
        logger.info(f"Loaded {len(self.results_df)} experimental results")
        
        return self.results_df
    
    def compute_coverage_statistics(self) -> pd.DataFrame:
        """Compute coverage statistics by heuristic and domain"""
        df = self.results_df
        
        coverage_stats = df.groupby(['heuristic_name', 'domain']).agg({
            'coverage': ['count', 'sum', 'mean'],
            'has_error': 'sum'
        }).round(3)
        
        coverage_stats.columns = ['total_instances', 'solved_instances', 'coverage_rate', 'error_count']
        coverage_stats = coverage_stats.reset_index()
        
        # Overall coverage by heuristic
        overall_coverage = df.groupby('heuristic_name').agg({
            'coverage': ['count', 'sum', 'mean'],
            'has_error': 'sum'
        }).round(3)
        overall_coverage.columns = ['total_instances', 'solved_instances', 'coverage_rate', 'error_count']
        
        return coverage_stats, overall_coverage
    
    def compute_performance_statistics(self) -> Dict[str, pd.DataFrame]:
        """Compute performance statistics for solved instances only"""
        df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(df) == 0:
            logger.warning("No solved instances found for performance analysis")
            return {}
        
        performance_metrics = [
            'wall_time', 'cpu_time', 'peak_memory', 'node_expansions', 
            'effective_branching_factor'
        ]
        
        stats_results = {}
        
        for metric in performance_metrics:
            if df[metric].sum() == 0:  # Skip metrics with no data
                continue
                
            metric_stats = df.groupby(['heuristic_name', 'domain'])[metric].agg([
                'count', 'mean', 'median', 'std', 
                lambda x: np.percentile(x, 25),  # Q1
                lambda x: np.percentile(x, 75),  # Q3
                'min', 'max'
            ]).round(4)
            
            metric_stats.columns = ['count', 'mean', 'median', 'std', 'q1', 'q3', 'min', 'max']
            stats_results[metric] = metric_stats.reset_index()
        
        return stats_results
    
    def compute_plan_quality_statistics(self) -> Dict[str, pd.DataFrame]:
        """Compute plan quality statistics"""
        df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(df) == 0:
            return {}
        
        quality_metrics = [
            'plan_cost', 'makespan', 'concurrency_index', 
            'goal_distance_mean', 'goal_distance_max'
        ]
        
        stats_results = {}
        
        for metric in quality_metrics:
            if df[metric].sum() == 0:
                continue
                
            metric_stats = df.groupby(['heuristic_name', 'domain'])[metric].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(4)
            
            stats_results[metric] = metric_stats.reset_index()
        
        return stats_results
    
    def perform_pairwise_comparisons(self, metric: str, alpha: float = 0.05) -> List[ComparisonResult]:
        """Perform pairwise statistical comparisons between heuristics"""
        df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(df) == 0 or metric not in df.columns:
            return []
        
        heuristics = df['heuristic_name'].unique()
        comparisons = []
        
        for i in range(len(heuristics)):
            for j in range(i + 1, len(heuristics)):
                h1, h2 = heuristics[i], heuristics[j]
                
                # Get matched instances (same domain/problem)
                h1_data = df[df['heuristic_name'] == h1].set_index(['domain', 'problem'])[metric]
                h2_data = df[df['heuristic_name'] == h2].set_index(['domain', 'problem'])[metric]
                
                # Find common instances
                common_instances = h1_data.index.intersection(h2_data.index)
                
                if len(common_instances) < 3:  # Need at least 3 paired observations
                    continue
                
                h1_values = h1_data.loc[common_instances].values
                h2_values = h2_data.loc[common_instances].values
                
                # Remove any zero or invalid values for time/memory metrics
                if metric in ['wall_time', 'cpu_time', 'peak_memory']:
                    valid_mask = (h1_values > 0) & (h2_values > 0)
                    h1_values = h1_values[valid_mask]
                    h2_values = h2_values[valid_mask]
                
                if len(h1_values) < 3:
                    continue
                
                # Perform Wilcoxon signed-rank test (paired)
                try:
                    statistic, p_value = wilcoxon(h1_values, h2_values, alternative='two-sided')
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(h1_values) - 1) * np.var(h1_values, ddof=1) + 
                                         (len(h2_values) - 1) * np.var(h2_values, ddof=1)) / 
                                        (len(h1_values) + len(h2_values) - 2))
                    effect_size = (np.mean(h1_values) - np.mean(h2_values)) / pooled_std if pooled_std > 0 else 0
                    
                    # Determine which is better (lower is better for time/memory, higher for quality)
                    better_lower = metric in ['wall_time', 'cpu_time', 'peak_memory', 'node_expansions', 'plan_cost']
                    h1_better = (np.median(h1_values) < np.median(h2_values)) if better_lower else (np.median(h1_values) > np.median(h2_values))
                    
                    # Calculate improvement ratio
                    med1, med2 = np.median(h1_values), np.median(h2_values)
                    if better_lower and med2 > 0:
                        improvement_ratio = med2 / med1 if h1_better else med1 / med2
                    elif not better_lower and med2 > 0:
                        improvement_ratio = med1 / med2 if h1_better else med2 / med1
                    else:
                        improvement_ratio = 1.0
                    
                    comparison = ComparisonResult(
                        heuristic_a=h1,
                        heuristic_b=h2,
                        metric=metric,
                        p_value=p_value,
                        statistic=statistic,
                        effect_size=abs(effect_size),
                        significant=p_value < alpha,
                        a_better=h1_better,
                        median_a=med1,
                        median_b=med2,
                        improvement_ratio=improvement_ratio
                    )
                    
                    comparisons.append(comparison)
                    
                except Exception as e:
                    logger.warning(f"Statistical test failed for {h1} vs {h2} on {metric}: {e}")
        
        return comparisons
    
    def compute_geometric_mean_speedup(self, baseline_heuristic: str = "DTG") -> pd.DataFrame:
        """Compute geometric mean speedup relative to baseline heuristic"""
        df = self.results_df[self.results_df['coverage'] == True].copy()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Get baseline times
        baseline_data = df[df['heuristic_name'] == baseline_heuristic].set_index(['domain', 'problem'])['wall_time']
        
        speedups = []
        
        for heuristic in df['heuristic_name'].unique():
            if heuristic == baseline_heuristic:
                continue
                
            heur_data = df[df['heuristic_name'] == heuristic].set_index(['domain', 'problem'])['wall_time']
            
            # Find common instances
            common_instances = baseline_data.index.intersection(heur_data.index)
            
            if len(common_instances) == 0:
                continue
            
            baseline_times = baseline_data.loc[common_instances]
            heur_times = heur_data.loc[common_instances]
            
            # Calculate speedup ratios
            ratios = baseline_times / heur_times
            
            # Remove outliers (ratios > 100 or < 0.01)
            ratios = ratios[(ratios >= 0.01) & (ratios <= 100)]
            
            if len(ratios) > 0:
                geometric_mean_speedup = np.exp(np.mean(np.log(ratios)))
                
                speedups.append({
                    'heuristic': heuristic,
                    'baseline': baseline_heuristic,
                    'geometric_mean_speedup': geometric_mean_speedup,
                    'num_instances': len(ratios),
                    'median_speedup': np.median(ratios),
                    'min_speedup': np.min(ratios),
                    'max_speedup': np.max(ratios)
                })
        
        return pd.DataFrame(speedups)
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        if self.results_df is None:
            self.load_results()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FMAP HEURISTIC COMPARISON EXPERIMENTAL REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall statistics
        total_experiments = len(self.results_df)
        total_solved = self.results_df['coverage'].sum()
        total_errors = self.results_df['has_error'].sum()
        
        report_lines.append(f"Overall Statistics:")
        report_lines.append(f"  Total experiments: {total_experiments}")
        report_lines.append(f"  Successfully solved: {total_solved} ({100*total_solved/total_experiments:.1f}%)")
        report_lines.append(f"  Errors: {total_errors} ({100*total_errors/total_experiments:.1f}%)")
        report_lines.append("")
        
        # Coverage by heuristic
        coverage_stats, overall_coverage = self.compute_coverage_statistics()
        report_lines.append("Coverage by Heuristic:")
        for _, row in overall_coverage.iterrows():
            heuristic = row.name
            coverage_rate = row['coverage_rate']
            solved = int(row['solved_instances'])
            total = int(row['total_instances'])
            report_lines.append(f"  {heuristic:20}: {solved:3d}/{total:3d} ({100*coverage_rate:5.1f}%)")
        report_lines.append("")
        
        # Performance comparison
        if total_solved > 0:
            report_lines.append("Performance Analysis (solved instances only):")
            
            # Geometric mean speedups
            speedups = self.compute_geometric_mean_speedup()
            if not speedups.empty:
                report_lines.append("  Geometric Mean Speedup vs DTG:")
                for _, row in speedups.iterrows():
                    heuristic = row['heuristic']
                    speedup = row['geometric_mean_speedup']
                    instances = int(row['num_instances'])
                    if speedup > 1:
                        report_lines.append(f"    {heuristic:20}: {speedup:6.2f}x faster ({instances} instances)")
                    else:
                        report_lines.append(f"    {heuristic:20}: {1/speedup:6.2f}x slower ({instances} instances)")
                report_lines.append("")
            
            # Statistical significance tests
            key_metrics = ['wall_time', 'plan_cost', 'node_expansions']
            for metric in key_metrics:
                comparisons = self.perform_pairwise_comparisons(metric)
                significant_comparisons = [c for c in comparisons if c.significant]
                
                if significant_comparisons:
                    report_lines.append(f"  Significant differences in {metric}:")
                    for comp in significant_comparisons:
                        direction = "better" if comp.a_better else "worse"
                        report_lines.append(f"    {comp.heuristic_a} vs {comp.heuristic_b}: {direction} (p={comp.p_value:.3f}, effect size={comp.effect_size:.2f})")
                    report_lines.append("")
        
        # Recommendations
        report_lines.append("Recommendations:")
        if total_solved > 0:
            best_coverage = overall_coverage['coverage_rate'].idxmax()
            report_lines.append(f"  Best coverage: {best_coverage}")
            
            if not speedups.empty:
                fastest = speedups.loc[speedups['geometric_mean_speedup'].idxmax(), 'heuristic']
                report_lines.append(f"  Fastest heuristic: {fastest}")
        else:
            report_lines.append("  Insufficient data for recommendations. Check experiment configuration.")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

def main():
    """Main entry point for analysis"""
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else "results/all_results.json"
    
    analyzer = ExperimentAnalyzer(results_file)
    
    try:
        analyzer.load_results()
        report = analyzer.generate_summary_report()
        print(report)
        
        # Save report
        report_path = Path("results") / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Analysis report saved to {report_path}")
        
    except FileNotFoundError as e:
        logger.error(f"Results file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 