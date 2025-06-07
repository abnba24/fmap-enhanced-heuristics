#!/usr/bin/env python3
"""
Generate a comprehensive metrics comparison table for FMAP heuristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob

def load_and_analyze_results():
    """Load all individual result files and create comprehensive comparison"""
    
    # Load all result files
    result_files = glob.glob("results/result_*.json")
    print(f"Loading {len(result_files)} result files...")
    
    all_results = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract key information
            config = data.get('config', {})
            search = data.get('search', {})
            plan = data.get('plan', {})
            
            # Map heuristic IDs to names (corrected to match Java source)
            heuristic_names = {
                1: "DTG",
                2: "DTG+Landmarks",
                3: "Inc_DTG+Landmarks",
                4: "Centroids",
                5: "MCS"
            }
            
            result = {
                'heuristic_id': config.get('heuristic'),
                'heuristic_name': heuristic_names.get(config.get('heuristic', 0), f"Unknown_{config.get('heuristic')}"),
                'domain': config.get('domain'),
                'problem': config.get('problem'),
                'agent_count': len(config.get('agents', [])),
                
                # Performance metrics
                'success': search.get('coverage', False),
                'wall_clock_time': search.get('wall_clock_time'),
                'cpu_time': search.get('cpu_time'),
                'peak_memory_mb': search.get('peak_memory_mb'),
                'search_nodes': search.get('search_nodes'),
                
                # Plan quality metrics
                'plan_found': plan.get('plan_found', False),
                'plan_length': plan.get('plan_length'),
                'makespan': plan.get('makespan'),
                
                # Error status
                'has_error': data.get('error_message') is not None
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return pd.DataFrame(all_results)

def generate_comprehensive_metrics_table(df):
    """Generate comprehensive metrics comparison table"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE HEURISTIC METRICS COMPARISON")
    print("="*100)
    
    # Filter to successful runs only for performance metrics
    successful_df = df[df['success'] == True].copy()
    
    if len(successful_df) == 0:
        print("No successful runs found!")
        return
    
    # Calculate comprehensive statistics for each heuristic
    metrics_comparison = []
    
    for heuristic in df['heuristic_name'].unique():
        # All experiments (for success rate)
        all_heur_data = df[df['heuristic_name'] == heuristic]
        # Successful experiments only (for performance metrics)
        successful_heur_data = successful_df[successful_df['heuristic_name'] == heuristic]
        
        if len(all_heur_data) > 0:
            stats = {
                'Heuristic': heuristic,
                
                # Success metrics
                'Total_Experiments': len(all_heur_data),
                'Successful_Runs': len(successful_heur_data),
                'Success_Rate_%': (len(successful_heur_data) / len(all_heur_data)) * 100,
                
                # Wall Clock Time metrics
                'Wall_Time_Mean_s': successful_heur_data['wall_clock_time'].mean() if len(successful_heur_data) > 0 else None,
                'Wall_Time_Median_s': successful_heur_data['wall_clock_time'].median() if len(successful_heur_data) > 0 else None,
                'Wall_Time_Std_s': successful_heur_data['wall_clock_time'].std() if len(successful_heur_data) > 0 else None,
                'Wall_Time_Min_s': successful_heur_data['wall_clock_time'].min() if len(successful_heur_data) > 0 else None,
                'Wall_Time_Max_s': successful_heur_data['wall_clock_time'].max() if len(successful_heur_data) > 0 else None,
                
                # CPU Time metrics
                'CPU_Time_Mean_s': successful_heur_data['cpu_time'].mean() if len(successful_heur_data) > 0 else None,
                'CPU_Time_Median_s': successful_heur_data['cpu_time'].median() if len(successful_heur_data) > 0 else None,
                'CPU_Time_Std_s': successful_heur_data['cpu_time'].std() if len(successful_heur_data) > 0 else None,
                
                # Memory metrics
                'Memory_Mean_MB': successful_heur_data['peak_memory_mb'].mean() if len(successful_heur_data) > 0 else None,
                'Memory_Median_MB': successful_heur_data['peak_memory_mb'].median() if len(successful_heur_data) > 0 else None,
                'Memory_Std_MB': successful_heur_data['peak_memory_mb'].std() if len(successful_heur_data) > 0 else None,
                'Memory_Min_MB': successful_heur_data['peak_memory_mb'].min() if len(successful_heur_data) > 0 else None,
                'Memory_Max_MB': successful_heur_data['peak_memory_mb'].max() if len(successful_heur_data) > 0 else None,
                
                # Search Nodes metrics
                'Search_Nodes_Mean': successful_heur_data['search_nodes'].mean() if len(successful_heur_data) > 0 else None,
                'Search_Nodes_Median': successful_heur_data['search_nodes'].median() if len(successful_heur_data) > 0 else None,
                'Search_Nodes_Std': successful_heur_data['search_nodes'].std() if len(successful_heur_data) > 0 else None,
                'Search_Nodes_Min': successful_heur_data['search_nodes'].min() if len(successful_heur_data) > 0 else None,
                'Search_Nodes_Max': successful_heur_data['search_nodes'].max() if len(successful_heur_data) > 0 else None,
                
                # Plan Quality metrics
                'Plan_Length_Mean': successful_heur_data['plan_length'].mean() if len(successful_heur_data) > 0 else None,
                'Plan_Length_Median': successful_heur_data['plan_length'].median() if len(successful_heur_data) > 0 else None,
                'Plan_Length_Std': successful_heur_data['plan_length'].std() if len(successful_heur_data) > 0 else None,
                
                'Makespan_Mean': successful_heur_data['makespan'].mean() if len(successful_heur_data) > 0 else None,
                'Makespan_Median': successful_heur_data['makespan'].median() if len(successful_heur_data) > 0 else None,
                'Makespan_Std': successful_heur_data['makespan'].std() if len(successful_heur_data) > 0 else None,
                
                # Domain coverage
                'Domains_Tested': all_heur_data['domain'].nunique(),
                'Problems_Tested': all_heur_data['problem'].nunique(),
                'Agent_Count_Range': f"{all_heur_data['agent_count'].min()}-{all_heur_data['agent_count'].max()}"
            }
            
            metrics_comparison.append(stats)
    
    # Create DataFrame and sort by success rate
    comparison_df = pd.DataFrame(metrics_comparison)
    comparison_df = comparison_df.sort_values('Success_Rate_%', ascending=False)
    
    return comparison_df

def create_summary_tables(comparison_df):
    """Create focused summary tables for different aspects"""
    
    print("\n" + "="*80)
    print("1. OVERALL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Overall performance table
    summary_cols = ['Heuristic', 'Total_Experiments', 'Successful_Runs', 'Success_Rate_%', 
                   'Wall_Time_Mean_s', 'Memory_Mean_MB', 'Search_Nodes_Mean', 'Plan_Length_Mean']
    
    summary_table = comparison_df[summary_cols].copy()
    print(summary_table.to_string(index=False, float_format='%.2f'))
    
    print("\n" + "="*80)
    print("2. DETAILED TIMING ANALYSIS")
    print("="*80)
    
    # Timing analysis table
    timing_cols = ['Heuristic', 'Wall_Time_Mean_s', 'Wall_Time_Median_s', 'Wall_Time_Std_s', 
                   'Wall_Time_Min_s', 'Wall_Time_Max_s', 'CPU_Time_Mean_s', 'CPU_Time_Median_s']
    
    timing_table = comparison_df[timing_cols].copy()
    print(timing_table.to_string(index=False, float_format='%.3f'))
    
    print("\n" + "="*80)
    print("3. MEMORY AND SEARCH EFFICIENCY")
    print("="*80)
    
    # Memory and search efficiency table
    efficiency_cols = ['Heuristic', 'Memory_Mean_MB', 'Memory_Median_MB', 'Memory_Std_MB',
                      'Search_Nodes_Mean', 'Search_Nodes_Median', 'Search_Nodes_Std']
    
    efficiency_table = comparison_df[efficiency_cols].copy()
    print(efficiency_table.to_string(index=False, float_format='%.1f'))
    
    print("\n" + "="*80)
    print("4. PLAN QUALITY ANALYSIS")
    print("="*80)
    
    # Plan quality table
    quality_cols = ['Heuristic', 'Success_Rate_%', 'Plan_Length_Mean', 'Plan_Length_Median', 
                   'Plan_Length_Std', 'Makespan_Mean', 'Makespan_Median']
    
    quality_table = comparison_df[quality_cols].copy()
    print(quality_table.to_string(index=False, float_format='%.2f'))
    
    return summary_table, timing_table, efficiency_table, quality_table

def create_ranking_analysis(comparison_df):
    """Create ranking analysis across different metrics"""
    
    print("\n" + "="*80)
    print("5. HEURISTIC RANKING BY DIFFERENT METRICS")
    print("="*80)
    
    rankings = {}
    
    # Success Rate Ranking
    success_ranking = comparison_df.nlargest(5, 'Success_Rate_%')[['Heuristic', 'Success_Rate_%']]
    rankings['Success Rate'] = success_ranking
    
    # Speed Ranking (lower is better)
    speed_ranking = comparison_df.nsmallest(5, 'Wall_Time_Mean_s')[['Heuristic', 'Wall_Time_Mean_s']]
    rankings['Speed (Wall Time)'] = speed_ranking
    
    # Memory Efficiency (lower is better)
    memory_ranking = comparison_df.nsmallest(5, 'Memory_Mean_MB')[['Heuristic', 'Memory_Mean_MB']]
    rankings['Memory Efficiency'] = memory_ranking
    
    # Search Efficiency (lower nodes is better)
    search_ranking = comparison_df.nsmallest(5, 'Search_Nodes_Mean')[['Heuristic', 'Search_Nodes_Mean']]
    rankings['Search Efficiency'] = search_ranking
    
    # Plan Quality (lower length might be better, but depends on domain)
    plan_ranking = comparison_df.nsmallest(5, 'Plan_Length_Mean')[['Heuristic', 'Plan_Length_Mean']]
    rankings['Plan Quality (Length)'] = plan_ranking
    
    for metric, ranking in rankings.items():
        print(f"\n{metric} Ranking:")
        print("-" * 40)
        for i, (_, row) in enumerate(ranking.iterrows(), 1):
            heuristic = row['Heuristic']
            value = row.iloc[1]  # Get the metric value
            if pd.notna(value):
                print(f"{i}. {heuristic}: {value:.3f}")
            else:
                print(f"{i}. {heuristic}: N/A")

def save_tables_to_files(comparison_df, summary_table, timing_table, efficiency_table, quality_table):
    """Save all tables to CSV files"""
    
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)
    
    # Save comprehensive table
    comparison_df.to_csv(output_dir / "comprehensive_metrics_comparison.csv", index=False)
    
    # Save summary tables
    summary_table.to_csv(output_dir / "performance_summary_table.csv", index=False)
    timing_table.to_csv(output_dir / "timing_analysis_table.csv", index=False)
    efficiency_table.to_csv(output_dir / "efficiency_analysis_table.csv", index=False)
    quality_table.to_csv(output_dir / "quality_analysis_table.csv", index=False)
    
    print(f"\n" + "="*80)
    print("TABLES SAVED TO FILES")
    print("="*80)
    print(f"📊 Comprehensive metrics: {output_dir}/comprehensive_metrics_comparison.csv")
    print(f"📈 Performance summary: {output_dir}/performance_summary_table.csv")
    print(f"⏱️  Timing analysis: {output_dir}/timing_analysis_table.csv")
    print(f"🧠 Efficiency analysis: {output_dir}/efficiency_analysis_table.csv")
    print(f"🎯 Quality analysis: {output_dir}/quality_analysis_table.csv")

def create_markdown_tables(comparison_df):
    """Create markdown formatted tables for README"""
    
    print(f"\n" + "="*80)
    print("MARKDOWN TABLES FOR README")
    print("="*80)
    
    # Main comparison table for README
    print("\n### Comprehensive Heuristic Performance Comparison\n")
    
    # Select key columns for main table
    main_cols = ['Heuristic', 'Success_Rate_%', 'Wall_Time_Mean_s', 'Memory_Mean_MB', 
                'Search_Nodes_Mean', 'Plan_Length_Mean', 'Total_Experiments']
    
    main_df = comparison_df[main_cols].copy()
    main_df.columns = ['Heuristic', 'Success Rate (%)', 'Avg Time (s)', 'Avg Memory (MB)', 
                      'Avg Search Nodes', 'Avg Plan Length', 'Total Experiments']
    
    # Format as markdown table
    print("| " + " | ".join(main_df.columns) + " |")
    print("|" + "|".join(["-" * (len(col)+2) for col in main_df.columns]) + "|")
    
    for _, row in main_df.iterrows():
        formatted_row = []
        for i, value in enumerate(row):
            if i == 0:  # Heuristic name
                formatted_row.append(f" **{value}** ")
            elif pd.isna(value):
                formatted_row.append(" N/A ")
            elif i in [1]:  # Success rate
                formatted_row.append(f" {value:.1f}% ")
            elif i in [2, 3]:  # Time and memory 
                formatted_row.append(f" {value:.2f} ")
            elif i in [4, 5]:  # Search nodes and plan length
                formatted_row.append(f" {value:.1f} ")
            else:  # Total experiments
                formatted_row.append(f" {int(value)} ")
        
        print("|" + "|".join(formatted_row) + "|")

def main():
    """Main function to generate all comparison tables"""
    
    print("FMAP Heuristic Metrics Comparison Analysis")
    print("=" * 50)
    
    # Load and analyze results
    df = load_and_analyze_results()
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Successful experiments: {len(df[df['success'] == True])}")
    print(f"Heuristics: {sorted(df['heuristic_name'].unique())}")
    print(f"Domains: {sorted(df['domain'].unique())}")
    
    # Generate comprehensive comparison table
    comparison_df = generate_comprehensive_metrics_table(df)
    
    if comparison_df.empty:
        print("Could not generate comparison table!")
        return
    
    # Create focused summary tables
    summary_table, timing_table, efficiency_table, quality_table = create_summary_tables(comparison_df)
    
    # Create ranking analysis
    create_ranking_analysis(comparison_df)
    
    # Save tables to files
    save_tables_to_files(comparison_df, summary_table, timing_table, efficiency_table, quality_table)
    
    # Create markdown tables
    create_markdown_tables(comparison_df)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main() 