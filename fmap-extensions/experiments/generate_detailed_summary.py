#!/usr/bin/env python3
"""
Generate a detailed summary report from the heuristic comparison analysis
"""

import pandas as pd
from pathlib import Path

def generate_detailed_summary():
    plots_dir = Path("results/plots")
    
    # Read the generated CSV files
    heuristic_summary = pd.read_csv(plots_dir / "heuristic_summary.csv")
    domain_analysis = pd.read_csv(plots_dir / "domain_analysis.csv")
    
    # Generate markdown report
    report = []
    report.append("# FMAP Heuristic Comparison Analysis Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    
    # Overall statistics
    total_experiments = heuristic_summary['Total_Experiments'].sum()
    total_successful = heuristic_summary['Successful'].sum()
    overall_success_rate = total_successful / total_experiments if total_experiments > 0 else 0
    
    report.append(f"- **Total Experiments**: {total_experiments}")
    report.append(f"- **Successful Experiments**: {total_successful}")
    report.append(f"- **Overall Success Rate**: {overall_success_rate:.1%}")
    report.append(f"- **Domains Tested**: {domain_analysis['Domain'].nunique()}")
    report.append(f"- **Heuristics Compared**: {len(heuristic_summary)}")
    report.append("")
    
    # Heuristic ranking
    report.append("## Heuristic Performance Ranking")
    report.append("")
    report.append("| Rank | Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |")
    report.append("|------|-----------|--------------|------------- |-----------------|-------------|")
    
    for i, row in heuristic_summary.iterrows():
        rank = i + 1
        heuristic = row['Heuristic']
        success_rate = f"{row['Success_Rate']:.1%}"
        avg_time = f"{row['Avg_Time']:.2f}" if pd.notna(row['Avg_Time']) else "N/A"
        avg_plan_length = f"{row['Avg_Plan_Length']:.1f}" if pd.notna(row['Avg_Plan_Length']) else "N/A"
        experiments = row['Total_Experiments']
        
        report.append(f"| {rank} | {heuristic} | {success_rate} | {avg_time} | {avg_plan_length} | {experiments} |")
    
    report.append("")
    
    # Detailed heuristic analysis
    report.append("## Detailed Heuristic Analysis")
    report.append("")
    
    for _, row in heuristic_summary.iterrows():
        heuristic = row['Heuristic']
        report.append(f"### {heuristic}")
        report.append("")
        report.append(f"- **Total Experiments**: {row['Total_Experiments']}")
        report.append(f"- **Successful**: {row['Successful']}")
        report.append(f"- **Success Rate**: {row['Success_Rate']:.1%}")
        
        if pd.notna(row['Avg_Time']):
            report.append(f"- **Average Execution Time**: {row['Avg_Time']:.3f} seconds")
            report.append(f"- **Median Execution Time**: {row['Median_Time']:.3f} seconds")
            report.append(f"- **Average Memory Usage**: {row['Avg_Memory']:.1f} MB")
            report.append(f"- **Average Plan Length**: {row['Avg_Plan_Length']:.1f} actions")
            report.append(f"- **Average Search Nodes**: {row['Avg_Search_Nodes']:.0f}")
        else:
            report.append("- **Performance**: No successful runs to analyze")
        
        report.append(f"- **Domains Tested**: {row['Domains_Tested']}")
        report.append(f"- **Problems Tested**: {row['Problems_Tested']}")
        report.append("")
    
    # Domain-specific analysis
    report.append("## Domain-Specific Performance")
    report.append("")
    
    for domain in domain_analysis['Domain'].unique():
        domain_data = domain_analysis[domain_analysis['Domain'] == domain]
        report.append(f"### {domain.title()} Domain")
        report.append("")
        
        report.append("| Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |")
        report.append("|-----------|--------------|------------- |-----------------|-------------|")
        
        for _, row in domain_data.iterrows():
            heuristic = row['Heuristic']
            success_rate = f"{row['Success_Rate']:.1%}"
            avg_time = f"{row['Avg_Time']:.2f}" if pd.notna(row['Avg_Time']) else "N/A"
            avg_plan_length = f"{row['Avg_Plan_Length']:.1f}" if pd.notna(row['Avg_Plan_Length']) else "N/A"
            experiments = row['Experiments']
            
            report.append(f"| {heuristic} | {success_rate} | {avg_time} | {avg_plan_length} | {experiments} |")
        
        report.append("")
    
    # Key insights
    report.append("## Key Insights")
    report.append("")
    
    # Best performing heuristic
    best_heuristic = heuristic_summary.iloc[0]
    report.append(f"1. **Best Overall Heuristic**: {best_heuristic['Heuristic']} with {best_heuristic['Success_Rate']:.1%} success rate")
    report.append("")
    
    # Speed analysis
    speed_ranking = heuristic_summary[heuristic_summary['Avg_Time'].notna()].sort_values('Avg_Time')
    if len(speed_ranking) > 0:
        fastest = speed_ranking.iloc[0]
        report.append(f"2. **Fastest Heuristic**: {fastest['Heuristic']} with average time of {fastest['Avg_Time']:.3f} seconds")
        report.append("")
    
    # Domain analysis
    domain_success_rates = domain_analysis.groupby('Domain')['Success_Rate'].mean().sort_values(ascending=False)
    if len(domain_success_rates) > 0:
        best_domain = domain_success_rates.index[0]
        worst_domain = domain_success_rates.index[-1]
        report.append(f"3. **Most Successful Domain**: {best_domain.title()} ({domain_success_rates.iloc[0]:.1%} average success rate)")
        report.append(f"4. **Most Challenging Domain**: {worst_domain.title()} ({domain_success_rates.iloc[-1]:.1%} average success rate)")
        report.append("")
    
    # Complexity analysis
    report.append("5. **Complexity Observations**:")
    complex_domains = domain_analysis[domain_analysis['Avg_Agents'] > 5]['Domain'].unique()
    if len(complex_domains) > 0:
        report.append(f"   - High-complexity domains (>5 agents): {', '.join(complex_domains)}")
    simple_domains = domain_analysis[domain_analysis['Avg_Agents'] <= 3]['Domain'].unique()
    if len(simple_domains) > 0:
        report.append(f"   - Simple domains (≤3 agents): {', '.join(simple_domains)}")
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("1. **For General Use**: Consider using the top-performing heuristics with high success rates")
    report.append("2. **For Speed-Critical Applications**: Use the fastest successful heuristics")
    report.append("3. **For Complex Problems**: Test multiple heuristics as performance varies by domain")
    report.append("4. **Future Research**: Focus on improving performance for challenging domains")
    report.append("")
    
    # Save the report
    report_content = "\n".join(report)
    
    with open(plots_dir / "detailed_analysis_report.md", "w") as f:
        f.write(report_content)
    
    # Also save as text
    with open(plots_dir / "detailed_analysis_report.txt", "w") as f:
        f.write(report_content)
    
    print("Detailed analysis report generated!")
    print(f"Files saved:")
    print(f"- {plots_dir / 'detailed_analysis_report.md'}")
    print(f"- {plots_dir / 'detailed_analysis_report.txt'}")
    print("")
    print("="*60)
    print(report_content)

if __name__ == "__main__":
    generate_detailed_summary() 