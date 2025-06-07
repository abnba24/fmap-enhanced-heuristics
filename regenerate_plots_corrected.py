#!/usr/bin/env python3
"""
Regenerate All Plots with Corrected Heuristic Names

This script regenerates all the plots using the corrected data
to ensure all visualizations show the proper heuristic names.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CorrectedPlotGenerator:
    """Generate plots with corrected heuristic names"""
    
    def __init__(self, results_dir="analysis_outputs/results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        
        # Load corrected data
        self.df = pd.read_csv(self.plots_dir / "comprehensive_metrics_comparison_corrected.csv")
        print(f"Loaded corrected data: {len(self.df)} heuristics")
        
    def generate_all_corrected_plots(self):
        """Generate all plots with corrected data"""
        print("🎨 Generating all plots with corrected heuristic names...")
        
        # 1. Main heuristic comparison
        self._plot_main_comparison()
        
        # 2. Success rates analysis
        self._plot_success_analysis()
        
        # 3. Performance matrices
        self._plot_performance_matrices()
        
        # 4. Scalability analysis
        self._plot_scalability_analysis()
        
        # 5. Domain-specific performance
        self._plot_domain_performance()
        
        print("✅ All corrected plots generated successfully!")
    
    def _plot_main_comparison(self):
        """Main heuristic comparison plot"""
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Success Rate
        success_rates = self.df['Success_Rate_%'] / 100
        colors = ['#2ecc71' if x >= 0.8 else '#f39c12' if x >= 0.5 else '#e74c3c' for x in success_rates]
        bars1 = ax1.bar(range(len(self.df)), success_rates, color=colors)
        ax1.set_title('Success Rate by Heuristic', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(self.df)))
        ax1.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(success_rates):
            ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Average Execution Time
        times = self.df['Wall_Time_Mean_s'].fillna(0)
        bars2 = ax2.bar(range(len(self.df)), times, color=sns.color_palette("viridis", len(self.df)))
        ax2.set_title('Average Execution Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xticks(range(len(self.df)))
        ax2.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        ax2.set_yscale('log')
        
        for i, v in enumerate(times):
            if v > 0:
                ax2.text(i, v * 1.1, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory Usage
        memory = self.df['Memory_Mean_MB'].fillna(0)
        bars3 = ax3.bar(range(len(self.df)), memory, color=sns.color_palette("plasma", len(self.df)))
        ax3.set_title('Average Memory Usage', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_xticks(range(len(self.df)))
        ax3.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        
        for i, v in enumerate(memory):
            if v > 0:
                ax3.text(i, v + max(memory) * 0.02, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Plan Length
        plan_lengths = self.df['Plan_Length_Mean'].fillna(0)
        bars4 = ax4.bar(range(len(self.df)), plan_lengths, color=sns.color_palette("coolwarm", len(self.df)))
        ax4.set_title('Average Plan Length', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Plan Length (actions)')
        ax4.set_xticks(range(len(self.df)))
        ax4.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        
        for i, v in enumerate(plan_lengths):
            if v > 0:
                ax4.text(i, v + max(plan_lengths) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Search Nodes
        search_nodes = self.df['Search_Nodes_Mean'].fillna(1)
        bars5 = ax5.bar(range(len(self.df)), search_nodes, color=sns.color_palette("Set1", len(self.df)))
        ax5.set_title('Average Search Nodes', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Search Nodes')
        ax5.set_yscale('log')
        ax5.set_xticks(range(len(self.df)))
        ax5.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        
        # 6. Performance Ranking
        # Create composite score: success_rate * 0.4 + (1/time) * 0.3 + (1/memory) * 0.3
        scores = []
        for _, row in self.df.iterrows():
            score = (row['Success_Rate_%'] / 100) * 0.4
            if pd.notna(row['Wall_Time_Mean_s']) and row['Wall_Time_Mean_s'] > 0:
                score += (1 / row['Wall_Time_Mean_s']) * 0.3
            if pd.notna(row['Memory_Mean_MB']) and row['Memory_Mean_MB'] > 0:
                score += (1 / row['Memory_Mean_MB']) * 0.3
            scores.append(score)
        
        bars6 = ax6.barh(range(len(self.df)), scores, color=sns.color_palette("Set2", len(self.df)))
        ax6.set_title('Overall Performance Ranking', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Composite Performance Score')
        ax6.set_yticks(range(len(self.df)))
        ax6.set_yticklabels(self.df['Heuristic'])
        
        for i, v in enumerate(scores):
            ax6.text(v + max(scores) * 0.01, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heuristic_comparison_corrected_final.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Generated main comparison plot")
    
    def _plot_success_analysis(self):
        """Success rate analysis plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Success rates with color coding
        success_rates = self.df['Success_Rate_%'] / 100
        colors = ['#27ae60' if x >= 0.8 else '#f39c12' if x >= 0.5 else '#e74c3c' for x in success_rates]
        
        bars = ax1.bar(range(len(self.df)), success_rates, color=colors)
        ax1.set_title('Success Rate Analysis', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(range(len(self.df)))
        ax1.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        
        # Add value labels and success indicators
        for i, (v, heuristic) in enumerate(zip(success_rates, self.df['Heuristic'])):
            ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            if v == 1.0:
                ax1.text(i, v - 0.05, '★', ha='center', va='top', fontsize=20, color='gold')
        
        # Success vs Performance scatter
        times = self.df['Wall_Time_Mean_s'].fillna(self.df['Wall_Time_Mean_s'].max())
        ax2.scatter(times, success_rates, s=200, alpha=0.7, c=colors)
        
        for i, heuristic in enumerate(self.df['Heuristic']):
            ax2.annotate(heuristic, (times.iloc[i], success_rates.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax2.set_xlabel('Average Execution Time (seconds)')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate vs Execution Time', fontsize=16, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'success_analysis_corrected.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Generated success analysis plot")
    
    def _plot_performance_matrices(self):
        """Performance matrices visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Create performance matrix
        metrics = ['Success_Rate_%', 'Wall_Time_Mean_s', 'Memory_Mean_MB', 'Plan_Length_Mean', 'Search_Nodes_Mean']
        matrix_data = self.df[['Heuristic'] + metrics].set_index('Heuristic')
        
        # Normalize data for heatmap (lower is better for time, memory, nodes)
        normalized_data = matrix_data.copy()
        normalized_data['Success_Rate_%'] = normalized_data['Success_Rate_%'] / 100
        
        # Invert time, memory, and nodes (so higher normalized value = better performance)
        for col in ['Wall_Time_Mean_s', 'Memory_Mean_MB', 'Search_Nodes_Mean']:
            if col in normalized_data.columns:
                max_val = normalized_data[col].max()
                normalized_data[col] = (max_val - normalized_data[col]) / max_val
        
        # Normalize plan length (shorter is generally better)
        if 'Plan_Length_Mean' in normalized_data.columns:
            max_val = normalized_data['Plan_Length_Mean'].max()
            normalized_data['Plan_Length_Mean'] = (max_val - normalized_data['Plan_Length_Mean']) / max_val
        
        # Performance heatmap
        sns.heatmap(normalized_data.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Normalized Performance'})
        ax1.set_title('Performance Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Heuristics')
        ax1.set_ylabel('Metrics')
        
        # Time vs Memory scatter
        times = self.df['Wall_Time_Mean_s'].fillna(0)
        memory = self.df['Memory_Mean_MB'].fillna(0)
        success_rates = self.df['Success_Rate_%'] / 100
        
        scatter = ax2.scatter(memory, times, s=success_rates*300, alpha=0.7, c=success_rates, cmap='RdYlGn')
        for i, heuristic in enumerate(self.df['Heuristic']):
            ax2.annotate(heuristic, (memory.iloc[i], times.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Time vs Memory (bubble size = success rate)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        plt.colorbar(scatter, ax=ax2, label='Success Rate')
        
        # Plan quality analysis
        plan_lengths = self.df['Plan_Length_Mean'].fillna(0)
        ax3.bar(range(len(self.df)), plan_lengths, color=sns.color_palette("viridis", len(self.df)))
        ax3.set_title('Plan Quality Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Average Plan Length')
        ax3.set_xticks(range(len(self.df)))
        ax3.set_xticklabels(self.df['Heuristic'], rotation=45, ha='right')
        
        # Efficiency radar-like plot (using bar chart)
        efficiency_metrics = ['Success_Rate_%', 'Wall_Time_Mean_s', 'Memory_Mean_MB']
        best_heuristic = self.df.loc[self.df['Success_Rate_%'].idxmax()]
        
        ax4.text(0.5, 0.8, f"Best Overall: {best_heuristic['Heuristic']}", 
                transform=ax4.transAxes, ha='center', fontsize=16, fontweight='bold')
        ax4.text(0.5, 0.6, f"Success Rate: {best_heuristic['Success_Rate_%']:.1f}%", 
                transform=ax4.transAxes, ha='center', fontsize=12)
        ax4.text(0.5, 0.5, f"Avg Time: {best_heuristic['Wall_Time_Mean_s']:.2f}s", 
                transform=ax4.transAxes, ha='center', fontsize=12)
        ax4.text(0.5, 0.4, f"Avg Memory: {best_heuristic['Memory_Mean_MB']:.1f} MB", 
                transform=ax4.transAxes, ha='center', fontsize=12)
        ax4.set_title('Best Performing Heuristic', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_matrices_corrected.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Generated performance matrices plot")

    def _plot_scalability_analysis(self):
        """Scalability analysis plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Agent count analysis (from domain data)
        domain_df = pd.read_csv(self.plots_dir / "domain_analysis_corrected.csv")

        # Success by agent count
        agent_success = domain_df.groupby(['Avg_Agents', 'Heuristic'])['Success_Rate'].mean().unstack()
        agent_success.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Success Rate by Agent Count', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_xlabel('Average Agent Count')
        ax1.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=0)

        # Time complexity analysis
        time_complexity = domain_df.groupby(['Avg_Agents', 'Heuristic'])['Avg_Time'].mean().unstack()
        time_complexity.plot(kind='line', ax=ax2, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Execution Time vs Agent Count', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Execution Time (seconds)')
        ax2.set_xlabel('Average Agent Count')
        ax2.set_yscale('log')
        ax2.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'scalability_analysis_corrected.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Generated scalability analysis plot")

    def _plot_domain_performance(self):
        """Domain-specific performance analysis"""
        domain_df = pd.read_csv(self.plots_dir / "domain_analysis_corrected.csv")

        domains = domain_df['Domain'].unique()
        n_domains = len(domains)

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

            domain_data = domain_df[domain_df['Domain'] == domain]

            # Success rate by heuristic for this domain
            heuristics = domain_data['Heuristic'].values
            success_rates = domain_data['Success_Rate'].values

            colors = ['#27ae60' if x >= 0.8 else '#f39c12' if x >= 0.5 else '#e74c3c' for x in success_rates]
            bars = axes[i].bar(range(len(heuristics)), success_rates, color=colors)

            axes[i].set_title(f'{domain.title()} Domain\nSuccess Rate by Heuristic',
                             fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Success Rate')
            axes[i].set_ylim(0, 1.1)
            axes[i].set_xticks(range(len(heuristics)))
            axes[i].set_xticklabels(heuristics, rotation=45, ha='right')

            # Add value labels
            for j, v in enumerate(success_rates):
                axes[i].text(j, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
                if v == 1.0:
                    axes[i].text(j, v - 0.05, '★', ha='center', va='top', fontsize=16, color='gold')

        # Hide unused subplots
        for i in range(len(domains), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'domain_performance_corrected.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Generated domain performance plot")

    def create_summary_report(self):
        """Create a summary report of the corrected analysis"""
        print("📄 Creating summary report...")

        report_file = self.plots_dir / "corrected_analysis_summary.md"

        with open(report_file, 'w') as f:
            f.write("# 📊 FMAP Heuristics Analysis - Corrected Results\n\n")
            f.write("*Generated with correct heuristic names from Java source mapping*\n\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}  \n")
            f.write(f"**Total Heuristics:** {len(self.df)}  \n")
            f.write(f"**Metrics Analyzed:** {len(self.df.columns)}  \n\n")

            f.write("## 🎯 Key Findings\n\n")

            # Best performers
            best_success = self.df.loc[self.df['Success_Rate_%'].idxmax()]
            fastest = self.df.loc[self.df['Wall_Time_Mean_s'].idxmin()]
            most_efficient = self.df.loc[self.df['Memory_Mean_MB'].idxmin()]

            f.write(f"### 🏆 Top Performers\n\n")
            f.write(f"- **Highest Success Rate:** {best_success['Heuristic']} ({best_success['Success_Rate_%']:.1f}%)\n")
            f.write(f"- **Fastest Execution:** {fastest['Heuristic']} ({fastest['Wall_Time_Mean_s']:.2f}s average)\n")
            f.write(f"- **Most Memory Efficient:** {most_efficient['Heuristic']} ({most_efficient['Memory_Mean_MB']:.1f} MB average)\n\n")

            f.write("### 📈 Performance Summary\n\n")
            f.write("| Rank | Heuristic | Success Rate | Avg Time | Memory | Plan Length |\n")
            f.write("|------|-----------|--------------|----------|--------|-------------|\n")

            # Sort by success rate for ranking
            sorted_df = self.df.sort_values('Success_Rate_%', ascending=False)
            for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                f.write(f"| {emoji} | **{row['Heuristic']}** | {row['Success_Rate_%']:.1f}% | {row['Wall_Time_Mean_s']:.2f}s | {row['Memory_Mean_MB']:.1f} MB | {row['Plan_Length_Mean']:.1f} |\n")

            f.write(f"\n### 🔧 Corrections Made\n\n")
            f.write("The following heuristic name corrections were applied:\n\n")
            f.write("- ❌ `DTG_Only` → ✅ `DTG`\n")
            f.write("- ❌ `Inc_DTG_Only` → ✅ `Inc_DTG+Landmarks`\n")
            f.write("- ❌ `FF_Heuristic` → ✅ `MCS`\n")
            f.write("- ❌ Incorrect ID 4 mapping → ✅ `Centroids`\n\n")

            f.write("### 📁 Generated Files\n\n")
            f.write("**Corrected Data:**\n")
            for csv_file in sorted(self.plots_dir.glob("*_corrected.csv")):
                f.write(f"- `{csv_file.name}`\n")

            f.write(f"\n**Beautiful Tables:**\n")
            for html_file in sorted(self.plots_dir.glob("*_beautiful.html")):
                f.write(f"- `{html_file.name}`\n")

            f.write(f"\n**Corrected Plots:**\n")
            for plot_file in sorted(self.plots_dir.glob("*_corrected*.png")):
                f.write(f"- `{plot_file.name}`\n")

            f.write(f"\n---\n*Analysis completed with FMAP Enhanced Heuristics Framework*\n")

        print(f"  ✅ Created summary report: {report_file}")

def main():
    """Main function"""
    generator = CorrectedPlotGenerator()
    generator.generate_all_corrected_plots()
    generator.create_summary_report()

    print("\n🎉 All corrected plots and analysis complete!")
    print("📁 Check the analysis_outputs/results/plots/ directory for:")
    print("  - Corrected CSV files (*_corrected.csv)")
    print("  - Beautiful HTML tables (*_beautiful.html)")
    print("  - Beautiful Markdown tables (*_beautiful.md)")
    print("  - Corrected plots (*_corrected*.png)")
    print("  - Summary report (corrected_analysis_summary.md)")

if __name__ == "__main__":
    main()
