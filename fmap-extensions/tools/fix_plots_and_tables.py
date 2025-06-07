#!/usr/bin/env python3
"""
Fix Heuristic Names in Plots and Convert CSVs to Nice Tables

This script:
1. Fixes incorrect heuristic names in existing CSV files
2. Regenerates plots with correct heuristic names
3. Converts CSV files to nicely formatted HTML and Markdown tables
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PlotAndTableFixer:
    """Fix heuristic names and create nice tables"""
    
    def __init__(self, results_dir="analysis_outputs/results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Correct heuristic name mapping (from Java source)
        self.heuristic_name_corrections = {
            "DTG_Only": "DTG",
            "Inc_DTG_Only": "Inc_DTG+Landmarks", 
            "FF_Heuristic": "MCS",
            "Inc_DTG+Landmarks": "Centroids"  # This was incorrectly mapped to ID 4
        }
        
        # Correct mapping for reference
        self.correct_heuristic_names = {
            1: "DTG",
            2: "DTG+Landmarks", 
            3: "Inc_DTG+Landmarks",
            4: "Centroids",
            5: "MCS"
        }
        
    def fix_heuristic_names_in_text(self, text):
        """Fix heuristic names in any text"""
        for old_name, new_name in self.heuristic_name_corrections.items():
            text = text.replace(old_name, new_name)
        return text
    
    def fix_csv_files(self):
        """Fix heuristic names in all CSV files"""
        print("🔧 Fixing heuristic names in CSV files...")
        
        csv_files = list(self.plots_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            print(f"  Fixing: {csv_file.name}")
            
            try:
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Fix heuristic names in 'Heuristic' column if it exists
                if 'Heuristic' in df.columns:
                    df['Heuristic'] = df['Heuristic'].replace(self.heuristic_name_corrections)
                
                # Fix any other columns that might contain heuristic names
                for col in df.columns:
                    if df[col].dtype == 'object':  # String columns
                        df[col] = df[col].astype(str).apply(self.fix_heuristic_names_in_text)
                
                # Save back to CSV
                df.to_csv(csv_file, index=False)
                print(f"    ✅ Fixed {csv_file.name}")
                
            except Exception as e:
                print(f"    ❌ Error fixing {csv_file.name}: {e}")
    
    def load_experiment_data(self):
        """Load experiment data from JSON files with correct heuristic mapping"""
        print("📊 Loading experiment data...")
        
        result_files = list(self.results_dir.glob("result_*.json"))
        print(f"Found {len(result_files)} result files")
        
        results = []
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                config = data.get('config', {})
                search = data.get('search', {})
                plan = data.get('plan', {})
                
                # Use correct heuristic mapping
                heuristic_id = config.get('heuristic')
                heuristic_name = self.correct_heuristic_names.get(heuristic_id, f"H{heuristic_id}")
                
                result = {
                    'domain': config.get('domain'),
                    'problem': config.get('problem'),
                    'heuristic_id': heuristic_id,
                    'heuristic_name': heuristic_name,
                    'agent_count': len(config.get('agents', [])),
                    'success': search.get('coverage', False),
                    'wall_clock_time': search.get('wall_clock_time'),
                    'cpu_time': search.get('cpu_time'),
                    'peak_memory_mb': search.get('peak_memory_mb'),
                    'search_nodes': search.get('search_nodes'),
                    'plan_found': plan.get('plan_found', False),
                    'plan_length': plan.get('plan_length'),
                    'makespan': plan.get('makespan')
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        df = pd.DataFrame(results)
        print(f"Loaded {len(df)} experiments with correct heuristic names")
        return df
    
    def regenerate_plots(self, df):
        """Regenerate plots with correct heuristic names"""
        print("🎨 Regenerating plots with correct heuristic names...")
        
        # 1. Heuristic Performance Summary
        self._plot_heuristic_summary(df)
        
        # 2. Success Rate Analysis
        self._plot_success_rates(df)
        
        # 3. Performance by Domain
        self._plot_domain_performance(df)
        
        # 4. Execution Time Analysis
        self._plot_execution_times(df)
        
        # 5. Comprehensive Comparison
        self._plot_comprehensive_comparison(df)
        
        print("✅ All plots regenerated with correct heuristic names")
    
    def _plot_heuristic_summary(self, df):
        """Plot heuristic performance summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Success rates
        success_by_heuristic = df.groupby('heuristic_name')['success'].mean()
        bars1 = ax1.bar(range(len(success_by_heuristic)), success_by_heuristic.values,
                       color=sns.color_palette("husl", len(success_by_heuristic)))
        ax1.set_title('Success Rate by Heuristic', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(success_by_heuristic)))
        ax1.set_xticklabels(success_by_heuristic.index, rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(success_by_heuristic.values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Execution times (successful runs only)
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='wall_clock_time', ax=ax2)
            ax2.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Wall Clock Time (seconds)')
            ax2.set_yscale('log')
            ax2.tick_params(axis='x', rotation=45)
        
        # Memory usage
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='peak_memory_mb', ax=ax3)
            ax3.set_title('Memory Usage Distribution', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Peak Memory (MB)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Plan length
        if len(successful_df) > 0:
            sns.boxplot(data=successful_df, x='heuristic_name', y='plan_length', ax=ax4)
            ax4.set_title('Plan Length Distribution', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Plan Length (actions)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heuristic_comparison_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rates(self, df):
        """Plot success rates by domain and heuristic"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall success rates
        success_by_heuristic = df.groupby('heuristic_name')['success'].mean()
        bars = ax1.bar(range(len(success_by_heuristic)), success_by_heuristic.values,
                      color=sns.color_palette("Set2", len(success_by_heuristic)))
        ax1.set_title('Overall Success Rate by Heuristic', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(success_by_heuristic)))
        ax1.set_xticklabels(success_by_heuristic.index, rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(success_by_heuristic.values):
            ax1.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Success rates by domain
        domain_success = df.groupby(['domain', 'heuristic_name'])['success'].mean().unstack()
        domain_success.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Success Rate by Domain and Heuristic', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate')
        ax2.legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'success_rates_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_domain_performance(self, df):
        """Plot performance by domain"""
        domains = df['domain'].unique()
        n_domains = len(domains)

        if n_domains == 0:
            return

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

            domain_data = df[df['domain'] == domain]
            domain_success = domain_data.groupby('heuristic_name')['success'].mean()

            bars = axes[i].bar(range(len(domain_success)), domain_success.values,
                              color=sns.color_palette("Set3", len(domain_success)))
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
        plt.savefig(self.plots_dir / 'domain_performance_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_execution_times(self, df):
        """Plot execution time analysis"""
        successful_df = df[df['success'] == True]
        if len(successful_df) == 0:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Box plot by heuristic
        sns.boxplot(data=successful_df, x='heuristic_name', y='wall_clock_time', ax=ax1)
        ax1.set_title('Execution Time by Heuristic', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Wall Clock Time (seconds)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='x', rotation=45)

        # Memory vs Time scatter
        for heuristic in successful_df['heuristic_name'].unique():
            heur_data = successful_df[successful_df['heuristic_name'] == heuristic]
            ax2.scatter(heur_data['peak_memory_mb'], heur_data['wall_clock_time'],
                       label=heuristic, alpha=0.7, s=30)
        ax2.set_xlabel('Peak Memory (MB)')
        ax2.set_ylabel('Wall Clock Time (seconds)')
        ax2.set_title('Memory vs Execution Time', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Agent count vs time
        for heuristic in successful_df['heuristic_name'].unique():
            heur_data = successful_df[successful_df['heuristic_name'] == heuristic]
            ax3.scatter(heur_data['agent_count'], heur_data['wall_clock_time'],
                       label=heuristic, alpha=0.7, s=30)
        ax3.set_xlabel('Agent Count')
        ax3.set_ylabel('Wall Clock Time (seconds)')
        ax3.set_title('Scalability: Agent Count vs Time', fontsize=14, fontweight='bold')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plan length vs time
        for heuristic in successful_df['heuristic_name'].unique():
            heur_data = successful_df[successful_df['heuristic_name'] == heuristic]
            ax4.scatter(heur_data['plan_length'], heur_data['wall_clock_time'],
                       label=heuristic, alpha=0.7, s=30)
        ax4.set_xlabel('Plan Length (actions)')
        ax4.set_ylabel('Wall Clock Time (seconds)')
        ax4.set_title('Plan Quality vs Time', fontsize=14, fontweight='bold')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'execution_analysis_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_comprehensive_comparison(self, df):
        """Create comprehensive comparison plot"""
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))

        # 1. Success Rate
        success_by_heuristic = df.groupby('heuristic_name')['success'].mean()
        bars1 = ax1.bar(range(len(success_by_heuristic)), success_by_heuristic.values,
                       color=sns.color_palette("husl", len(success_by_heuristic)))
        ax1.set_title('Success Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        ax1.set_xticks(range(len(success_by_heuristic)))
        ax1.set_xticklabels(success_by_heuristic.index, rotation=45, ha='right')

        for i, v in enumerate(success_by_heuristic.values):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. Average execution time
        successful_df = df[df['success'] == True]
        if len(successful_df) > 0:
            avg_time = successful_df.groupby('heuristic_name')['wall_clock_time'].mean()
            bars2 = ax2.bar(range(len(avg_time)), avg_time.values,
                           color=sns.color_palette("viridis", len(avg_time)))
            ax2.set_title('Average Execution Time', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_xticks(range(len(avg_time)))
            ax2.set_xticklabels(avg_time.index, rotation=45, ha='right')

            for i, v in enumerate(avg_time.values):
                ax2.text(i, v + max(avg_time.values) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

        # 3. Average memory usage
        if len(successful_df) > 0:
            avg_memory = successful_df.groupby('heuristic_name')['peak_memory_mb'].mean()
            bars3 = ax3.bar(range(len(avg_memory)), avg_memory.values,
                           color=sns.color_palette("plasma", len(avg_memory)))
            ax3.set_title('Average Memory Usage', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Memory (MB)')
            ax3.set_xticks(range(len(avg_memory)))
            ax3.set_xticklabels(avg_memory.index, rotation=45, ha='right')

            for i, v in enumerate(avg_memory.values):
                ax3.text(i, v + max(avg_memory.values) * 0.02, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')

        # 4. Success rate by domain heatmap
        domain_success = df.groupby(['domain', 'heuristic_name'])['success'].mean().unstack()
        sns.heatmap(domain_success, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4)
        ax4.set_title('Success Rate by Domain', fontsize=12, fontweight='bold')

        # 5. Plan length comparison
        if len(successful_df) > 0:
            avg_plan_length = successful_df.groupby('heuristic_name')['plan_length'].mean()
            bars5 = ax5.bar(range(len(avg_plan_length)), avg_plan_length.values,
                           color=sns.color_palette("coolwarm", len(avg_plan_length)))
            ax5.set_title('Average Plan Length', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Plan Length (actions)')
            ax5.set_xticks(range(len(avg_plan_length)))
            ax5.set_xticklabels(avg_plan_length.index, rotation=45, ha='right')

            for i, v in enumerate(avg_plan_length.values):
                ax5.text(i, v + max(avg_plan_length.values) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

        # 6. Search nodes comparison
        if len(successful_df) > 0:
            avg_nodes = successful_df.groupby('heuristic_name')['search_nodes'].mean()
            bars6 = ax6.bar(range(len(avg_nodes)), avg_nodes.values,
                           color=sns.color_palette("Set1", len(avg_nodes)))
            ax6.set_title('Average Search Nodes', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Search Nodes')
            ax6.set_yscale('log')
            ax6.set_xticks(range(len(avg_nodes)))
            ax6.set_xticklabels(avg_nodes.index, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'comprehensive_comparison_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_nice_tables(self):
        """Convert CSV files to nicely formatted HTML and Markdown tables"""
        print("📋 Creating nicely formatted tables...")

        csv_files = list(self.plots_dir.glob("*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                base_name = csv_file.stem

                # Create HTML table
                html_file = self.plots_dir / f"{base_name}.html"
                self._create_html_table(df, html_file, base_name)

                # Create Markdown table
                md_file = self.plots_dir / f"{base_name}.md"
                self._create_markdown_table(df, md_file, base_name)

                print(f"  ✅ Created tables for {csv_file.name}")

            except Exception as e:
                print(f"  ❌ Error creating tables for {csv_file.name}: {e}")

    def _create_html_table(self, df, output_file, title):
        """Create a nicely formatted HTML table"""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title.replace('_', ' ').title()}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            padding: 12px 8px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        td {{
            padding: 10px 8px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
        }}
        .numeric {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        .heuristic-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title.replace('_', ' ').title()}</h1>
        {self._df_to_html_with_formatting(df)}
        <div class="footer">
            Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | FMAP Enhanced Heuristics Analysis
        </div>
    </div>
</body>
</html>
"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

    def _df_to_html_with_formatting(self, df):
        """Convert DataFrame to HTML with custom formatting"""
        html = "<table>\n<thead>\n<tr>\n"

        # Headers
        for col in df.columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"

        # Rows
        for _, row in df.iterrows():
            html += "<tr>\n"
            for i, (col, value) in enumerate(row.items()):
                css_class = ""
                if col == 'Heuristic':
                    css_class = 'class="heuristic-name"'
                elif pd.api.types.is_numeric_dtype(df[col]):
                    css_class = 'class="numeric"'
                    if pd.notna(value):
                        if isinstance(value, float):
                            if col.lower().find('rate') != -1 or col.lower().find('percent') != -1:
                                value = f"{value:.1%}" if value <= 1 else f"{value:.1f}%"
                            elif col.lower().find('time') != -1:
                                value = f"{value:.2f}"
                            elif col.lower().find('memory') != -1 or col.lower().find('mb') != -1:
                                value = f"{value:.1f}"
                            else:
                                value = f"{value:.2f}"

                html += f"<td {css_class}>{value}</td>\n"
            html += "</tr>\n"

        html += "</tbody>\n</table>"
        return html

    def _create_markdown_table(self, df, output_file, title):
        """Create a nicely formatted Markdown table"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {title.replace('_', ' ').title()}\n\n")
            f.write(f"*Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            # Create markdown table
            headers = [col.replace('_', ' ').title() for col in df.columns]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["-" * (len(header) + 2) for header in headers]) + "|\n")

            for _, row in df.iterrows():
                formatted_row = []
                for col, value in row.items():
                    if pd.isna(value):
                        formatted_row.append(" N/A ")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        if isinstance(value, float):
                            if col.lower().find('rate') != -1 or col.lower().find('percent') != -1:
                                formatted_row.append(f" {value:.1%} " if value <= 1 else f" {value:.1f}% ")
                            elif col.lower().find('time') != -1:
                                formatted_row.append(f" {value:.2f} ")
                            elif col.lower().find('memory') != -1 or col.lower().find('mb') != -1:
                                formatted_row.append(f" {value:.1f} ")
                            else:
                                formatted_row.append(f" {value:.2f} ")
                        else:
                            formatted_row.append(f" {value} ")
                    else:
                        if col == 'Heuristic':
                            formatted_row.append(f" **{value}** ")
                        else:
                            formatted_row.append(f" {value} ")

                f.write("|" + "|".join(formatted_row) + "|\n")

            f.write(f"\n---\n*FMAP Enhanced Heuristics Analysis*\n")

    def run_complete_fix(self):
        """Run the complete fixing process"""
        print("🚀 Starting comprehensive plot and table fixing...")
        print("=" * 60)

        # Step 1: Fix CSV files
        self.fix_csv_files()

        # Step 2: Load corrected data
        df = self.load_experiment_data()

        # Step 3: Regenerate plots
        self.regenerate_plots(df)

        # Step 4: Create nice tables
        self.create_nice_tables()

        print("\n" + "=" * 60)
        print("✅ COMPLETE! All fixes applied successfully:")
        print(f"  📊 Fixed CSV files with correct heuristic names")
        print(f"  🎨 Regenerated plots with correct heuristic names")
        print(f"  📋 Created nicely formatted HTML and Markdown tables")
        print(f"  📁 All files saved to: {self.plots_dir}")

        # List generated files
        print(f"\n📁 Generated Files:")
        for file_type, pattern in [("PNG plots", "*.png"), ("HTML tables", "*.html"), ("Markdown tables", "*.md"), ("CSV data", "*.csv")]:
            files = list(self.plots_dir.glob(pattern))
            if files:
                print(f"  {file_type}: {len(files)} files")
                for f in sorted(files):
                    print(f"    - {f.name}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Fix heuristic names in plots and create nice tables')
    parser.add_argument('--results-dir', default='analysis_outputs/results',
                       help='Directory containing results and plots')

    args = parser.parse_args()

    fixer = PlotAndTableFixer(args.results_dir)
    fixer.run_complete_fix()

if __name__ == "__main__":
    main()
