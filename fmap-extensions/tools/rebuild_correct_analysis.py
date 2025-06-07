#!/usr/bin/env python3
"""
Rebuild Analysis with Correct Heuristic Names

This script completely rebuilds the analysis from the raw JSON data
using the correct heuristic mapping to ensure accuracy.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class CorrectAnalysisBuilder:
    """Rebuild analysis with correct heuristic names from raw data"""
    
    def __init__(self, results_dir="analysis_outputs/results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Correct heuristic mapping (from Java source)
        self.correct_heuristic_names = {
            1: "DTG",
            2: "DTG+Landmarks", 
            3: "Inc_DTG+Landmarks",
            4: "Centroids",
            5: "MCS"
        }
        
    def load_and_process_data(self):
        """Load raw JSON data and process with correct heuristic mapping"""
        print("Loading and processing raw experiment data...")

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
                    'filename': file_path.stem,
                    'domain': config.get('domain'),
                    'problem': config.get('problem'),
                    'heuristic_id': heuristic_id,
                    'heuristic_name': heuristic_name,
                    'agent_count': len(config.get('agents', [])),
                    'agents': config.get('agents', []),

                    # Search performance
                    'coverage': search.get('coverage', False),
                    'success': search.get('coverage', False),
                    'wall_clock_time': search.get('wall_clock_time'),
                    'cpu_time': search.get('cpu_time'),
                    'peak_memory_mb': search.get('peak_memory_mb'),
                    'search_nodes': search.get('search_nodes'),

                    # Plan quality
                    'plan_found': plan.get('plan_found', False),
                    'plan_length': plan.get('plan_length'),
                    'makespan': plan.get('makespan'),

                    # Error information
                    'error_message': data.get('error_message'),
                    'has_error': data.get('error_message') is not None
                }

                results.append(result)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        df = pd.DataFrame(results)
        print(f"Loaded {len(df)} experiments with correct heuristic names")
        
        # Add derived columns
        df['complexity'] = df['agent_count'].apply(self._categorize_complexity)
        df['success_rate'] = df['success'].astype(int)
        
        return df
    
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
    
    def generate_corrected_csv_files(self, df):
        """Generate corrected CSV files with proper heuristic names"""
        print("Generating corrected CSV files...")

        # 1. Heuristic Summary
        self._generate_heuristic_summary(df)

        # 2. Comprehensive Metrics Comparison
        self._generate_comprehensive_metrics(df)

        # 3. Domain Analysis
        self._generate_domain_analysis(df)

        # 4. Performance Summary Table
        self._generate_performance_summary(df)

        # 5. Timing Analysis Table
        self._generate_timing_analysis(df)

        # 6. Efficiency Analysis Table
        self._generate_efficiency_analysis(df)

        # 7. Quality Analysis Table
        self._generate_quality_analysis(df)

        print("All corrected CSV files generated")
    
    def _generate_heuristic_summary(self, df):
        """Generate heuristic summary table"""
        summary_stats = []
        
        for heuristic in df['heuristic_name'].unique():
            heur_data = df[df['heuristic_name'] == heuristic]
            successful = heur_data[heur_data['success'] == True]
            
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
        summary_df.to_csv(self.plots_dir / 'heuristic_summary_corrected.csv', index=False)
        print("  ✅ Generated heuristic_summary_corrected.csv")
    
    def _generate_comprehensive_metrics(self, df):
        """Generate comprehensive metrics comparison"""
        metrics_comparison = []
        
        for heuristic in df['heuristic_name'].unique():
            all_heur_data = df[df['heuristic_name'] == heuristic]
            successful_heur_data = df[(df['heuristic_name'] == heuristic) & (df['success'] == True)]
            
            if len(all_heur_data) > 0:
                stats = {
                    'Heuristic': heuristic,
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
        
        comparison_df = pd.DataFrame(metrics_comparison)
        comparison_df = comparison_df.sort_values('Success_Rate_%', ascending=False)
        comparison_df.to_csv(self.plots_dir / 'comprehensive_metrics_comparison_corrected.csv', index=False)
        print("  ✅ Generated comprehensive_metrics_comparison_corrected.csv")
    
    def _generate_domain_analysis(self, df):
        """Generate domain analysis table"""
        domain_analysis = []
        for domain in df['domain'].unique():
            domain_data = df[df['domain'] == domain]
            
            for heuristic in domain_data['heuristic_name'].unique():
                heur_domain_data = domain_data[domain_data['heuristic_name'] == heuristic]
                successful = heur_domain_data[heur_domain_data['success'] == True]
                
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
        domain_df.to_csv(self.plots_dir / 'domain_analysis_corrected.csv', index=False)
        print("  ✅ Generated domain_analysis_corrected.csv")

    def _generate_performance_summary(self, df):
        """Generate performance summary table"""
        summary_cols = ['Heuristic', 'Total_Experiments', 'Successful_Runs', 'Success_Rate_%',
                       'Wall_Time_Mean_s', 'Memory_Mean_MB', 'Search_Nodes_Mean', 'Plan_Length_Mean']

        # Use comprehensive metrics as base
        comprehensive_df = pd.read_csv(self.plots_dir / 'comprehensive_metrics_comparison_corrected.csv')
        summary_table = comprehensive_df[summary_cols].copy()
        summary_table.to_csv(self.plots_dir / 'performance_summary_table_corrected.csv', index=False)
        print("  ✅ Generated performance_summary_table_corrected.csv")

    def _generate_timing_analysis(self, df):
        """Generate timing analysis table"""
        timing_cols = ['Heuristic', 'Wall_Time_Mean_s', 'Wall_Time_Median_s', 'Wall_Time_Std_s',
                       'Wall_Time_Min_s', 'Wall_Time_Max_s', 'CPU_Time_Mean_s', 'CPU_Time_Median_s']

        comprehensive_df = pd.read_csv(self.plots_dir / 'comprehensive_metrics_comparison_corrected.csv')
        timing_table = comprehensive_df[timing_cols].copy()
        timing_table.to_csv(self.plots_dir / 'timing_analysis_table_corrected.csv', index=False)
        print("  ✅ Generated timing_analysis_table_corrected.csv")

    def _generate_efficiency_analysis(self, df):
        """Generate efficiency analysis table"""
        efficiency_cols = ['Heuristic', 'Memory_Mean_MB', 'Memory_Median_MB', 'Memory_Std_MB',
                          'Search_Nodes_Mean', 'Search_Nodes_Median', 'Search_Nodes_Std']

        comprehensive_df = pd.read_csv(self.plots_dir / 'comprehensive_metrics_comparison_corrected.csv')
        efficiency_table = comprehensive_df[efficiency_cols].copy()
        efficiency_table.to_csv(self.plots_dir / 'efficiency_analysis_table_corrected.csv', index=False)
        print("  ✅ Generated efficiency_analysis_table_corrected.csv")

    def _generate_quality_analysis(self, df):
        """Generate quality analysis table"""
        quality_cols = ['Heuristic', 'Success_Rate_%', 'Plan_Length_Mean', 'Plan_Length_Median',
                       'Plan_Length_Std', 'Makespan_Mean', 'Makespan_Median']

        comprehensive_df = pd.read_csv(self.plots_dir / 'comprehensive_metrics_comparison_corrected.csv')
        quality_table = comprehensive_df[quality_cols].copy()
        quality_table.to_csv(self.plots_dir / 'quality_analysis_table_corrected.csv', index=False)
        print("  ✅ Generated quality_analysis_table_corrected.csv")

    def create_beautiful_tables(self):
        """Create beautiful HTML and Markdown tables from corrected CSV files"""
        print("🎨 Creating beautiful formatted tables...")

        corrected_csv_files = list(self.plots_dir.glob("*_corrected.csv"))

        for csv_file in corrected_csv_files:
            try:
                df = pd.read_csv(csv_file)
                base_name = csv_file.stem.replace('_corrected', '')

                # Create HTML table
                html_file = self.plots_dir / f"{base_name}_beautiful.html"
                self._create_beautiful_html_table(df, html_file, base_name)

                # Create Markdown table
                md_file = self.plots_dir / f"{base_name}_beautiful.md"
                self._create_beautiful_markdown_table(df, md_file, base_name)

                print(f"  ✅ Created beautiful tables for {base_name}")

            except Exception as e:
                print(f"  ❌ Error creating tables for {csv_file.name}: {e}")

    def _create_beautiful_html_table(self, df, output_file, title):
        """Create a beautifully formatted HTML table"""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title.replace('_', ' ').title()} - FMAP Analysis</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Roboto', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .table-container {{
            padding: 30px;
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0;
            font-size: 14px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-radius: 10px;
            overflow: hidden;
        }}

        th {{
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            font-weight: 600;
            padding: 15px 12px;
            text-align: left;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
            transition: background-color 0.3s ease;
        }}

        tr:hover {{
            background-color: #f8f9fa;
        }}

        tr:nth-child(even) {{
            background-color: #fdfdfd;
        }}

        .numeric {{
            text-align: right;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-weight: 500;
        }}

        .heuristic-name {{
            font-weight: 700;
            color: #2c3e50;
            background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .success-rate {{
            font-weight: 600;
        }}

        .success-high {{ color: #27ae60; }}
        .success-medium {{ color: #f39c12; }}
        .success-low {{ color: #e74c3c; }}

        .footer {{
            background: #ecf0f1;
            padding: 20px;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }}

        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}

        .stat-item {{
            text-align: center;
            padding: 10px;
        }}

        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}

        .stat-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title.replace('_', ' ').title()}</h1>
            <p>FMAP Enhanced Heuristics Performance Analysis</p>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number">{len(df)}</div>
                    <div class="stat-label">Heuristics</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(df.columns)}</div>
                    <div class="stat-label">Metrics</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{pd.Timestamp.now().strftime('%Y')}</div>
                    <div class="stat-label">Analysis Year</div>
                </div>
            </div>
        </div>
        <div class="table-container">
            {self._df_to_beautiful_html(df)}
        </div>
        <div class="footer">
            <strong>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}</strong><br>
            FMAP Enhanced Heuristics Analysis Framework | Research-Grade Performance Evaluation
        </div>
    </div>
</body>
</html>
"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

    def _df_to_beautiful_html(self, df):
        """Convert DataFrame to beautifully formatted HTML"""
        html = "<table>\n<thead>\n<tr>\n"

        # Headers
        for col in df.columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"

        # Rows
        for _, row in df.iterrows():
            html += "<tr>\n"
            for col, value in row.items():
                css_class = ""
                formatted_value = value

                if col == 'Heuristic':
                    css_class = 'class="heuristic-name"'
                elif pd.api.types.is_numeric_dtype(df[col]) and pd.notna(value):
                    css_class = 'class="numeric"'
                    if isinstance(value, float):
                        if 'rate' in col.lower() or 'percent' in col.lower():
                            if value <= 1:
                                formatted_value = f"{value:.1%}"
                                if value >= 0.8:
                                    css_class = 'class="numeric success-high"'
                                elif value >= 0.5:
                                    css_class = 'class="numeric success-medium"'
                                else:
                                    css_class = 'class="numeric success-low"'
                            else:
                                formatted_value = f"{value:.1f}%"
                        elif 'time' in col.lower():
                            formatted_value = f"{value:.2f}s"
                        elif 'memory' in col.lower() or 'mb' in col.lower():
                            formatted_value = f"{value:.1f} MB"
                        elif 'nodes' in col.lower():
                            formatted_value = f"{value:,.0f}"
                        else:
                            formatted_value = f"{value:.2f}"

                if pd.isna(value):
                    formatted_value = "N/A"
                    css_class = 'class="numeric" style="color: #bdc3c7;"'

                html += f"<td {css_class}>{formatted_value}</td>\n"
            html += "</tr>\n"

        html += "</tbody>\n</table>"
        return html

    def _create_beautiful_markdown_table(self, df, output_file, title):
        """Create a beautifully formatted Markdown table"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 📊 {title.replace('_', ' ').title()}\n\n")
            f.write("*FMAP Enhanced Heuristics Performance Analysis*\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d at %H:%M:%S')}  \n")
            f.write(f"**Heuristics Analyzed:** {len(df)}  \n")
            f.write(f"**Metrics Evaluated:** {len(df.columns)}  \n\n")

            f.write("---\n\n")

            # Create markdown table
            headers = [col.replace('_', ' ').title() for col in df.columns]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["-" * (len(header) + 2) for header in headers]) + "|\n")

            for _, row in df.iterrows():
                formatted_row = []
                for col, value in row.items():
                    if pd.isna(value):
                        formatted_row.append(" *N/A* ")
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        if isinstance(value, float):
                            if 'rate' in col.lower() or 'percent' in col.lower():
                                if value <= 1:
                                    if value >= 0.8:
                                        formatted_row.append(f" **{value:.1%}** 🟢 ")
                                    elif value >= 0.5:
                                        formatted_row.append(f" **{value:.1%}** 🟡 ")
                                    else:
                                        formatted_row.append(f" **{value:.1%}** 🔴 ")
                                else:
                                    formatted_row.append(f" **{value:.1f}%** ")
                            elif 'time' in col.lower():
                                formatted_row.append(f" `{value:.2f}s` ")
                            elif 'memory' in col.lower() or 'mb' in col.lower():
                                formatted_row.append(f" `{value:.1f} MB` ")
                            elif 'nodes' in col.lower():
                                formatted_row.append(f" `{value:,.0f}` ")
                            else:
                                formatted_row.append(f" `{value:.2f}` ")
                        else:
                            formatted_row.append(f" `{value}` ")
                    else:
                        if col == 'Heuristic':
                            formatted_row.append(f" **{value}** ")
                        else:
                            formatted_row.append(f" {value} ")

                f.write("|" + "|".join(formatted_row) + "|\n")

            f.write(f"\n---\n\n")
            f.write("## 📈 Key Insights\n\n")

            # Add some automatic insights based on the data
            if 'Success_Rate_%' in df.columns:
                best_heuristic = df.loc[df['Success_Rate_%'].idxmax(), 'Heuristic']
                best_rate = df['Success_Rate_%'].max()
                f.write(f"- **Best Success Rate:** {best_heuristic} ({best_rate:.1f}%)\n")

            if 'Wall_Time_Mean_s' in df.columns:
                fastest_heuristic = df.loc[df['Wall_Time_Mean_s'].idxmin(), 'Heuristic']
                fastest_time = df['Wall_Time_Mean_s'].min()
                f.write(f"- **Fastest Execution:** {fastest_heuristic} ({fastest_time:.2f}s average)\n")

            if 'Memory_Mean_MB' in df.columns:
                most_efficient = df.loc[df['Memory_Mean_MB'].idxmin(), 'Heuristic']
                min_memory = df['Memory_Mean_MB'].min()
                f.write(f"- **Most Memory Efficient:** {most_efficient} ({min_memory:.1f} MB average)\n")

            f.write(f"\n*Analysis completed with FMAP Enhanced Heuristics Framework*\n")

    def run_complete_rebuild(self):
        """Run the complete rebuild process"""
        print("Starting complete analysis rebuild with correct heuristic names...")
        print("=" * 70)

        # Step 1: Load and process raw data
        df = self.load_and_process_data()

        # Step 2: Generate corrected CSV files
        self.generate_corrected_csv_files(df)

        # Step 3: Create beautiful tables
        self.create_beautiful_tables()

        print("\n" + "=" * 70)
        print("COMPLETE! Analysis rebuilt successfully with correct heuristic names:")
        print(f"  Generated corrected CSV files")
        print(f"  Created beautiful HTML and Markdown tables")
        print(f"  All files saved to: {self.plots_dir}")

        # Show summary of what was found
        print(f"\nAnalysis Summary:")
        print(f"  Total Experiments: {len(df)}")
        print(f"  Successful Experiments: {len(df[df['success'] == True])}")
        print(f"  Overall Success Rate: {df['success'].mean():.1%}")
        print(f"  Heuristics: {', '.join(sorted(df['heuristic_name'].unique()))}")
        print(f"  Domains: {', '.join(sorted(df['domain'].unique()))}")

        # List generated files
        print(f"\nGenerated Files:")
        for file_type, pattern in [("Corrected CSV", "*_corrected.csv"), ("Beautiful HTML", "*_beautiful.html"), ("Beautiful Markdown", "*_beautiful.md")]:
            files = list(self.plots_dir.glob(pattern))
            if files:
                print(f"  {file_type}: {len(files)} files")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Rebuild analysis with correct heuristic names')
    parser.add_argument('--results-dir', default='analysis_outputs/results',
                       help='Directory containing results')

    args = parser.parse_args()

    builder = CorrectAnalysisBuilder(args.results_dir)
    builder.run_complete_rebuild()

if __name__ == "__main__":
    main()
