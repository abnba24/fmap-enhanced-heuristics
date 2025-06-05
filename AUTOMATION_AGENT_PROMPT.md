# 🤖 FMAP GUI Automation Agent Prompt

## 🎯 MISSION OBJECTIVE

You are tasked with automating comprehensive FMAP GUI experiments to compare heuristic performance across multiple domains. Use pyautogui to control the FMAP GUI interface, extract performance metrics from trace windows, and generate detailed comparative analysis with visualizations.

## 🔧 TECHNICAL SETUP

### **Required Dependencies**
Install these Python packages:
```bash
pip install pyautogui pandas matplotlib seaborn numpy pillow opencv-python
```

### **Working Environment**
- **Base Directory**: `/Users/abrahamadeniyi/Desktop/altorler-fmap-2ce663469695`
- **FMAP JAR**: `FMAP.jar` (in base directory)
- **Domains**: `Domains/` subdirectory
- **Screen Resolution**: Detect automatically with pyautogui
- **GUI Interface**: FMAP Java Swing application

## 📊 EXPERIMENTAL MATRIX

### **Test Suite (Priority Order)**

| Domain | Problem | Agents | Complexity | Command Template |
|--------|---------|--------|------------|------------------|
| driverlog | Pfile1 | 2 | EASY | `java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h {H} -gui` |
| driverlog | Pfile2 | 2 | MEDIUM | `java -jar FMAP.jar driver1 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile2/agent-list.txt -h {H} -gui` |
| driverlog | Pfile5 | 3 | HARD | `java -jar FMAP.jar driver1 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver2.pddl driver3 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver3.pddl Domains/driverlog/Pfile5/agent-list.txt -h {H} -gui` |
| ma-blocksworld | Pfile6-2 | 4 | MEDIUM | `java -jar FMAP.jar r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h {H} -gui` |
| elevators | Pfile1 | 3 | EASY | `java -jar FMAP.jar fast0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsfast0.pddl slow0-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow0-0.pddl slow1-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow1-0.pddl Domains/elevators/Pfile1/agent-list.txt -h {H} -gui` |

### **Heuristics to Test**
Replace `{H}` with each value:
- **0**: FF (Fast-Forward)
- **1**: DTG (Domain Transition Graph)
- **2**: DTG + Landmarks (Default)
- **3**: Inc. DTG + Landmarks
- **4**: Centroids (FIXED - Special Focus)
- **5**: MCS (Min. Covering States)

## 🎮 GUI AUTOMATION PROTOCOL

### **Step 1: Environment Setup**
```python
import pyautogui
import subprocess
import time
import pandas as pd
import re
from datetime import datetime

# Configure pyautogui
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1.0  # 1 second between actions

# Set up screen detection
screen_width, screen_height = pyautogui.size()
print(f"Screen resolution: {screen_width}x{screen_height}")
```

### **Step 2: Experiment Automation Flow**

For each domain/problem/heuristic combination:

1. **Launch FMAP GUI**
   ```python
   # Start FMAP with specific command
   process = subprocess.Popen(command, shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
   time.sleep(3)  # Wait for GUI to appear
   ```

2. **GUI Window Detection**
   ```python
   # Find FMAP window (look for distinctive elements)
   # Possible window titles: "FMAP", "Multi-Agent Planning", etc.
   fmap_window = pyautogui.getWindowsWithTitle("FMAP")
   if fmap_window:
       fmap_window[0].activate()
   ```

3. **Monitor Planning Process**
   ```python
   # Wait for planning to complete (look for specific text patterns)
   # Monitor for: "Planning completed", "Solution found", "No solution"
   # Timeout after 300 seconds (5 minutes)
   ```

4. **Extract Trace Data**
   ```python
   # Locate trace window/text area
   # Scroll to bottom to find statistics
   # Look for patterns like:
   # "Planning (expansion) time: X.XXX sec."
   # "Evaluation time: X.XXX sec."
   # "Communication time: X.XXX sec."
   # "Average branching factor: X.XXX"
   # "Used memory: XXXkb"
   # "Plan length: XX"
   # "Number of messages: XXXX"
   # "Total time: X.XXX sec."
   ```

5. **Data Extraction Methods**
   ```python
   def extract_metrics_from_trace(trace_text):
       metrics = {}
       
       # Regex patterns for key metrics
       patterns = {
           'planning_time': r'Planning \(expansion\) time: ([\d.]+) sec\.',
           'evaluation_time': r'Evaluation time: ([\d.]+) sec\.',
           'communication_time': r'Communication time: ([\d.]+) sec\.',
           'branching_factor': r'Average branching factor: ([\d.]+)',
           'memory_usage': r'Used memory: (\d+)kb',
           'plan_length': r'Plan length: (\d+)',
           'messages': r'Number of messages: (\d+)',
           'total_time': r'Total time: ([\d.]+) sec\.',
           'discarded_plans': r'Discarded plans: (\d+)'
       }
       
       for key, pattern in patterns.items():
           match = re.search(pattern, trace_text)
           metrics[key] = float(match.group(1)) if match else None
           
       return metrics
   ```

6. **Screenshot Capture**
   ```python
   # Capture final GUI state for verification
   screenshot = pyautogui.screenshot()
   screenshot.save(f"results/experiment_{domain}_{problem}_{heuristic}_{timestamp}.png")
   ```

7. **Process Cleanup**
   ```python
   # Close FMAP GUI
   pyautogui.hotkey('alt', 'f4')  # or appropriate close method
   time.sleep(2)
   
   # Kill any remaining processes
   subprocess.run(["pkill", "-f", "FMAP.jar"], capture_output=True)
   ```

## 📊 DATA COLLECTION STRUCTURE

### **Output Data Format**
Create a comprehensive DataFrame with these columns:
```python
columns = [
    'domain', 'problem', 'agent_count', 'complexity',
    'heuristic_id', 'heuristic_name', 'timestamp',
    'success', 'timeout', 'error_message',
    'planning_time_sec', 'evaluation_time_sec', 'communication_time_sec',
    'total_time_sec', 'memory_mb', 'plan_length', 'messages',
    'branching_factor', 'discarded_plans',
    'search_efficiency', 'memory_per_agent', 'messages_per_second'
]
```

### **Derived Metrics**
Calculate additional metrics:
```python
def calculate_derived_metrics(row):
    row['search_efficiency'] = row['plan_length'] / row['total_time_sec'] if row['total_time_sec'] > 0 else 0
    row['memory_per_agent'] = row['memory_mb'] / row['agent_count']
    row['messages_per_second'] = row['messages'] / row['total_time_sec'] if row['total_time_sec'] > 0 else 0
    return row
```

## 📈 ANALYSIS AND VISUALIZATION REQUIREMENTS

### **1. Performance Comparison Tables**

Create tables showing:
```python
# Summary table by heuristic
summary_table = df.groupby('heuristic_name').agg({
    'success': 'mean',
    'total_time_sec': 'mean',
    'memory_mb': 'mean',
    'plan_length': 'mean',
    'messages': 'mean'
}).round(3)

# Domain-specific performance
domain_performance = df.groupby(['domain', 'heuristic_name']).agg({
    'success': 'mean',
    'total_time_sec': 'mean'
}).round(3)
```

### **2. Visualization Requirements**

Generate these plots:

**A. Performance Heatmap**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Success rate heatmap
success_matrix = df.pivot_table(values='success', 
                               index='heuristic_name', 
                               columns='domain', 
                               aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(success_matrix, annot=True, cmap='RdYlGn', 
            fmt='.2f', cbar_kws={'label': 'Success Rate'})
plt.title('Heuristic Success Rate by Domain')
plt.tight_layout()
plt.savefig('results/success_rate_heatmap.png', dpi=300)
```

**B. Performance Comparison Charts**
```python
# Execution time comparison
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[df['success']==True], 
            x='heuristic_name', y='total_time_sec')
plt.yscale('log')
plt.title('Execution Time Distribution by Heuristic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/execution_time_comparison.png', dpi=300)

# Memory usage comparison
plt.figure(figsize=(12, 8))
sns.barplot(data=df[df['success']==True], 
            x='heuristic_name', y='memory_mb', ci=95)
plt.title('Memory Usage by Heuristic')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/memory_usage_comparison.png', dpi=300)
```

**C. Scalability Analysis**
```python
# Agent count vs performance
plt.figure(figsize=(10, 6))
for heuristic in df['heuristic_name'].unique():
    heuristic_data = df[df['heuristic_name'] == heuristic]
    plt.plot(heuristic_data['agent_count'], 
             heuristic_data['total_time_sec'], 
             marker='o', label=heuristic)
plt.xlabel('Number of Agents')
plt.ylabel('Total Time (sec)')
plt.title('Scalability Analysis: Agents vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/scalability_analysis.png', dpi=300)
```

**D. Centroids Focus Analysis**
```python
# Special focus on Centroids (h=4) performance
centroids_data = df[df['heuristic_id'] == 4]
other_heuristics = df[df['heuristic_id'].isin([1, 2, 5])]

plt.figure(figsize=(12, 8))
# Create comparison plots specifically highlighting Centroids
# Show before/after fix validation
```

### **3. Statistical Analysis**

Perform statistical tests:
```python
from scipy import stats

# Compare Centroids vs other heuristics
centroids_times = df[df['heuristic_id']==4]['total_time_sec'].dropna()
dtg_times = df[df['heuristic_id']==1]['total_time_sec'].dropna()

# T-test for significant differences
t_stat, p_value = stats.ttest_ind(centroids_times, dtg_times)
print(f"Centroids vs DTG t-test: t={t_stat:.3f}, p={p_value:.3f}")
```

### **4. Report Generation**

Generate automated report:
```python
def generate_experiment_report(df):
    report = f"""
# FMAP Heuristic Performance Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total experiments: {len(df)}
- Success rate: {df['success'].mean():.1%}
- Domains tested: {df['domain'].nunique()}
- Heuristics compared: {df['heuristic_name'].nunique()}

## Key Findings
### Best Overall Heuristic: {df.groupby('heuristic_name')['success'].mean().idxmax()}
### Fastest Heuristic: {df[df['success']==True].groupby('heuristic_name')['total_time_sec'].mean().idxmin()}
### Most Memory Efficient: {df[df['success']==True].groupby('heuristic_name')['memory_mb'].mean().idxmin()}

## Centroids Heuristic Analysis
- Success rate: {df[df['heuristic_id']==4]['success'].mean():.1%}
- Average time: {df[(df['heuristic_id']==4) & (df['success']==True)]['total_time_sec'].mean():.3f} sec
- Status: {'WORKING CORRECTLY' if df[df['heuristic_id']==4]['success'].sum() > 0 else 'ISSUES DETECTED'}
"""
    return report
```

## 🎯 DELIVERABLES REQUIRED

### **Files to Generate:**
1. **`experiment_results.csv`** - Raw data from all experiments
2. **`performance_summary.csv`** - Aggregated statistics
3. **`experiment_report.md`** - Automated analysis report
4. **`results/`** directory with:
   - Success rate heatmaps
   - Performance comparison charts
   - Memory usage analysis
   - Scalability plots
   - Screenshots from key experiments

### **Key Analysis Questions to Answer:**
1. **Is the fixed Centroids heuristic working correctly?**
2. **Which heuristic performs best overall?**
3. **How do heuristics scale with agent count?**
4. **Are there domain-specific heuristic advantages?**
5. **What are the trade-offs between speed and solution quality?**

## ⚠️ ERROR HANDLING

Implement robust error handling:
```python
def handle_experiment_error(domain, problem, heuristic, error):
    error_entry = {
        'domain': domain,
        'problem': problem, 
        'heuristic_id': heuristic,
        'success': False,
        'error_message': str(error),
        'timestamp': datetime.now()
    }
    return error_entry
```

## 🚀 EXECUTION CHECKLIST

- [ ] Install required Python packages
- [ ] Test pyautogui screen detection
- [ ] Verify FMAP.jar launches correctly
- [ ] Test GUI window detection
- [ ] Validate metric extraction patterns
- [ ] Run pilot experiment on one domain/heuristic
- [ ] Execute full experimental matrix
- [ ] Generate all required visualizations
- [ ] Compile comprehensive analysis report

**SUCCESS METRIC**: Complete analysis comparing all 6 heuristics across multiple domains with detailed performance metrics, statistical validation of the Centroids fix, and actionable insights for multi-agent planning optimization.

Execute this automation systematically and provide comprehensive results that validate the heuristic implementations and guide future multi-agent planning research! 