# FMAP Experiment Systems

This repository contains **TWO** main experiment systems for testing FMAP heuristics:

## 1. Command Line Experiment Runner 📊
**Location**: `experiments/experiment_runner.py`

### Features:
- **Comprehensive Testing**: Tests all 5 heuristics (DTG, DTG+Landmarks, Inc_DTG+Landmarks, Centroids, MCS) across ALL domains
- **Scalability Analysis**: Automatically tests problems with varying agent counts (2-12 agents) and complexities
- **Smart Sampling**: Stratified sampling across domains for efficient testing
- **Rich Metrics**: Collects performance, plan quality, heuristic quality, and coordination metrics
- **Advanced Visualizations**: Generates comprehensive graphs and analysis
- **Statistical Analysis**: Provides correlation matrices, scaling analysis, and statistical summaries

### Usage:
```bash
cd experiments
source ../gui_automation_env/bin/activate

# Run sampled experiments (recommended for testing)
python experiment_runner.py

# Run ALL experiments (1000+ configurations)
python experiment_runner.py --full

# Analyze existing results only
python experiment_runner.py --analyze-only
```

### Generated Outputs:
- `results/all_results.json` - Complete experiment data
- `results/plots/` - Performance visualizations by domain and agent count
- `results/statistical_summary.txt` - Comprehensive statistical analysis

## 2. GUI Automation System 🖥️
**Location**: `fmap_gui_automation.py`

### Features:
- **GUI Integration**: Automates FMAP's graphical interface using PyAutoGUI
- **Real-time Monitoring**: Captures screenshots and GUI interactions
- **Quick Testing**: Focused on rapid validation of specific scenarios
- **Visual Feedback**: Provides visual confirmation of experiment execution

### Usage:
```bash
source gui_automation_env/bin/activate

# Run GUI automation
python fmap_gui_automation.py

# Quick GUI test
python quick_gui_test.py
```

### Generated Outputs:
- `gui_automation_results/` - Screenshots, logs, and analysis
- GUI interaction recordings and visual confirmations

## Key Differences

| Feature | Command Line Runner | GUI Automation |
|---------|-------------------|----------------|
| **Scope** | Comprehensive (1000+ configs) | Targeted testing |
| **Performance** | Fast, efficient | Slower (GUI dependent) |
| **Data Collection** | Rich metrics & analysis | Basic validation |
| **Visualization** | Advanced graphs & statistics | Screenshots & logs |
| **Use Case** | Research & analysis | GUI validation & demos |

## Recommended Workflow

1. **Development/Testing**: Use Command Line Runner for comprehensive analysis
2. **Validation**: Use GUI Automation for visual confirmation of specific scenarios
3. **Research**: Command Line Runner provides all necessary data and visualizations
4. **Demos**: GUI Automation provides visual proof of operation

## Domain Coverage

Both systems test across multiple planning domains:
- **driverlog** - Transportation logistics
- **logistics** - Package delivery coordination  
- **rovers** - Mars rover mission planning
- **satellite** - Satellite observation scheduling
- **elevators** - Multi-elevator coordination
- **ma-blocksworld** - Multi-agent blocks world
- **openstacks** - Manufacturing optimization
- **woodworking** - Production planning
- **zenotravel** - Travel planning
- **depots** - Warehouse management

## Agent Scaling

Problems are automatically categorized by:
- **Agent Count**: 2-12 agents per problem
- **Complexity**: Small/Medium/Large based on problem size and agent count
- **Domain Characteristics**: Each domain has unique multi-agent coordination challenges

## Comprehensive Heuristic Analysis

Both systems provide balanced analysis of **ALL heuristics**:
- Performance comparison across all domains
- Scaling analysis with agent count for each heuristic
- Statistical ranking and significance testing
- Relative performance analysis (normalized comparisons)
- Best performing heuristic identification per domain 