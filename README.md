# FMAP Heuristic Comparison Analysis

A comprehensive experimental evaluation and analysis of distributed multi-agent planning heuristics using the FMAP platform.

![Heuristic Performance](fmap-extensions/experiments/results/plots/heuristic_comparison_analysis.png)

## Project Overview

This repository contains an extended version of the FMAP (Factored Multi-Agent Planning) platform with comprehensive experimental analysis comparing different heuristic functions for distributed multi-agent planning. The project evaluates heuristic performance across multiple domains, agent configurations, and problem complexities.

### Research Focus

- **Heuristic Comparison**: Systematic evaluation of 5 different heuristics
- **Performance Analysis**: Execution time, memory usage, plan quality, and success rates
- **Scalability Study**: Performance across different agent counts and problem complexities
- **Domain Analysis**: Comparative evaluation across multiple planning domains

## Key Findings

### Heuristic Performance Ranking

| Rank | Heuristic | Success Rate | Avg Time | Memory Usage | Plan Quality |
|------|-----------|--------------|----------|--------------|--------------|
| 1 | **DTG+Landmarks** | 100.0% | 4.89s | 144.7 MB | 14.1 actions |
| 2 | **Inc_DTG+Landmarks** | 100.0% | 4.93s | 144.4 MB | 14.1 actions |
| 3 | **DTG** | 100.0% | 11.88s | 166.3 MB | 11.8 actions |
| 4 | **MCS** | 61.5% | 382.6s | 597.4 MB | 7.5 actions |
| 5 | **Centroids** | 46.2% | 55.9s | 197.0 MB | 6.7 actions |

### Domain-Specific Performance

- **Most Successful**: Driverlog, Elevators, Zenotravel (100% success for top heuristics)
- **Most Challenging**: Openstacks (0% success across all heuristics)
- **Complex Domains**: Depots (8 agents), Elevators (variable complexity)

### Key Insights

1. **DTG+Landmarks** provides the optimal balance of speed, memory efficiency, and success rate
2. **Incremental DTG with landmarks** offers competitive performance with minimal overhead
3. **DTG-based heuristics** consistently achieve 100% success rates across tested domains
4. **MCS and Centroids** struggle with complex multi-agent coordination, showing lower success rates

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install pandas matplotlib seaborn numpy scipy psutil
```

### Running Analysis

```bash
# Navigate to experiments directory
cd fmap-extensions/experiments

# Run comprehensive heuristic comparison analysis
python3 experiment_runner.py --analyze-only

# Generate custom analysis with detailed tables
python3 heuristic_comparison_analysis.py
```

### Sample FMAP Execution

```bash
# Run FMAP with DTG+Landmarks heuristic
java -jar FMAP.jar -h 2 agent1 domain1.pddl problem1.pddl agent-list.txt
```

## Project Structure

```
fmap-extensions/
├── experiments/
│   ├── experiment_runner.py      # Main experiment framework
│   ├── heuristic_comparison_analysis.py  # Custom analysis tools
│   ├── results/
│   │   ├── plots/                # Generated visualizations
│   │   ├── *.json               # Individual experiment results
│   │   └── statistical_summary.txt
│   └── analysis_env/            # Python virtual environment
├── Domains/                     # Planning domains and problems
│   ├── driverlog/
│   ├── elevators/
│   ├── zenotravel/
│   ├── openstacks/
│   └── depots/
├── fmap-original/              # Original FMAP source code
├── javadoc/                    # API documentation
└── FMAP.jar                   # Executable FMAP platform
```

## Experimental Analysis

### Available Heuristics

| ID | Heuristic | Description |
|----|-----------|-------------|
| 1  | **DTG** | Domain Transition Graph - Basic DTG-based heuristic for multi-agent planning |
| 2  | **DTG+Landmarks** | DTG combined with landmark detection for enhanced guidance |
| 3  | **Inc_DTG+Landmarks** | Incremental DTG with landmark detection for efficiency |
| 4  | **Centroids** | Centroid-based heuristic for distributed coordination |
| 5  | **MCS** | Minimum Covering States - State coverage heuristic for comprehensive exploration |

> **Note**: Heuristic IDs correspond to the Java implementation in `HeuristicFactory.java`. The mapping is consistent across all analysis scripts.

### Evaluation Metrics

- **Search Performance**: Execution time, memory usage, node expansions
- **Plan Quality**: Plan length, makespan, action coordination
- **Success Rate**: Coverage across different problem instances
- **Scalability**: Performance vs. agent count and problem complexity

### Experimental Domains

- **Driverlog**: Transportation and logistics (2-3 agents)
- **Elevators**: Building elevator control (3 agents)
- **Zenotravel**: Travel planning (2-3 agents)
- **Openstacks**: Manufacturing processes (2 agents)
- **Depots**: Warehouse operations (8 agents)

## Generated Visualizations

### Available Plots

1. **`heuristic_comparison_analysis.png`**: Overall performance comparison
2. **`heuristic_performance_by_domain.png`**: Domain-specific heatmaps
3. **`scalability_analysis.png`**: Performance vs. complexity
4. **`performance_matrices.png`**: Correlation analysis
5. **`heuristic_comparison.png`**: 9-panel comprehensive analysis

### Data Tables

1. **`heuristic_summary.csv`**: Overall performance statistics
2. **`domain_analysis.csv`**: Domain-specific breakdowns
3. **`detailed_analysis_report.md`**: Comprehensive analysis report
4. **`statistical_summary.txt`**: Statistical analysis summary

## Tools and Scripts

### Analysis Tools

- **`experiment_runner.py`**: Main experimental framework with analysis capabilities
- **`heuristic_comparison_analysis.py`**: Custom visualization and analysis tools
- **`generate_detailed_summary.py`**: Report generation utilities
- **`create_all_results.py`**: Result aggregation utilities

### Usage Examples

```bash
# Run analysis on existing results
python3 experiment_runner.py --analyze-only

# Generate custom comparison plots
python3 heuristic_comparison_analysis.py --plots-dir results/plots

# Create detailed markdown report
python3 generate_detailed_summary.py
```

## Results Summary

### Statistical Overview

- **Total Experiments**: 65 across 5 domains
- **Successful Runs**: 53 (81.5% overall success rate)
- **Best Performing**: DTG+Landmarks (4.89s average, 100% success rate)
- **Most Efficient**: Inc_DTG+Landmarks (4.93s average, 100% success rate)
- **Most Challenging**: Centroids (46.2% success) and MCS (61.5% success)

### Performance Insights

1. **DTG+Landmarks** achieves optimal performance with fastest execution and 100% success
2. **Inc_DTG+Landmarks** provides competitive alternative with minimal performance difference
3. **DTG-based approaches** demonstrate superior reliability across all tested domains
4. **MCS** shows better success rate (61.5%) than Centroids (46.2%) but with high execution times
5. **Centroids** require significant optimization for practical multi-agent applications

## Installation & Setup

### System Requirements

- Java 8+ (for FMAP execution)
- Python 3.8+ (for analysis tools)
- 8GB+ RAM (recommended for large experiments)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd altorler-fmap
   ```

2. **Set up Python environment**
   ```bash
   cd fmap-extensions/experiments
   python3 -m venv analysis_env
   source analysis_env/bin/activate
   pip install -r ../../requirements.txt
   ```

3. **Verify FMAP installation**
   ```bash
   java -jar FMAP.jar --help
   ```

## Research Applications

This analysis framework is suitable for:

- **Heuristic Development**: Testing new multi-agent planning heuristics
- **Domain Analysis**: Understanding domain-specific performance characteristics
- **Scalability Studies**: Evaluating performance across agent counts
- **Comparative Research**: Systematic heuristic comparison methodology
- **Educational Use**: Teaching multi-agent planning concepts

## Documentation

### Core Documentation

- **Original FMAP Manual**: See `README_GITHUB.md` for original FMAP documentation
- **Javadoc**: Complete API documentation in `javadoc/` directory
- **Analysis Reports**: Generated reports in `fmap-extensions/experiments/results/plots/`

### Research Papers & References

- [FMAP Platform Documentation](https://altorler.bitbucket.io/fmap/)
- Multi-Agent Planning Competition (CoDMAP) benchmarks
- DTG and Landmark heuristic literature

## Contributing

Contributions welcome! Areas for enhancement:

- **New Heuristics**: Implement additional heuristic functions
- **Domains**: Add new planning domains for evaluation
- **Analysis Tools**: Enhance visualization and statistical analysis
- **Performance**: Optimize experimental framework efficiency

## License

This project extends the original FMAP platform. See `LICENSE.md` for details.

## Links

- **Original FMAP**: [Bitbucket Repository](https://bitbucket.org/altorler/fmap/)
- **CoDMAP Competition**: [Official Website](http://agents.fel.cvut.cz/codmap/)
- **Planning Benchmarks**: Domain collections and problem instances

---

**Experimental Analysis & Extensions** | **FMAP Platform** | **Multi-Agent Planning Research**