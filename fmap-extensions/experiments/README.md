# FMAP Heuristic Comparison Experiments

This experimental framework provides comprehensive comparison of heuristics in FMAP, including the newly implemented **Centroids** and **Minimum Covering States (MCS)** heuristics.

## Overview

The framework measures and compares heuristics across five key categories:

1. **Search Performance**: Coverage, time, memory, node expansions, branching factor
2. **Plan Quality**: Cost, makespan, concurrency index, goal-distance metrics  
3. **Heuristic Quality**: MAE, RMSE, correlation, computation time
4. **Coordination**: Messages, data volume, sync rounds, privacy scores
5. **Statistical Analysis**: Confidence intervals, significance tests

## Quick Start

### Prerequisites

1. **Python 3.8+** with required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **FMAP.jar** in the project root directory

3. **Domain files** in the `Domains/` directory

### Run Quick Test

```bash
cd experiments
python run_experiments.py --quick
```

This runs a limited test with 2 problems per domain to verify the setup.

### Run Full Experiments

```bash
python run_experiments.py
```

This runs the complete experimental suite comparing all 5 heuristics:
- DTG (baseline)
- DTG + Landmarks
- Incremental DTG + Landmarks  
- **Centroids** (new)
- **MCS** (new)

## Command Line Options

```bash
python run_experiments.py [OPTIONS]

Options:
  --quick                     Run quick test (2 problems per domain)
  --domains DOMAINS          Domains to test (default: driverlog logistics rovers satellite)
  --heuristics HEURISTICS    Heuristic IDs to test (default: 1 2 3 4 5)
  --max-problems N           Max problems per domain (default: 5)
  --timeout SECONDS          Timeout per experiment (default: 1800)
  --memory-limit MB          Memory limit (default: 8192)
  --fmap-jar PATH            Path to FMAP jar (default: FMAP.jar)
  --results-dir DIR          Results directory (default: results)
  --plots-dir DIR            Plots directory (default: plots)
  --skip-experiments         Only analyze existing results
  --skip-analysis            Only run experiments
  --skip-plots               Skip visualization generation
```

## Heuristic IDs

| ID | Heuristic | Description |
|----|-----------|-------------|
| 1  | DTG | Domain Transition Graph (baseline) |
| 2  | DTG+Landmarks | DTG with landmarks |
| 3  | Inc_DTG+Landmarks | Incremental DTG with landmarks |
| 4  | **Centroids** | **Minimizes mean cost to goals** |
| 5  | **MCS** | **Minimizes max cost to goals** |

## Output Structure

```
results/
├── all_results.json           # Complete experimental data
├── analysis_report.txt        # Statistical analysis summary
├── result_XXXX_*.json        # Individual experiment results
└── ...

plots/
├── coverage_comparison.png    # Coverage rates by heuristic
├── performance_comparison.png # Time, memory, search effort
├── time_quality_tradeoff.png  # Pareto frontier analysis
├── speedup_analysis_vs_DTG.png # Geometric mean speedups
├── scaling_analysis.png       # Performance vs number of agents
├── domain_specific_analysis.png # Per-domain breakdowns
└── statistical_significance_heatmap.png # Significance tests
```

## Understanding the Results

### Coverage Analysis
- **Success rate** by heuristic across all problems
- **Domain-specific** coverage to identify strengths/weaknesses

### Performance Metrics
- **Geometric mean speedup** vs baseline (DTG)
- **Statistical significance** using Wilcoxon signed-rank tests
- **Effect sizes** to measure practical importance

### Key Visualizations

1. **Coverage Comparison**: Which heuristics solve more problems?
2. **Performance Boxes**: Distribution of time, memory, search effort
3. **Time vs Quality Scatter**: Pareto trade-offs between speed and plan quality
4. **Speedup Analysis**: How much faster are new heuristics?
5. **Scaling Analysis**: How do heuristics perform with more agents?
6. **Statistical Heatmap**: Which differences are statistically significant?

## Experimental Protocol

The framework follows established planning competition standards:

- **30-minute timeout** per experiment (IPC 2023 standard)
- **8GB memory limit** (IPC 2023 standard)
- **Geometric mean speedup** calculation (standard metric)
- **Wilcoxon signed-rank tests** for paired statistical comparison
- **Effect size** reporting (Cohen's d)

## Expected Results

Based on the literature, you should expect:

### Centroids Heuristic
- **Better coverage** on loosely-coupled domains (logistics, satellite)
- **Moderate speedup** vs DTG baseline (~1.2-2x)
- **Good plan quality** (minimizes expected cost to goals)
- **Higher coordination overhead** in distributed settings

### MCS Heuristic  
- **Robust performance** across different coupling levels
- **Conservative planning** (minimizes worst-case scenarios)
- **Slightly higher search effort** but better worst-case guarantees
- **Good scalability** with number of agents

## Troubleshooting

### Common Issues

1. **No results generated**:
   - Check FMAP.jar exists in project root
   - Verify domain files exist in Domains/ directory
   - Check Java is installed and accessible

2. **All experiments timeout**:
   - Reduce timeout with `--timeout 300` (5 minutes)
   - Use `--quick` mode for testing
   - Check specific domain/problem combinations

3. **Import errors**:
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version is 3.8+

4. **Memory errors**:
   - Reduce memory limit with `--memory-limit 4096`
   - Test fewer domains/problems first

### Debug Mode

For debugging, modify the logging level in the scripts:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Heuristic Testing

To test only specific heuristics:
```bash
python run_experiments.py --heuristics 1 4 5  # DTG, Centroids, MCS only
```

### Domain-Specific Analysis

To test specific domains:
```bash
python run_experiments.py --domains driverlog rovers
```

### Parallel Execution

The framework supports natural parallelization by domain:
```bash
# Terminal 1
python run_experiments.py --domains driverlog logistics --results-dir results_batch1

# Terminal 2  
python run_experiments.py --domains rovers satellite --results-dir results_batch2
```

Then combine results for analysis.

### Result Analysis Only

If you have existing results:
```bash
python run_experiments.py --skip-experiments --results-dir existing_results
```

## Publication-Quality Output

The generated plots are suitable for academic publication:

- **300 DPI resolution** for crisp printing
- **Consistent color scheme** across all plots
- **Statistical annotations** (p-values, effect sizes)
- **Publication-standard formatting** (fonts, sizes, layouts)

## Integration with FMAP

This framework integrates seamlessly with your existing FMAP installation:

- Uses the same FMAP.jar and domain files
- Preserves all FMAP command-line interface conventions
- Collects metrics from FMAP's standard output
- Compatible with existing FMAP benchmarks

## Next Steps

After running experiments:

1. **Review the analysis report** (`results/analysis_report.txt`)
2. **Examine key visualizations** (especially `speedup_analysis_vs_DTG.png`)
3. **Check statistical significance** (`statistical_significance_heatmap.png`)
4. **Identify best-performing heuristics** for your specific domains
5. **Integrate findings** into your planning workflow

The framework provides everything needed for a comprehensive, publication-quality comparison of your new Centroids and MCS heuristics against established baselines. 