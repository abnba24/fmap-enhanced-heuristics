# FMAP Extensions

This directory contains our extensions to the original FMAP (Multi-Agent Planning) system developed at Universidad Politécnica de Valencia.

## Directory Structure

### `/automation/`
Automation tools for running experiments and collecting statistics:
- `fmap_gui_automation.py` - GUI automation for FMAP experiments
- `domain_automation_pipeline.py` - Automated domain processing pipeline
- `fmap_stats_wrapper.py` - Statistics collection wrapper
- `simple_fmap_stats.py` - Simplified statistics interface
- `quick_gui_test.py` - Quick GUI automation testing

### `/experiments/`
Experiment runners and test scripts:
- `run_comprehensive_experiments.py` - Cross-domain experiment runner
- `monitor_experiments.py` - Experiment monitoring utilities
- `resume_experiments.py` - Experiment resumption functionality
- `test_centroids_fixed.sh` - Centroids heuristic validation
- `test_mcs_implementation.sh` - MCS heuristic validation
- `verify_mcs_logic.sh` - MCS logic verification
- Additional experiment data and configurations

### `/analysis/`
Analysis and visualisation tools:
- `regenerate_plots_improved.py` - Enhanced plot generation
- `plot_comprehensive_results.py` - Comprehensive results visualisation

### `/documentation/`
Technical documentation and guides:
- `AUTOMATION_AGENT_PROMPT.md` - Automation agent documentation
- `EXPERIMENT_SYSTEMS.md` - Experimental systems overview
- `FMAP_STATS_GUIDE.md` - Statistics collection guide
- `heuristic_comparison_results.md` - Heuristic comparison analysis

### `/src/`
Java source code for new heuristics:
- `MCSHeuristic.java` - Minimum Covering States heuristic implementation
- `CentroidsHeuristic.java` - Centroids heuristic implementation
- Modified factory classes for heuristic integration

### `/data/`
Experimental data and configuration files:
- Working problem sets
- Experiment results
- Performance matrices
- Configuration templates

### `/Domains/`
PDDL domain and problem files for testing:
- Multi-agent planning domains used in experiments
- Problem instances organized by domain type

## Core Extension Files

### JAR Files
- `FMAP.jar` - Our extended FMAP version with new heuristics
- `FMAP_final.jar` - Final release version

## New Heuristics

### 1. Minimum Covering States (MCS) Heuristic
- **Formula**: `f_mcs(s) = max{ĥ(s, Gi)}`
- **Based on**: Pozanco et al. "Finding Centroids and Minimum Covering States in Planning" (ICAPS 2019)
- **Purpose**: Provides upper bound estimation using maximum heuristic values across goal subsets

### 2. Centroids Heuristic
- **Formula**: `h_centroids(s) = (1/|G|) * Σ h(s, g_i)`
- **Purpose**: Averages heuristic values across all individual goals
- **Application**: Balanced approach for multi-goal planning scenarios

## Usage

1. **Running Experiments**: Use scripts in `/experiments/` directory
2. **Automation**: Leverage tools in `/automation/` for batch processing
3. **Analysis**: Generate visualisations using `/analysis/` tools
4. **Documentation**: Refer to `/documentation/` for detailed guides

## Integration with Original FMAP

Our extensions maintain compatibility with the original FMAP system while adding:
- New heuristic algorithms (MCS and Centroids)
- Enhanced statistics collection
- Automated experimentation framework
- Comprehensive analysis tools

## Citation

If you use these extensions in your research, please cite both:
1. Original FMAP: Sapena, O., Torreno, A., Onaindia, E. "FMAP: Distributed cooperative multi-agent planning" (Applied Intelligence, 2015)
2. Our extensions: [Your research paper citation] 