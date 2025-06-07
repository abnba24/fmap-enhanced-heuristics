# Analysis Outputs

This directory contains all experimental results, plots, and analysis outputs from the FMAP heuristic evaluation framework.

## Directory Structure

- **`results/`** - Raw experimental results in JSON format and summary files
  - `all_results.json` - Complete experimental dataset (43MB)
  - `result_*.json` - Individual experiment results
  - `statistical_summary.txt` - Statistical analysis summary
  - `universal_analysis_summary.txt` - Universal analysis overview
  - `plots/` - Generated visualization files and CSV data

- **`plots/`** - Main plots directory (if separate from results/plots)

- **Test Result Files** - Individual test outputs from development
  - `centroids_test_result.txt` - Centroids heuristic test results
  - `dtg_result.txt` - DTG heuristic test results
  - `dtg_test_result.txt` - DTG test validation
  - `test_new_port_result.txt` - Port compatibility test results

## Key Files

### Visualisations (in results/plots/)
- `heuristic_comparison_analysis.png` - Main heuristic comparison
- `performance_matrices.png` - Performance matrix visualisation
- `scalability_analysis.png` - Scalability analysis
- `heuristic_performance_by_agent_count.png` - Performance vs agent count
- `heuristic_performance_by_domain.png` - Performance by planning domain

### Data Tables (in results/plots/)
- `comprehensive_metrics_comparison.csv` - Complete metrics comparison
- `domain_analysis.csv` - Domain-specific analysis
- `heuristic_summary.csv` - Heuristic performance summary
- `detailed_analysis_report.md` - Detailed analysis report

## Heuristics Evaluated

1. **FF** - Fast-Forward heuristic
2. **DTG** - Domain Transition Graph heuristic
3. **DTG + Landmarks** - DTG with landmarks
4. **Centroids** - Our implementation (minimises mean cost to goals)
5. **MCS** - Minimum Covering States (minimises max cost to goals)

## Domains Tested

- `driverlog` - Driver logistics planning
- `depots` - Depot management
- `zenotravel` - Travel planning
- `elevators` - Elevator control
- `openstacks` - Stack manipulation
- Various problem files (Pfile1-Pfile20+)

## Generated on

Last updated: June 2024
Framework: FMAP Extensions with Centroids and MCS heuristics 