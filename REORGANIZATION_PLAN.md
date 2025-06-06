# FMAP Codebase Reorganization Plan (Updated After Cleanup)

This document outlines the current clean state of the FMAP codebase after removing redundant files and organizing our extensions.

## Cleanup Summary

**Removed Files:**
- Temporary/redundant files: `temp_count_experiments.py`, `temp_agent.txt`, various `.log` files
- Virtual environments: `gui_automation_env/`, `analysis_env/`, `experiments/venv/`
- Redundant JARs: `FMAP_updated.jar`, `FMAP_new.jar`
- Duplicate scripts: `regenerate_plots.py` (kept improved version), `test_5_agent_problems.py`, `test_5_agent_quick.py` (kept fixed version)
- Shell scripts: `test_centroids.sh` (kept fixed version)
- Compiled classes: `org/` directory with `.class` files
- Eclipse settings: `.settings/` directory
- Python cache: `__pycache__/` directories
- Emojis: Removed from all Python files for professional appearance

## Current Codebase Structure

### Original FMAP (Universidad Politécnica de Valencia)
- **Location**: `fmap-original/` directory (already organized)
- **Core JAR**: `FMAP_original.jar` - Original implementation
- **License**: GNU General Public License v3
- **Authors**: Oscar Sapena, Alejandro Torreno, Eva Onaindia

### Our FMAP Extensions

#### Core Extension Files:
- **`FMAP.jar`** - Our extended version with new heuristics
- **`FMAP_final.jar`** - Final release version

#### New Heuristic Implementations:
- Located in `fmap-extensions/src/` (Java source files)
- **MCSHeuristic.java** - Minimum Covering States heuristic
- **CentroidsHeuristic.java** - Centroids heuristic

#### Test and Validation Scripts:
- **`test_more_5_agent_problems_fixed.py`** - Core 5-agent problem testing (KEEP)
- **`test_centroids_fixed.sh`** - Centroids heuristic validation (KEEP)
- **`test_mcs_implementation.sh`** - MCS heuristic validation (KEEP)
- **`verify_mcs_logic.sh`** - MCS logic verification (KEEP)
- **`test_stats_example.py`** - Statistics wrapper testing (KEEP)

#### Experiment Framework:
- **`run_final_experiments.py`** - Comprehensive experiment runner (KEEP)
- **`run_final_experiments_simple.py`** - Simplified experiment runner (KEEP)
- **`run_comprehensive_experiments.py`** - Cross-domain experiments (KEEP)
- **`resume_experiments.py`** - Experiment resumption utility (KEEP)
- **`monitor_experiments.py`** - Experiment monitoring (KEEP)

#### Analysis and Visualization:
- **`regenerate_plots_improved.py`** - Enhanced plot generation (KEEP)
- **`plot_comprehensive_results.py`** - Results visualization (KEEP)

#### Automation Tools:
- **`fmap_gui_automation.py`** - GUI automation for experiments (KEEP)
- **`domain_automation_pipeline.py`** - Domain processing pipeline (KEEP)
- **`fmap_stats_wrapper.py`** - Statistics collection wrapper (KEEP)
- **`simple_fmap_stats.py`** - Simple statistics interface (KEEP)
- **`quick_gui_test.py`** - GUI automation testing (KEEP)

#### Data and Results:
- **`working_problems.json`** - Verified working problem set (KEEP)
- **`experiment_results.json`** - Experiment output data (KEEP)
- **`targeted_experiment_results.json`** - Focused experiment results (KEEP)
- **`performance_matrix.csv`** - Performance comparison matrix (KEEP)
- **`heuristic_experiment_template.csv`** - Experiment template (KEEP)
- **`results/`** - Directory with experimental results (KEEP)
- **`experiments/`** - Complete experimental framework (KEEP)

#### Documentation:
- **Core Documentation** (KEEP ALL):
  - `01_Methodology_and_Design.md`
  - `02_Implementation.md`
  - `03_Experiments_and_Evaluation_Methodology.md`
  - `04_Results_and_Analysis.md`
  - `FMAP_Extension_Design_and_Implementation.md`
  - `FMAP_Extension_Comparison.md`
  - `MY_FMAP_EXTENSIONS.md`
- **Experimental Documentation** (KEEP ALL):
  - `GUI_AUTOMATION_README.md`
  - `FMAP_STATS_GUIDE.md`
  - `GUI_HEURISTIC_EXPERIMENT_PLAN.md`
  - `AUTOMATION_AGENT_PROMPT.md`
  - `EXPERIMENT_SYSTEMS.md`
  - Various experiment summaries and analysis reports

#### Test Domains:
- **`Domains/`** - PDDL domain and problem files for testing (KEEP)

#### Supporting Infrastructure:
- **`requirements.txt`** - Python dependencies (KEEP)
- **`build.xml`** - Ant build file (KEEP)
- **`MANIFEST.MF`** - JAR manifest (KEEP)

## Current State Benefits:

- **Clean Organization**: Removed all temporary, redundant, and cache files
- **Professional Appearance**: No emojis in code, clean formatting
- **Clear Attribution**: Separation between original FMAP and our extensions
- **Maintained Functionality**: All core tests and experiments preserved
- **Reduced Size**: Eliminated virtual environments and compiled cache files
- **Academic Quality**: Proper documentation and organization for research publication

## Files to Keep vs. Remove Decision Matrix:

**KEPT (Core Functionality):**
- All test scripts ending in `_fixed.py` or `.sh` - these are the working versions
- All experiment runners - each serves a different purpose
- All documentation - needed for research and publication
- All results data - valuable experimental data
- Working problem sets and templates

**REMOVED (Redundant/Temporary):**
- Original versions when fixed versions exist
- Temporary log files and cache directories
- Virtual environments (can be recreated)
- Duplicate plotting scripts (kept improved version)
- Compiled class files (can be regenerated)

The codebase is now clean, professional, and ready for research publication while maintaining all core functionality and experimental capabilities. 