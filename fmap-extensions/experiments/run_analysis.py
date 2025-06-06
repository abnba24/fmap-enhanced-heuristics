#!/usr/bin/env python3
"""
Run comprehensive analysis on completed experiments using existing framework
"""

import sys
import os
from pathlib import Path

# Add the experiments directory to path so we can import from experiment_runner
sys.path.append(str(Path(__file__).parent))

# Import the ExperimentAnalyzer from the main experiment runner
from experiment_runner import ExperimentAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive analysis on all completed experiments"""
    results_dir = "results"
    
    logger.info("Starting comprehensive analysis...")
    logger.info(f"Looking for results in: {results_dir}")
    
    # Create analyzer
    analyzer = ExperimentAnalyzer(results_dir=results_dir)
    
    # Load results
    df = analyzer.load_results()
    
    if df.empty:
        logger.error("No results found to analyze")
        logger.info("Make sure experiments have completed and results exist in the results/ directory")
        return
    
    logger.info(f"Loaded {len(df)} experiment results")
    logger.info(f"Successful experiments: {len(df[df['coverage'] == True])}")
    logger.info(f"Domains: {df['domain'].unique()}")
    logger.info(f"Heuristics: {df['heuristic_name'].unique()}")
    
    # Generate comprehensive analysis
    analyzer.generate_comprehensive_analysis(df)
    
    logger.info("Analysis complete!")
    logger.info(f"- Plots saved to: {analyzer.plots_dir}")
    logger.info(f"- Statistical summary: {analyzer.results_dir}/statistical_summary.txt")
    
    # Print quick summary of what was generated
    plots_generated = list(analyzer.plots_dir.glob("*.png"))
    logger.info(f"Generated {len(plots_generated)} visualization plots:")
    for plot in sorted(plots_generated):
        logger.info(f"  - {plot.name}")

if __name__ == "__main__":
    main() 