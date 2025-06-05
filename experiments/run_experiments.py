#!/usr/bin/env python3
"""
FMAP Heuristic Comparison Experiments - Main Controller

This script orchestrates the complete experimental pipeline:
1. Run experiments comparing heuristics
2. Analyze results with statistical tests
3. Generate comprehensive visualizations
4. Produce final report

Usage:
    python run_experiments.py [--quick] [--domains DOMAINS] [--heuristics HEURISTICS]
"""

import argparse
import sys
import time
from pathlib import Path
import logging

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent))

from experiment_runner import ExperimentRunner, ExperimentConfig
from data_analyzer import ExperimentAnalyzer
from visualizer import ExperimentVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run FMAP heuristic comparison experiments')
    
    parser.add_argument('--quick', action='store_true',
                      help='Run quick test with limited domains and problems')
    
    parser.add_argument('--domains', nargs='+', 
                      default=['driverlog', 'logistics', 'rovers', 'satellite'],
                      help='Domains to test (default: driverlog logistics rovers satellite)')
    
    parser.add_argument('--heuristics', nargs='+', type=int,
                      default=[1, 2, 3, 4, 5],
                      help='Heuristic IDs to test (default: 1 2 3 4 5 for all heuristics)')
    
    parser.add_argument('--max-problems', type=int, default=5,
                      help='Maximum number of problems per domain (default: 5)')
    
    parser.add_argument('--timeout', type=int, default=1800,
                      help='Timeout per experiment in seconds (default: 1800 = 30 min)')
    
    parser.add_argument('--memory-limit', type=int, default=8192,
                      help='Memory limit in MB (default: 8192 = 8GB)')
    
    parser.add_argument('--fmap-jar', default='../FMAP.jar',
                      help='Path to FMAP jar file (default: ../FMAP.jar)')
    
    parser.add_argument('--results-dir', default='results',
                      help='Results directory (default: results)')
    
    parser.add_argument('--plots-dir', default='plots',
                      help='Plots directory (default: plots)')
    
    parser.add_argument('--skip-experiments', action='store_true',
                      help='Skip experiments and only analyze existing results')
    
    parser.add_argument('--skip-analysis', action='store_true',
                      help='Skip analysis and only run experiments')
    
    parser.add_argument('--skip-plots', action='store_true',
                      help='Skip plot generation')
    
    return parser.parse_args()

def create_experiment_configs(args) -> list:
    """Create experiment configurations based on arguments"""
    configs = []
    domains_dir = Path("../Domains")
    
    # Heuristic name mapping
    heuristic_names = {
        1: "DTG",
        2: "DTG+Landmarks", 
        3: "Inc_DTG+Landmarks",
        4: "Centroids",
        5: "MCS"
    }
    
    logger.info(f"Testing heuristics: {[heuristic_names.get(h, f'Unknown({h})') for h in args.heuristics]}")
    logger.info(f"Testing domains: {args.domains}")
    
    for domain_name in args.domains:
        domain_path = domains_dir / domain_name
        if not domain_path.exists():
            logger.warning(f"Domain directory not found: {domain_path}")
            continue
            
        # Find problem instances
        problem_dirs = sorted([d for d in domain_path.iterdir() if d.is_dir()])
        
        if args.quick:
            problem_dirs = problem_dirs[:2]  # Only 2 problems for quick test
        else:
            problem_dirs = problem_dirs[:args.max_problems]
        
        logger.info(f"Found {len(problem_dirs)} problems in {domain_name}")
        
        for problem_dir in problem_dirs:
            # Extract agents from problem files
            agents, agent_files = extract_agents_from_problem(problem_dir)
            
            if len(agents) >= 2:  # Only test multi-agent problems
                logger.debug(f"  {problem_dir.name}: {len(agents)} agents")
                
                for heuristic_id in args.heuristics:
                    config = ExperimentConfig(
                        domain=domain_name,
                        problem=problem_dir.name,
                        heuristic=heuristic_id,
                        agents=agents,
                        agent_files=agent_files,
                        timeout_seconds=args.timeout,
                        memory_limit_mb=args.memory_limit
                    )
                    configs.append(config)
            else:
                logger.debug(f"  Skipping {problem_dir.name}: only {len(agents)} agents")
    
    return configs

def extract_agents_from_problem(problem_dir: Path):
    """Extract agent information from problem directory"""
    agents = []
    agent_files = []
    
    domain_name = problem_dir.parent.name
    
    # Find domain file
    domain_file = None
    for file in problem_dir.iterdir():
        if file.name.startswith("Domain") and file.suffix == ".pddl":
            domain_file = str(file)
            break
    
    if not domain_file:
        return agents, agent_files
        
    # Find agent problem files
    for file in problem_dir.iterdir():
        if file.name.startswith("Problem") and file.suffix == ".pddl":
            # Extract agent name from filename
            agent_name = file.name.replace("Problem", "").replace(domain_name, "").replace(".pddl", "")
            if agent_name:
                agents.append(agent_name)
                agent_files.append((domain_file, str(file)))
    
    return agents, agent_files

def run_experiments_phase(args) -> bool:
    """Run the experiments phase"""
    logger.info("=" * 60)
    logger.info("PHASE 1: RUNNING EXPERIMENTS")
    logger.info("=" * 60)
    
    if not Path(args.fmap_jar).exists():
        logger.error(f"FMAP jar file not found: {args.fmap_jar}")
        return False
    
    # Create experiment configurations
    configs = create_experiment_configs(args)
    
    if not configs:
        logger.error("No experiment configurations generated")
        return False
    
    logger.info(f"Generated {len(configs)} experiment configurations")
    
    if args.quick:
        logger.info("Running in QUICK mode - limited experiments")
    
    # Run experiments
    runner = ExperimentRunner(fmap_jar_path=args.fmap_jar, results_dir=args.results_dir)
    
    start_time = time.time()
    results = runner.run_experiments(configs)
    elapsed_time = time.time() - start_time
    
    # Summary
    successful = sum(1 for r in results if r.performance.coverage)
    logger.info(f"Experiments completed in {elapsed_time:.1f} seconds")
    logger.info(f"Results: {successful}/{len(results)} successful ({100*successful/len(results):.1f}%)")
    
    return len(results) > 0

def run_analysis_phase(args) -> bool:
    """Run the analysis phase"""
    logger.info("=" * 60)
    logger.info("PHASE 2: ANALYZING RESULTS")
    logger.info("=" * 60)
    
    results_file = Path(args.results_dir) / "all_results.json"
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return False
    
    # Analyze results
    analyzer = ExperimentAnalyzer(str(results_file))
    
    try:
        analyzer.load_results()
        report = analyzer.generate_summary_report()
        
        # Print report
        print("\n" + report)
        
        # Save report
        report_path = Path(args.results_dir) / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis report saved to {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def run_visualization_phase(args) -> bool:
    """Run the visualization phase"""
    logger.info("=" * 60)
    logger.info("PHASE 3: GENERATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    results_file = Path(args.results_dir) / "all_results.json"
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return False
    
    # Generate visualizations
    visualizer = ExperimentVisualizer(str(results_file), args.plots_dir)
    
    try:
        visualizer.create_all_visualizations()
        logger.info(f"Visualizations saved to {args.plots_dir}/")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return False

def main():
    """Main entry point"""
    args = parse_arguments()
    
    logger.info("FMAP Heuristic Comparison Experiments")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  FMAP jar: {args.fmap_jar}")
    logger.info(f"  Results dir: {args.results_dir}")
    logger.info(f"  Plots dir: {args.plots_dir}")
    logger.info(f"  Timeout: {args.timeout}s")
    logger.info(f"  Memory limit: {args.memory_limit}MB")
    logger.info(f"  Quick mode: {args.quick}")
    
    success = True
    
    # Phase 1: Run experiments
    if not args.skip_experiments:
        success &= run_experiments_phase(args)
        if not success:
            logger.error("Experiments phase failed")
            return 1
    else:
        logger.info("Skipping experiments phase")
    
    # Phase 2: Analyze results
    if not args.skip_analysis and success:
        success &= run_analysis_phase(args)
        if not success:
            logger.error("Analysis phase failed")
            return 1
    else:
        logger.info("Skipping analysis phase")
    
    # Phase 3: Generate visualizations
    if not args.skip_plots and success:
        success &= run_visualization_phase(args)
        if not success:
            logger.error("Visualization phase failed")
            return 1
    else:
        logger.info("Skipping visualization phase")
    
    if success:
        logger.info("=" * 60)
        logger.info("ALL PHASES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Results: {args.results_dir}/")
        logger.info(f"Plots: {args.plots_dir}/")
        logger.info(f"Report: {args.results_dir}/analysis_report.txt")
        return 0
    else:
        logger.error("Some phases failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 