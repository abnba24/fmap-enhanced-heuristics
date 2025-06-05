#!/usr/bin/env python3
"""
Test script for FMAP experimental framework

This script verifies that the framework is set up correctly and shows
what experiments would be run without actually executing them.
"""

import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy.stats
        import psutil
        
        logger.info("✓ All required packages imported successfully")
        logger.info(f"  NumPy: {np.__version__}")
        logger.info(f"  Pandas: {pd.__version__}")
        logger.info(f"  Matplotlib: {plt.matplotlib.__version__}")
        logger.info(f"  Seaborn: {sns.__version__}")
        logger.info(f"  SciPy: {scipy.__version__}")
        logger.info(f"  PSUtil: {psutil.__version__}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        return False

def test_fmap_setup():
    """Test that FMAP and domain files are available"""
    fmap_jar = Path("../FMAP.jar")
    domains_dir = Path("../Domains")
    
    if fmap_jar.exists():
        logger.info("✓ FMAP.jar found")
    else:
        logger.warning("⚠ FMAP.jar not found - experiments will fail")
        return False
    
    if domains_dir.exists():
        domains = [d for d in domains_dir.iterdir() if d.is_dir()]
        logger.info(f"✓ Domains directory found with {len(domains)} domains:")
        for domain in sorted(domains)[:5]:  # Show first 5
            problems = [p for p in domain.iterdir() if p.is_dir()]
            logger.info(f"    {domain.name}: {len(problems)} problems")
        if len(domains) > 5:
            logger.info(f"    ... and {len(domains) - 5} more domains")
    else:
        logger.warning("⚠ Domains directory not found - no experiments available")
        return False
    
    return True

def test_experimental_config():
    """Test experimental configuration generation"""
    try:
        from run_experiments import create_experiment_configs, extract_agents_from_problem
        
        # Create a mock arguments object
        class MockArgs:
            domains = ['driverlog', 'logistics']
            heuristics = [1, 2, 4, 5]  # DTG, DTG+Landmarks, Centroids, MCS
            quick = True
            max_problems = 2
            timeout = 300
            memory_limit = 4096
        
        args = MockArgs()
        configs = create_experiment_configs(args)
        
        logger.info(f"✓ Generated {len(configs)} experiment configurations")
        
        # Group by heuristic
        heuristic_names = {1: "DTG", 2: "DTG+Landmarks", 4: "Centroids", 5: "MCS"}
        by_heuristic = {}
        for config in configs:
            h_name = heuristic_names.get(config.heuristic, f"Unknown({config.heuristic})")
            if h_name not in by_heuristic:
                by_heuristic[h_name] = []
            by_heuristic[h_name].append(config)
        
        logger.info("Experiments by heuristic:")
        for heuristic, heur_configs in by_heuristic.items():
            logger.info(f"  {heuristic}: {len(heur_configs)} experiments")
        
        # Show example configurations
        if configs:
            logger.info("Example experiments:")
            for i, config in enumerate(configs[:4]):  # Show first 4
                h_name = heuristic_names.get(config.heuristic, f"Unknown({config.heuristic})")
                logger.info(f"  {i+1}. {config.domain}/{config.problem} with {h_name} ({len(config.agents)} agents)")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test experimental configuration: {e}")
        return False

def test_analysis_modules():
    """Test that analysis modules can be imported"""
    try:
        from data_analyzer import ExperimentAnalyzer
        from visualizer import ExperimentVisualizer
        
        logger.info("✓ Analysis modules imported successfully")
        
        # Test that matplotlib backend works
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        import matplotlib.pyplot as plt
        
        # Create a simple test plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        plt.close(fig)
        
        logger.info("✓ Matplotlib plotting works")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test analysis modules: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Testing FMAP Experimental Framework")
    logger.info("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("FMAP setup", test_fmap_setup),
        ("Experimental configuration", test_experimental_config),
        ("Analysis modules", test_analysis_modules),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nTesting {test_name}...")
        try:
            if test_func():
                logger.info(f"✓ {test_name} passed")
                passed += 1
            else:
                logger.error(f"✗ {test_name} failed")
                failed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Framework is ready for experiments.")
        logger.info("\nTo run experiments:")
        logger.info("  Quick test: python run_experiments.py --quick")
        logger.info("  Full run:   python run_experiments.py")
    else:
        logger.error("❌ Some tests failed. Please fix issues before running experiments.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 