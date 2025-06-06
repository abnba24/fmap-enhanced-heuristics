#!/usr/bin/env python3
"""
Resume FMAP Experiments

This script resumes the experiment runner from experiment 0060 and also reruns experiment 0000.
It uses the existing experiment_runner_resume.py but with fixes for multi-domain files.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Add experiments directory to path
sys.path.append('experiments')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(experiment_index):
    """Run a specific experiment using the resume runner"""
    
    logger.info(f"Running experiment {experiment_index}")
    
    # Change to experiments directory
    original_dir = os.getcwd()
    os.chdir('experiments')
    
    try:
        # Run the specific experiment
        cmd = [
            'python3', 'experiment_runner_resume.py',
            '--experiments', str(experiment_index),
            '--jar', '../FMAP.jar'
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✓ Experiment {experiment_index} completed successfully")
            return True
        else:
            logger.error(f"✗ Experiment {experiment_index} failed")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    finally:
        os.chdir(original_dir)

def resume_from_60():
    """Resume experiments starting from experiment 0060"""
    
    logger.info("Resuming experiments from 0060")
    
    # Change to experiments directory
    original_dir = os.getcwd()
    os.chdir('experiments')
    
    try:
        # Run starting from experiment 60
        cmd = [
            'python3', 'experiment_runner_resume.py',
            '--start', '60',
            '--jar', '../FMAP.jar'
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        
        if result.returncode == 0:
            logger.info("✓ Resume from 0060 completed successfully")
            return True
        else:
            logger.error("✗ Resume from 0060 failed")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    finally:
        os.chdir(original_dir)

def main():
    """Main function to resume experiments"""
    
    logger.info("Starting experiment resumption")
    
    # First, rerun experiment 0000
    logger.info("=== RERUNNING EXPERIMENT 0000 ===")
    success_0000 = run_experiment(0)
    
    if success_0000:
        logger.info("✓ Experiment 0000 completed successfully")
    else:
        logger.error("✗ Experiment 0000 failed - continuing anyway")
    
    # Then resume from experiment 0060
    logger.info("=== RESUMING FROM EXPERIMENT 0060 ===")
    success_resume = resume_from_60()
    
    if success_resume:
        logger.info("✓ All experiments completed successfully")
    else:
        logger.error("✗ Resume failed")
    
    # Summary
    logger.info("=== SUMMARY ===")
    logger.info(f"Experiment 0000 rerun: {'SUCCESS' if success_0000 else 'FAILED'}")
    logger.info(f"Resume from 0060: {'SUCCESS' if success_resume else 'FAILED'}")

if __name__ == "__main__":
    main() 