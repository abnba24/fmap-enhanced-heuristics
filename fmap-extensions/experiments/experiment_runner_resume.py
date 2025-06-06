#!/usr/bin/env python3
"""
FMAP Experiment Runner with Resume Functionality

Enhanced version of experiment_runner.py that supports:
- Resuming from a specific experiment index
- Skipping already completed experiments
- Progress tracking and recovery
"""

import subprocess
import json
import time
import os
import sys
import threading
import psutil
import signal
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    domain: str
    problem: str
    heuristic: int
    agents: List[str]
    agent_files: List[Tuple[str, str]]  # (domain_file, problem_file) pairs
    timeout_seconds: int = 1800  # 30 minutes default
    memory_limit_mb: int = 8192  # 8 GB

@dataclass
class SearchMetrics:
    """Search performance metrics from FMAP execution"""
    coverage: bool = False
    wall_clock_time: float = 0.0
    cpu_time: float = 0.0
    peak_memory_mb: float = 0.0
    search_nodes: int = 0
    dtg_heuristic_values: List[int] = None
    landmark_heuristic_values: List[int] = None

@dataclass
class PlanMetrics:
    """Plan quality metrics from FMAP solution"""
    plan_found: bool = False
    plan_length: int = 0
    makespan: float = 0.0
    actions: List[Tuple[float, str]] = None

@dataclass
class ExperimentResult:
    """Complete results for a single experiment"""
    config: ExperimentConfig
    search: SearchMetrics
    plan: PlanMetrics
    error_message: Optional[str] = None
    stdout_log: str = ""
    stderr_log: str = ""

class ResourceMonitor:
    """Monitors resource usage during experiment execution"""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.cpu_times = []
        self.monitoring = False
        self.process = None
        
    def start_monitoring(self, process):
        self.process = process
        self.monitoring = True
        self.peak_memory = 0.0
        self.cpu_times = []
        
        def monitor():
            try:
                proc = psutil.Process(process.pid)
                while self.monitoring and process.poll() is None:
                    try:
                        memory_info = proc.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        self.peak_memory = max(self.peak_memory, memory_mb)
                        
                        cpu_times = proc.cpu_times()
                        self.cpu_times.append(cpu_times.user + cpu_times.system)
                        
                        time.sleep(0.1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    def get_peak_memory(self):
        return self.peak_memory
    
    def get_total_cpu_time(self):
        return sum(self.cpu_times) if self.cpu_times else 0.0

def kill_existing_java_processes():
    """Kill any existing FMAP Java processes to prevent port conflicts"""
    try:
        # Find and kill FMAP.jar processes
        result = subprocess.run(['pkill', '-f', 'FMAP.jar'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Killed existing FMAP Java processes")
        
        # Wait a moment for processes to terminate
        time.sleep(2)
        
        # Check if any Java processes are still running on FMAP
        result = subprocess.run(['pgrep', '-f', 'FMAP.jar'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.warning("Some FMAP processes may still be running")
            # Force kill if needed
            subprocess.run(['pkill', '-9', '-f', 'FMAP.jar'], 
                          capture_output=True, text=True)
            time.sleep(1)
        
    except Exception as e:
        logger.warning(f"Error killing Java processes: {e}")

def check_port_availability():
    """Check if common ports used by FMAP are available"""
    import socket
    
    # Common ports that FMAP might use
    ports_to_check = [8080, 8081, 8082, 8083, 8084, 8085]
    
    for port in ports_to_check:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                logger.warning(f"Port {port} is already in use")
                return False
        except Exception:
            pass
    
    return True

class ExperimentRunnerResume:
    """Enhanced experiment runner with resume functionality"""
    
    def __init__(self, fmap_jar_path: str = "FMAP.jar", results_dir: str = "results"):
        self.fmap_jar_path = fmap_jar_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Define heuristics to test
        self.heuristics = {
            "DTG": 1,
            "DTG+Landmarks": 2,
            "Inc_DTG+Landmarks": 3,
            "Centroids": 4,
            "MCS": 5
        }
    
    def get_finalized_sample_configs(self) -> List[ExperimentConfig]:
        """Generate configs from our finalized working_problems.json sample set"""
        configs = []
        
        # Load our finalized working problems
        working_problems_file = Path("../working_problems.json")
        if not working_problems_file.exists():
            logger.error(f"Working problems file not found: {working_problems_file}")
            return configs
        
        with open(working_problems_file, 'r') as f:
            working_problems = json.load(f)
        
        # Filter to only working problems
        test_problems = [
            p for p in working_problems 
            if p.get('test_result', {}).get('status') in ['tested_working', 'timeout_but_working', 'assumed_working']
        ]
        
        # Sort by agent count so smaller problems run first
        test_problems = sorted(test_problems, key=lambda x: x.get('agent_count', 0))
        
        logger.info(f"Loaded {len(test_problems)} working problems from finalized sample set")
        
        # All heuristics to test on each problem
        heuristics_to_test = [1, 2, 3, 4, 5]
        
        for problem_data in test_problems:
            domain = problem_data['domain']
            problem = problem_data['problem']
            
            problem_dir = Path(f"../Domains/{domain}/{problem}")
            
            if not problem_dir.exists():
                logger.warning(f"Problem directory not found: {problem_dir}")
                continue
            
            # Extract agent information
            agents, agent_files = self._extract_agents_from_problem(problem_dir)
            
            if not agents:
                logger.warning(f"No agents found for {domain}/{problem}")
                continue
            
            # Create config for each heuristic
            for heuristic_id in heuristics_to_test:
                config = ExperimentConfig(
                    domain=domain,
                    problem=problem,
                    heuristic=heuristic_id,
                    agents=agents,
                    agent_files=agent_files,
                    timeout_seconds=1800  # 30 minutes
                )
                configs.append(config)
        
        # Sort configs by agent count, then by domain/problem to ensure consistency
        configs = sorted(configs, key=lambda x: (len(x.agents), x.domain, x.problem, x.heuristic))
        
        logger.info(f"Generated {len(configs)} configurations from finalized working problems")
        
        return configs
    
    def _extract_agents_from_problem(self, problem_dir: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract agent names and files from problem directory with multi-domain support"""
        agents = []
        agent_files = []
        
        # Get all domain files
        domain_files = list(problem_dir.glob("Domain*.pddl"))
        if not domain_files:
            logger.warning(f"No domain file found in {problem_dir}")
            return agents, agent_files
        
        # Get agent problem files
        problem_files = list(problem_dir.glob("Problem*.pddl"))
        
        # Create a mapping of agent types to domain files for multi-domain setups
        domain_mapping = {}
        
        # For domains with multiple domain files, create a mapping
        if len(domain_files) > 1:
            for domain_file in domain_files:
                domain_filename = domain_file.name.lower()
                if "depot" in domain_filename:
                    # DomainDepotsDepot.pddl expects place agents but is used by truck agents (counterintuitive!)
                    domain_mapping["truck"] = str(domain_file)
                elif "truck" in domain_filename:
                    # DomainDepotsTruck.pddl expects truck agents but is used by depot/distributor agents (counterintuitive!)
                    domain_mapping["depot"] = str(domain_file)
                    domain_mapping["distributor"] = str(domain_file)
                else:
                    # For other types, use filename pattern
                    agent_type = domain_filename.replace("domain", "").replace(".pddl", "")
                    if agent_type:
                        domain_mapping[agent_type] = str(domain_file)
        else:
            # Single domain file - use for all agents
            default_domain = str(domain_files[0])
        
        for problem_file in problem_files:
            # Extract agent name from filename
            filename = problem_file.name
            if "Problem" in filename:
                # Extract agent name after the domain name
                # e.g., ProblemDriverlogdriver1.pddl -> driver1
                # e.g., ProblemDepotsdepot0.pddl -> depot0
                prefix_part = filename.replace("Problem", "").replace(".pddl", "")
                domain_name = problem_dir.parent.name.lower()
                
                if prefix_part.lower().startswith(domain_name):
                    agent_name = prefix_part[len(domain_name):]
                    if agent_name:
                        # Determine the appropriate domain file for this agent
                        agent_domain_file = None
                        
                        if len(domain_files) > 1:
                            # Look for agent type in the agent name
                            agent_type = None
                            for known_type in ["depot", "distributor", "truck"]:
                                if known_type in agent_name.lower():
                                    agent_type = known_type
                                    break
                            
                            if agent_type and agent_type in domain_mapping:
                                agent_domain_file = domain_mapping[agent_type]
                            else:
                                # Fallback: try to match by agent name pattern
                                for domain_key, domain_file_path in domain_mapping.items():
                                    if domain_key in agent_name.lower():
                                        agent_domain_file = domain_file_path
                                        break
                                
                                # If still no match, use the first domain file
                                if not agent_domain_file:
                                    agent_domain_file = str(domain_files[0])
                        else:
                            agent_domain_file = default_domain
                        
                        if agent_domain_file:
                            agents.append(agent_name)
                            agent_files.append((agent_domain_file, str(problem_file)))
                            logger.debug(f"Agent {agent_name} -> domain: {Path(agent_domain_file).name}")
        
        logger.info(f"Extracted {len(agents)} agents from {problem_dir}: {agents}")
        return agents, agent_files
    
    def get_completed_experiments(self) -> set:
        """Get set of experiment indices that are already completed"""
        completed = set()
        
        for result_file in self.results_dir.glob("result_*.json"):
            try:
                # Extract experiment index from filename
                # Format: result_XXXX_domain_problem_heuristic.json
                filename = result_file.name
                if filename.startswith("result_") and filename.endswith(".json"):
                    index_str = filename.split("_")[1]
                    index = int(index_str)
                    
                    # Verify the file contains valid results (not just error)
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                        # Check if it's a valid result (not just address error)
                        if "Address already in use" not in result_data.get('stdout_log', ''):
                            completed.add(index)
            except (ValueError, json.JSONDecodeError, KeyError):
                continue
        
        return completed
    
    def run_experiments_with_resume(self, configs: List[ExperimentConfig] = None, 
                                  start_index: int = 0, force_restart: bool = False, 
                                  specific_experiments: List[int] = None) -> List[ExperimentResult]:
        """Run experiments with resume functionality"""
        if configs is None:
            configs = self.get_finalized_sample_configs()
        
        # CRITICAL: Kill any existing Java processes before starting
        logger.info("Cleaning up any existing Java processes...")
        kill_existing_java_processes()
        
        # Check port availability
        if not check_port_availability():
            logger.warning("Some ports may still be in use. Waiting 5 seconds...")
            time.sleep(5)
        
        # Get completed experiments
        completed = self.get_completed_experiments() if not force_restart else set()
        
        logger.info(f"Found {len(completed)} completed experiments")
        if completed and not force_restart:
            logger.info(f"Completed experiments: {sorted(list(completed))[:10]}{'...' if len(completed) > 10 else ''}")
        
        logger.info(f"Starting {len(configs)} experiments from index {start_index}")
        results = []
        
        for i, config in enumerate(configs):
            current_index = i
            
            # If specific experiments are requested, only run those
            if specific_experiments and current_index not in specific_experiments:
                continue
            
            # Skip if before start index
            if current_index < start_index:
                logger.debug(f"Skipping experiment {current_index} (before start index {start_index})")
                continue
            
            # Skip if already completed
            if current_index in completed and not force_restart:
                logger.info(f"Skipping experiment {current_index} (already completed)")
                continue
            
            logger.info(f"Running experiment {current_index}/{len(configs)-1}: {config.domain}/{config.problem} (heuristic {config.heuristic})")
            logger.info(f"Progress: {current_index+1}/{len(configs)} ({((current_index+1)/len(configs)*100):.1f}%)")
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            # Save intermediate result
            self._save_result(result, current_index)
            
            # Log result summary
            if result.search.coverage:
                logger.info(f"✓ Success: {result.search.wall_clock_time:.2f}s, {result.search.peak_memory_mb:.1f}MB")
            else:
                error_summary = result.error_message or "Unknown error"
                if "Address already in use" in result.stdout_log:
                    error_summary = "Address already in use"
                logger.warning(f"✗ Failed: {error_summary}")
        
        # Save all results
        if results:
            self._save_all_results(results)
        
        return results
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment"""
        logger.debug(f"Starting experiment: {config.domain}/{config.problem} with heuristic {config.heuristic}")
        
        result = ExperimentResult(
            config=config,
            search=SearchMetrics(),
            plan=PlanMetrics()
        )
        
        try:
            # Build command (no longer using agent list file)
            cmd = self._build_fmap_command(config, "")
            
            # Run experiment with monitoring
            monitor = ResourceMonitor()
            start_time = time.time()
            
            # Run from the parent directory (where the JAR and Domains are located)
            working_dir = Path(self.fmap_jar_path).parent.resolve()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(working_dir)
            )
            
            monitor.start_monitoring(process)
            
            try:
                stdout, stderr = process.communicate(timeout=config.timeout_seconds)
                wall_clock_time = time.time() - start_time
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                wall_clock_time = config.timeout_seconds
                result.error_message = "Timeout"
            
            monitor.stop_monitoring()
            
            # Store logs
            result.stdout_log = stdout
            result.stderr_log = stderr
            
            # Parse output
            self._parse_fmap_output(stdout, stderr, result)
            
            # Set timing and memory
            result.search.wall_clock_time = wall_clock_time
            result.search.cpu_time = monitor.get_total_cpu_time()
            result.search.peak_memory_mb = monitor.get_peak_memory()
            
            # Clean up (no longer needed since we don't create agent list file)
            pass
        
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Experiment failed: {e}")
        
        return result
    
    def _create_agent_list_file(self, config: ExperimentConfig) -> str:
        """Create temporary agent list file for FMAP"""
        agent_list_content = f"{len(config.agents)}\n"
        
        # Convert to absolute paths to avoid working directory issues
        working_dir = Path(self.fmap_jar_path).parent.resolve()
        
        for agent, (domain_file, problem_file) in zip(config.agents, config.agent_files):
            # Convert relative paths to absolute paths
            if not Path(domain_file).is_absolute():
                domain_file = str((working_dir / domain_file).resolve())
            if not Path(problem_file).is_absolute():
                problem_file = str((working_dir / problem_file).resolve())
                
            agent_list_content += f"{agent} {domain_file} {problem_file}\n"
        
        agent_list_file = f"temp_agent_list_{int(time.time())}.txt"
        with open(agent_list_file, 'w') as f:
            f.write(agent_list_content)
        
        return agent_list_file
    
    def _build_fmap_command(self, config: ExperimentConfig, agent_list_file: str) -> List[str]:
        """Build FMAP command"""
        # Convert to absolute path to avoid working directory issues
        jar_path = Path(self.fmap_jar_path).resolve()
        
        # Build command in the format: java -jar FMAP.jar [<agent-name> <domain-file> <problem-file>]+ [-h heuristic]
        cmd = [
            "java",
            f"-Xmx{config.memory_limit_mb}m",
            "-jar", str(jar_path)
        ]
        
        # Add each agent with its domain and problem files
        for agent, (domain_file, problem_file) in zip(config.agents, config.agent_files):
            # Convert relative paths to absolute paths
            if not Path(domain_file).is_absolute():
                domain_file = str(Path(domain_file).resolve())
            if not Path(problem_file).is_absolute():
                problem_file = str(Path(problem_file).resolve())
            
            cmd.extend([agent, domain_file, problem_file])
        
        # Add heuristic option
        cmd.extend(["-h", str(config.heuristic)])
        
        return cmd
    
    def _parse_fmap_output(self, stdout: str, stderr: str, result: ExperimentResult):
        """Parse FMAP output to extract metrics"""
        lines = stdout.split('\n')
        
        # Check for coverage/solution found
        if ("Solution plan" in stdout or "*** Coverage ***" in stdout or 
            any(line.strip() and ":" in line and "(" in line and ")" in line 
                for line in lines if not line.strip().startswith(";"))):
            result.search.coverage = True
            result.plan.plan_found = True
        
        # Parse plan actions if solution found
        if result.plan.plan_found:
            actions = []
            for line in lines:
                line = line.strip()
                # Look for action lines like "0: (Walk driver1 s2 p1-2)"
                if (line and ":" in line and "(" in line and ")" in line and 
                    not line.startswith(";") and not line.startswith("Hdtg")):
                    try:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            timestamp = float(parts[0].strip())
                            action = parts[1].strip()
                            actions.append((timestamp, action))
                    except ValueError:
                        continue
            
            if actions:
                result.plan.actions = actions
                result.plan.plan_length = len(actions)
                if actions:
                    result.plan.makespan = max(action[0] for action in actions)
        
        # Parse search nodes from heuristic evaluations
        heuristic_lines = [line for line in lines if "Hdtg" in line]
        if heuristic_lines:
            result.search.search_nodes = len(heuristic_lines)
        
        # Parse timing and other metrics from output
        for line in lines:
            line = line.strip()
            
            if "Time to solve:" in line:
                try:
                    time_str = line.split("Time to solve:")[1].strip().split()[0]
                    result.search.wall_clock_time = float(time_str)
                except (IndexError, ValueError):
                    pass
    
    def _save_result(self, result: ExperimentResult, index: int):
        """Save a single experiment result"""
        filename = f"result_{index:04d}_{result.config.domain}_{result.config.problem}_{result.config.heuristic}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    def _save_all_results(self, results: List[ExperimentResult]):
        """Save all results in a single file"""
        filepath = self.results_dir / "all_results.json"
        
        results_data = [asdict(result) for result in results]
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(results)} results to {filepath}")

def main():
    """Main entry point with resume functionality"""
    parser = argparse.ArgumentParser(description='FMAP Experiment Runner with Resume')
    parser.add_argument('--jar', default='../FMAP.jar', help='Path to FMAP jar file')
    parser.add_argument('--start', type=int, default=0, 
                        help='Start from specific experiment index (default: 0)')
    parser.add_argument('--force-restart', action='store_true', 
                        help='Force restart all experiments, ignore completed ones')
    parser.add_argument('--list-completed', action='store_true',
                        help='List completed experiments and exit')
    parser.add_argument('--kill-java', action='store_true',
                        help='Kill existing Java processes and exit')
    parser.add_argument('--experiments', type=str, 
                        help='Comma-separated list of specific experiment indices to run')
    
    args = parser.parse_args()
    
    # Kill Java processes if requested
    if args.kill_java:
        logger.info("Killing existing Java processes...")
        kill_existing_java_processes()
        logger.info("Done.")
        return
    
    if not os.path.exists(args.jar):
        logger.error(f"FMAP jar file not found: {args.jar}")
        sys.exit(1)
    
    runner = ExperimentRunnerResume(fmap_jar_path=args.jar)
    
    # List completed experiments if requested
    if args.list_completed:
        completed = runner.get_completed_experiments()
        print(f"Completed experiments: {len(completed)}")
        if completed:
            completed_sorted = sorted(list(completed))
            print(f"Indices: {completed_sorted}")
            print(f"Range: {min(completed_sorted)} - {max(completed_sorted)}")
            
            # Show gaps for resume
            gaps = []
            for i in range(max(completed_sorted) + 1):
                if i not in completed:
                    gaps.append(i)
            
            if gaps:
                print(f"Missing experiments: {gaps[:20]}{'...' if len(gaps) > 20 else ''}")
                print(f"Suggested resume from: {min(gaps)}")
        return
    
    # Generate configurations
    configs = runner.get_finalized_sample_configs()
    
    if not configs:
        logger.error("No experiment configurations generated")
        sys.exit(1)
    
    # Parse specific experiments if provided
    specific_experiments = None
    if args.experiments:
        try:
            specific_experiments = [int(x.strip()) for x in args.experiments.split(',')]
            logger.info(f"Running specific experiments: {specific_experiments}")
        except ValueError:
            logger.error("Invalid experiment list format. Use comma-separated integers.")
            sys.exit(1)
    
    logger.info(f"Total experiments: {len(configs)}")
    logger.info(f"Starting from index: {args.start}")
    logger.info(f"Force restart: {args.force_restart}")
    
    # Run experiments
    results = runner.run_experiments_with_resume(
        configs=configs,
        start_index=args.start,
        force_restart=args.force_restart,
        specific_experiments=specific_experiments
    )
    
    # Print summary
    successful = sum(1 for r in results if r.search.coverage)
    logger.info(f"Experiment summary: {successful}/{len(results)} successful")

if __name__ == "__main__":
    main() 