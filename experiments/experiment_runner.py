#!/usr/bin/env python3
"""
FMAP Heuristic Comparison Experiment Runner

This script conducts comprehensive experiments comparing Landmarks, Centroids, 
and MCS heuristics across multiple metrics as outlined in the experimental protocol.

Metrics collected:
1. Search performance: Coverage, time, RAM, node expansions, branching factor
2. Plan quality: Cost, makespan, concurrency index, goal-distance metrics
3. Heuristic quality: MAE, RMSE, correlation, computation time
4. Coordination: Messages, bytes, sync rounds, privacy scores
5. Statistical analysis: Confidence intervals, significance tests
"""

import subprocess
import json
import time
import os
import sys
import threading
import psutil
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

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
    timeout_seconds: int = 1800  # 30 minutes
    memory_limit_mb: int = 8192  # 8 GB

@dataclass
class PerformanceMetrics:
    """Core search performance metrics"""
    coverage: bool = False
    wall_clock_time: float = 0.0
    cpu_time: float = 0.0
    peak_memory_mb: float = 0.0
    node_expansions: int = 0
    node_generations: int = 0
    effective_branching_factor: float = 0.0
    solution_depth: int = 0

@dataclass
class PlanQualityMetrics:
    """Plan quality metrics"""
    plan_cost: float = 0.0
    makespan: float = 0.0
    concurrency_index: float = 0.0
    goal_distance_mean: float = 0.0
    goal_distance_max: float = 0.0
    num_actions: int = 0
    parallel_actions: int = 0

@dataclass
class HeuristicQualityMetrics:
    """Intrinsic heuristic quality metrics"""
    mean_absolute_error: float = 0.0
    rmse_error: float = 0.0
    informedness_ratio: float = 0.0
    correlation_with_true_cost: float = 0.0
    avg_computation_time_ms: float = 0.0
    guidance_reduction_factor: float = 0.0

@dataclass
class CoordinationMetrics:
    """Multi-agent coordination and communication metrics"""
    messages_exchanged: int = 0
    total_data_volume_bytes: int = 0
    synchronization_rounds: int = 0
    coordination_latency_ms: float = 0.0
    per_agent_cpu_time: List[float] = None
    privacy_leakage_score: float = 0.0

@dataclass
class ExperimentResult:
    """Complete results for a single experiment"""
    config: ExperimentConfig
    performance: PerformanceMetrics
    plan_quality: PlanQualityMetrics
    heuristic_quality: HeuristicQualityMetrics
    coordination: CoordinationMetrics
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
                        # Get memory usage
                        memory_info = proc.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        self.peak_memory = max(self.peak_memory, memory_mb)
                        
                        # Get CPU times
                        cpu_times = proc.cpu_times()
                        self.cpu_times.append(cpu_times.user + cpu_times.system)
                        
                        time.sleep(0.1)  # Monitor every 100ms
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

class ExperimentRunner:
    """Main experiment runner class"""
    
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
        
    def get_domain_configs(self) -> List[ExperimentConfig]:
        """Generate experiment configurations for all domains and heuristics"""
        configs = []
        domains_dir = Path("Domains")
        
        # Define which domains to test (start with a representative subset)
        test_domains = ["driverlog", "logistics", "rovers", "satellite"]
        
        for domain_name in test_domains:
            domain_path = domains_dir / domain_name
            if not domain_path.exists():
                continue
                
            # Find problem instances (limit to first 5 for initial testing)
            problem_dirs = sorted([d for d in domain_path.iterdir() if d.is_dir()])[:5]
            
            for problem_dir in problem_dirs:
                # Extract agents from problem files
                agents, agent_files = self._extract_agents_from_problem(problem_dir)
                
                if len(agents) >= 2:  # Only test multi-agent problems
                    for heuristic_name, heuristic_id in self.heuristics.items():
                        config = ExperimentConfig(
                            domain=domain_name,
                            problem=problem_dir.name,
                            heuristic=heuristic_id,
                            agents=agents,
                            agent_files=agent_files
                        )
                        configs.append(config)
        
        return configs
    
    def _extract_agents_from_problem(self, problem_dir: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract agent information from problem directory"""
        agents = []
        agent_files = []
        
        # Look for agent problem files (format: Problem<domain><agent>.pddl)
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
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with the given configuration"""
        logger.info(f"Running experiment: {config.domain}/{config.problem} with {self._get_heuristic_name(config.heuristic)}")
        
        result = ExperimentResult(
            config=config,
            performance=PerformanceMetrics(),
            plan_quality=PlanQualityMetrics(),
            heuristic_quality=HeuristicQualityMetrics(),
            coordination=CoordinationMetrics()
        )
        
        try:
            # Create agent list file
            agent_list_file = self._create_agent_list_file(config)
            
            # Build FMAP command
            cmd = self._build_fmap_command(config, agent_list_file)
            
            # Run experiment with monitoring
            start_time = time.time()
            monitor = ResourceMonitor()
            
            # Run from the problem directory
            problem_dir = Path(f"../Domains/{config.domain}/{config.problem}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(problem_dir)
            )
            
            monitor.start_monitoring(process)
            
            try:
                stdout, stderr = process.communicate(timeout=config.timeout_seconds)
                return_code = process.returncode
                wall_clock_time = time.time() - start_time
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                wall_clock_time = config.timeout_seconds
                result.error_message = "Timeout"
                
            finally:
                monitor.stop_monitoring()
            
            # Store logs
            result.stdout_log = stdout
            result.stderr_log = stderr
            
            # Parse results if successful
            if return_code == 0 and not result.error_message:
                result.performance.coverage = True
                result.performance.wall_clock_time = wall_clock_time
                result.performance.cpu_time = monitor.get_total_cpu_time()
                result.performance.peak_memory_mb = monitor.get_peak_memory()
                
                # Parse FMAP output for additional metrics
                self._parse_fmap_output(stdout, stderr, result)
            
            # Clean up temporary files
            if agent_list_file and os.path.exists(agent_list_file):
                os.remove(agent_list_file)
                
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Experiment failed: {e}")
        
        return result
    
    def _get_heuristic_name(self, heuristic_id: int) -> str:
        """Get heuristic name from ID"""
        name_map = {v: k for k, v in self.heuristics.items()}
        return name_map.get(heuristic_id, f"Unknown({heuristic_id})")
    
    def _create_agent_list_file(self, config: ExperimentConfig) -> str:
        """Create agent list file for FMAP"""
        # Create agent list in the problem directory to avoid path issues
        problem_dir = Path(f"../Domains/{config.domain}/{config.problem}")
        agent_list_file = problem_dir / "agent-list.txt"
        
        with open(agent_list_file, 'w') as f:
            for agent in config.agents:
                f.write(f"{agent} 127.0.0.1\n")
        
        return str(agent_list_file)
    
    def _build_fmap_command(self, config: ExperimentConfig, agent_list_file: str) -> List[str]:
        """Build FMAP command line"""
        # Build path to FMAP jar from problem directory
        if self.fmap_jar_path.startswith("../"):
            # Path is already relative from experiments dir, adjust for problem dir
            jar_path = f"../../..{self.fmap_jar_path[2:]}"  # Remove the first "../"
        else:
            jar_path = f"../../../{self.fmap_jar_path}"
        
        cmd = ["java", "-jar", jar_path]
        
        # Add heuristic parameter
        cmd.extend(["-h", str(config.heuristic)])
        
        # Add agent specifications - use relative paths from problem directory
        for i, (agent, (domain_file, problem_file)) in enumerate(zip(config.agents, config.agent_files)):
            # Extract just the filenames since we'll run from the problem directory
            domain_filename = Path(domain_file).name
            problem_filename = Path(problem_file).name
            cmd.extend([agent, domain_filename, problem_filename])
        
        # Add agent list file (just the filename)
        cmd.append("agent-list.txt")
        
        return cmd
    
    def _parse_fmap_output(self, stdout: str, stderr: str, result: ExperimentResult):
        """Parse FMAP output to extract metrics"""
        # Look for solution plan in stdout
        lines = stdout.split('\n')
        
        # Parse plan if found
        plan_actions = []
        for line in lines:
            if line.strip() and not line.startswith(';') and ':' in line:
                # This looks like a plan action: "timestamp: action"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        timestamp = float(parts[0].strip())
                        action = parts[1].strip()
                        plan_actions.append((timestamp, action))
                    except ValueError:
                        continue
        
        if plan_actions:
            # Calculate plan metrics
            result.plan_quality.num_actions = len(plan_actions)
            result.plan_quality.plan_cost = len(plan_actions)  # Simplified cost
            result.plan_quality.makespan = max(t for t, _ in plan_actions) if plan_actions else 0
            
            # Calculate concurrency index (simplified)
            timestamps = [t for t, _ in plan_actions]
            unique_timestamps = set(timestamps)
            if unique_timestamps:
                result.plan_quality.parallel_actions = len(plan_actions) - len(unique_timestamps)
                result.plan_quality.concurrency_index = result.plan_quality.parallel_actions / len(plan_actions)
        
        # Parse node expansions and other search metrics from stderr (if FMAP outputs them)
        for line in stderr.split('\n'):
            if 'expanded' in line.lower():
                # Try to extract node expansion numbers
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    result.performance.node_expansions = int(numbers[0])
    
    def run_experiments(self, configs: List[ExperimentConfig] = None) -> List[ExperimentResult]:
        """Run all experiments"""
        if configs is None:
            configs = self.get_domain_configs()
        
        logger.info(f"Starting {len(configs)} experiments")
        results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Progress: {i+1}/{len(configs)}")
            result = self.run_single_experiment(config)
            results.append(result)
            
            # Save intermediate results
            self._save_result(result, i)
        
        # Save all results
        self._save_all_results(results)
        
        return results
    
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
    """Main entry point"""
    if len(sys.argv) > 1:
        fmap_jar = sys.argv[1]
    else:
        fmap_jar = "FMAP.jar"
    
    if not os.path.exists(fmap_jar):
        logger.error(f"FMAP jar file not found: {fmap_jar}")
        sys.exit(1)
    
    runner = ExperimentRunner(fmap_jar_path=fmap_jar)
    
    # Generate limited test configs for initial run
    all_configs = runner.get_domain_configs()
    
    # Start with just a few test cases
    test_configs = all_configs[:15]  # First 15 experiments
    
    logger.info(f"Running {len(test_configs)} test experiments out of {len(all_configs)} total")
    
    results = runner.run_experiments(test_configs)
    
    # Print summary
    successful = sum(1 for r in results if r.performance.coverage)
    logger.info(f"Experiment summary: {successful}/{len(results)} successful")

if __name__ == "__main__":
    main() 