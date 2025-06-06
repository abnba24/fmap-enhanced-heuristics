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
    timeout_seconds: int = 300  # 5 minutes for full experiments
    memory_limit_mb: int = 8192  # 8 GB

@dataclass
class SearchMetrics:
    """Search performance metrics from FMAP execution"""
    coverage: bool = False
    wall_clock_time: float = 0.0
    cpu_time: float = 0.0
    peak_memory_mb: float = 0.0
    search_nodes: int = 0  # Count of heuristic evaluations
    dtg_heuristic_values: List[int] = None  # DTG heuristic values during search
    landmark_heuristic_values: List[int] = None  # Landmark heuristic values during search

@dataclass
class PlanMetrics:
    """Plan quality metrics from FMAP solution"""
    plan_found: bool = False
    plan_length: int = 0  # Number of actions
    makespan: float = 0.0  # Highest timestamp in plan
    actions: List[Tuple[float, str]] = None  # List of (timestamp, action) pairs

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
        
    def get_minimal_test_configs(self) -> List[ExperimentConfig]:
        """Generate minimal test configurations for known-working problems"""
        configs = []
        
        # Test only specific known-working 2-agent problems
        test_cases = [
            {
                "domain": "driverlog",
                "problem": "Pfile1", 
                "agents": ["driver1", "driver2"],
                "agent_files": [
                    ("../Domains/driverlog/Pfile1/DomainDriverlog.pddl", "../Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl"),
                    ("../Domains/driverlog/Pfile1/DomainDriverlog.pddl", "../Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl")
                ]
            },
            {
                "domain": "driverlog",
                "problem": "Pfile2", 
                "agents": ["driver1", "driver2"],
                "agent_files": [
                    ("../Domains/driverlog/Pfile2/DomainDriverlog.pddl", "../Domains/driverlog/Pfile2/ProblemDriverlogdriver1.pddl"),
                    ("../Domains/driverlog/Pfile2/DomainDriverlog.pddl", "../Domains/driverlog/Pfile2/ProblemDriverlogdriver2.pddl")
                ]
            },
            {
                "domain": "driverlog",
                "problem": "Pfile3", 
                "agents": ["driver1", "driver2"],
                "agent_files": [
                    ("../Domains/driverlog/Pfile3/DomainDriverlog.pddl", "../Domains/driverlog/Pfile3/ProblemDriverlogdriver1.pddl"),
                    ("../Domains/driverlog/Pfile3/DomainDriverlog.pddl", "../Domains/driverlog/Pfile3/ProblemDriverlogdriver2.pddl")
                ]
            }
        ]
        
        # Test all 5 heuristics on known-working problem
        test_heuristics = {
            "DTG": 1,
            "DTG+Landmarks": 2, 
            "Inc_DTG+Landmarks": 3,
            "Centroids": 4,
            "MCS": 5
        }
        
        for test_case in test_cases:
            for heuristic_name, heuristic_id in test_heuristics.items():
                config = ExperimentConfig(
                    domain=test_case["domain"],
                    problem=test_case["problem"], 
                    heuristic=heuristic_id,
                    agents=test_case["agents"],
                    agent_files=test_case["agent_files"],
                    timeout_seconds=30  # Short timeout for quick testing
                )
                configs.append(config)
        
        logger.info(f"Generated {len(configs)} minimal test configurations")
        return configs

    def get_domain_configs(self) -> List[ExperimentConfig]:
        """Generate experiment configurations for all domains and heuristics"""
        configs = []
        domains_dir = Path("../Domains")  # Adjust path since we're in experiments/
        
        # Test ALL available domains for comprehensive analysis
        test_domains = [
            "driverlog", "logistics", "rovers", "satellite", "elevators", 
            "ma-blocksworld", "openstacks", "woodworking", "zenotravel", "depots"
        ]
        
        agent_count_distribution = defaultdict(list)  # Track problems by agent count
        
        for domain_name in test_domains:
            domain_path = domains_dir / domain_name
            if not domain_path.exists():
                logger.warning(f"Domain not found: {domain_name}")
                continue
                
            logger.info(f"Scanning domain: {domain_name}")
            
            # Find ALL problem instances and categorize by complexity
            problem_dirs = sorted([d for d in domain_path.iterdir() if d.is_dir()])
            
            for problem_dir in problem_dirs:
                # Extract agents from problem files
                agents, agent_files = self._extract_agents_from_problem(problem_dir)
                
                # Test all multi-agent problems to analyze scalability across agent counts
                if len(agents) >= 2:  # Test all multi-agent problems (2-12 agents)
                    agent_count = len(agents)
                    
                    # Categorize problem complexity based on agent count and problem number
                    complexity = self._categorize_problem_complexity(problem_dir.name, agent_count)
                    
                    agent_count_distribution[agent_count].append(f"{domain_name}/{problem_dir.name}")
                    
                    # Test all 5 heuristics for comprehensive analysis
                    for heuristic_name, heuristic_id in self.heuristics.items():
                        config = ExperimentConfig(
                            domain=domain_name,
                            problem=problem_dir.name,
                            heuristic=heuristic_id,
                            agents=agents,
                            agent_files=agent_files
                        )
                        configs.append(config)
        
        # Log distribution of problems by agent count
        logger.info("Agent count distribution:")
        for agent_count in sorted(agent_count_distribution.keys()):
            problems = agent_count_distribution[agent_count]
            logger.info(f"  {agent_count} agents: {len(problems)} problems")
        
        logger.info(f"Total experiment configurations: {len(configs)}")
        return configs
    
    def _extract_agents_from_problem(self, problem_dir: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Extract agent information from problem directory"""
        agents = []
        agent_files = []
        
        # Find domain file
        domain_file = None
        for file in problem_dir.iterdir():
            if file.name.startswith("Domain") and file.suffix == ".pddl":
                domain_file = str(file)
                break
        
        if not domain_file:
            return agents, agent_files
        
        # First try to use agents.txt if it exists (most reliable)
        agents_txt = problem_dir / "agents.txt"
        if agents_txt.exists():
            try:
                with open(agents_txt, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract agent name (format: "agent_name ip_address")
                            parts = line.split()
                            if parts:
                                agent_name = parts[0]
                                agents.append(agent_name)
                                
                                # Find corresponding problem file
                                problem_file = None
                                for file in problem_dir.iterdir():
                                    if file.name.startswith("Problem") and agent_name in file.name and file.suffix == ".pddl":
                                        problem_file = str(file)
                                        break
                                
                                if problem_file:
                                    agent_files.append((domain_file, problem_file))
                                else:
                                    logger.warning(f"Could not find problem file for agent {agent_name} in {problem_dir}")
                
                logger.info(f"Found {len(agents)} agents from agents.txt: {agents}")
                return agents, agent_files
                
            except Exception as e:
                logger.warning(f"Could not read agents.txt in {problem_dir}: {e}")
        
        # Fallback: extract from problem file names
        domain_name = problem_dir.parent.name
        for file in problem_dir.iterdir():
            if file.name.startswith("Problem") and file.suffix == ".pddl":
                filename = file.name
                
                # Extract agent name - try different patterns
                # Pattern: Problem<Domain><Agent>.pddl
                agent_name = filename.replace("Problem", "").replace(".pddl", "")
                
                # Remove domain name variations
                domain_variations = [domain_name, domain_name.capitalize(), domain_name.upper(), domain_name.lower()]
                for domain_var in domain_variations:
                    agent_name = agent_name.replace(domain_var, "")
                
                # Clean up
                agent_name = agent_name.strip("-_")
                
                if agent_name and agent_name not in agents:
                    agents.append(agent_name)
                    agent_files.append((domain_file, str(file)))
        
        logger.info(f"Found {len(agents)} agents from filenames: {agents}")
        return agents, agent_files
    
    def _categorize_problem_complexity(self, problem_name: str, agent_count: int) -> str:
        """Categorize problem complexity based on name and agent count"""
        # Extract problem number if present
        import re
        numbers = re.findall(r'\d+', problem_name)
        problem_num = int(numbers[-1]) if numbers else 0
        
        # Complexity categorization
        if agent_count <= 2:
            if problem_num <= 5:
                return "SMALL"
            elif problem_num <= 15:
                return "MEDIUM"
            else:
                return "LARGE"
        elif agent_count <= 5:
            if problem_num <= 3:
                return "SMALL"
            elif problem_num <= 10:
                return "MEDIUM"
            else:
                return "LARGE"
        else:  # agent_count > 5
            if problem_num <= 2:
                return "SMALL"
            elif problem_num <= 5:
                return "MEDIUM"
            else:
                return "LARGE"
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with the given configuration"""
        logger.info(f"Running experiment: {config.domain}/{config.problem} with {self._get_heuristic_name(config.heuristic)}")
        
        result = ExperimentResult(
            config=config,
            search=SearchMetrics(),
            plan=PlanMetrics()
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
                result.search.coverage = True
                result.search.wall_clock_time = wall_clock_time
                result.search.cpu_time = monitor.get_total_cpu_time()
                result.search.peak_memory_mb = monitor.get_peak_memory()
                
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
        agent_list_file = problem_dir / "agents.txt"
        
        # Use the original format - FMAP will handle internal coordination
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
        cmd.append("agents.txt")
        
        return cmd
    
    def _parse_fmap_output(self, stdout: str, stderr: str, result: ExperimentResult):
        """Parse FMAP output to extract metrics"""
        lines = stdout.split('\n')
        
        # Parse heuristic values during search
        dtg_values = []
        landmark_values = []
        
        for line in lines:
            if line.startswith('; Hdtg = '):
                # Parse heuristic values: "; Hdtg = X, Hlan = Y"
                import re
                dtg_match = re.search(r'Hdtg = (\d+)', line)
                hlan_match = re.search(r'Hlan = (\d+)', line)
                
                if dtg_match:
                    dtg_values.append(int(dtg_match.group(1)))
                if hlan_match:
                    landmark_values.append(int(hlan_match.group(1)))
        
        # Store heuristic values
        if dtg_values:
            result.search.dtg_heuristic_values = dtg_values
        if landmark_values:
            result.search.landmark_heuristic_values = landmark_values
        
        # Count of heuristic evaluations (search nodes)
        result.search.search_nodes = len(dtg_values)
        
        # Parse solution plan
        plan_actions = []
        in_solution_section = False
        
        for line in lines:
            if "; Solution plan - CoDMAP Distributed format" in line:
                in_solution_section = True
                continue
            elif "; Stopping..." in line:
                in_solution_section = False
                break
            elif in_solution_section and line.strip() and not line.startswith(';'):
                # Parse plan action: "timestamp: action"
                if ':' in line and line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        try:
                            timestamp = float(parts[0].strip())
                            action = parts[1].strip()
                            plan_actions.append((timestamp, action))
                        except ValueError:
                            continue
        
        # Set plan metrics
        if plan_actions:
            result.plan.plan_found = True
            result.plan.plan_length = len(plan_actions)
            result.plan.makespan = max(t for t, _ in plan_actions) if plan_actions else 0.0
            result.plan.actions = plan_actions
        
        # If we have a solution, search was successful
        if result.plan.plan_found:
            result.search.coverage = True
    
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

    def get_strategic_configs(self) -> List[ExperimentConfig]:
        """Generate focused experiments for comprehensive heuristic analysis"""
        configs = []
        
        # Get all available configs first
        all_configs = self.get_domain_configs()
        
        # Group by agent count and domain
        from collections import defaultdict
        by_agents_domain = defaultdict(lambda: defaultdict(list))
        
        for config in all_configs:
            agent_count = len(config.agents)
            by_agents_domain[agent_count][config.domain].append(config)
        
        # FOCUSED SELECTION: Focus on 2-agent problems that work reliably
        # Start with working configurations and expand gradually
        target_domains = ["driverlog", "openstacks", "zenotravel"]  # Known working 2-agent domains
        target_agent_counts = [2]  # Start with 2-agent problems that work consistently
        
        logger.info(f"Focused selection: {target_domains} domains with {target_agent_counts} agent counts")
        
        # For each target combination, select representative problems
        for agent_count in target_agent_counts:
            if agent_count not in by_agents_domain:
                logger.warning(f"No problems found for {agent_count} agents")
                continue
                
            logger.info(f"Selecting experiments for {agent_count} agents...")
            
            # Get domains available for this agent count
            domains_for_agents = by_agents_domain[agent_count]
            
            # Filter to only target domains
            for domain in target_domains:
                if domain not in domains_for_agents:
                    logger.warning(f"Domain {domain} not available for {agent_count} agents")
                    continue
                    
                domain_configs = domains_for_agents[domain]
                logger.info(f"  Found {len(domain_configs)} configs for {domain} with {agent_count} agents")
                
                # Group by problem complexity (based on problem number)
                complexity_groups = {"SMALL": [], "MEDIUM": [], "LARGE": []}
                
                for config in domain_configs:
                    complexity = self._categorize_problem_complexity(config.problem, agent_count)
                    complexity_groups[complexity].append(config)
                
                # Select one representative problem from each complexity level
                for complexity, complexity_configs in complexity_groups.items():
                    if not complexity_configs:
                        continue
                    
                    # Sort by problem number to get consistent selection
                    sorted_configs = sorted(complexity_configs, key=lambda x: x.problem)
                    
                    # TEST ALL 5 HEURISTICS equally on every selected problem
                    heuristics_to_test = [1, 2, 3, 4, 5]  # All heuristics for equal comparison
                    
                    # Select one problem per complexity and test ALL heuristics
                    representative_config = sorted_configs[0]
                    logger.info(f"    Selected {domain}/{representative_config.problem} ({complexity}) for all heuristics")
                    
                    for heuristic_id in heuristics_to_test:
                        # Find the config with this heuristic
                        matching_configs = [c for c in sorted_configs if c.heuristic == heuristic_id]
                        if matching_configs:
                            configs.append(matching_configs[0])
        
        logger.info(f"Focused selection: {len(configs)} experiments across {len(target_agent_counts)} agent count levels")
        
        # Show distribution
        agent_distribution = defaultdict(int)
        heuristic_distribution = defaultdict(int)
        domain_distribution = defaultdict(int)
        
        for config in configs:
            agent_distribution[len(config.agents)] += 1
            heuristic_distribution[self._get_heuristic_name(config.heuristic)] += 1
            domain_distribution[config.domain] += 1
        
        logger.info("Focused distribution:")
        logger.info(f"  Agent counts: {dict(sorted(agent_distribution.items()))}")
        logger.info(f"  Heuristics: {dict(heuristic_distribution)}")
        logger.info(f"  Domains: {dict(domain_distribution)}")
        
        return configs
    
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
        
        # Filter to only working problems (include all, including 8-agent problem)
        test_problems = [
            p for p in working_problems 
            if p.get('test_result', {}).get('status') in ['tested_working', 'timeout_but_working', 'assumed_working']
        ]
        
        # Sort by agent count so smaller problems run first, larger problems last
        test_problems = sorted(test_problems, key=lambda x: x.get('agent_count', 0))
        
        logger.info(f"Loaded {len(test_problems)} working problems from finalized sample set")
        
        # All heuristics to test on each problem
        heuristics_to_test = [1, 2, 3, 4, 5]  # DTG, DTG+Landmarks, Inc_DTG+Landmarks, Centroids, MCS
        
        for problem_data in test_problems:
            domain = problem_data['domain']
            problem = problem_data['problem']
            agent_count = problem_data['agent_count']
            
            problem_dir = Path(f"../Domains/{domain}/{problem}")
            
            if not problem_dir.exists():
                logger.warning(f"Problem directory not found: {problem_dir}")
                continue
            
            # Extract agent information using the proven method
            agents, agent_files = self._extract_agents_from_problem(problem_dir)
            
            if not agents:
                logger.warning(f"No agents found for {domain}/{problem}")
                continue
            
            # Set 30-minute timeout for all experiments as requested
            timeout = 1800  # 30 minutes for all problems
            
            # Create config for each heuristic
            for heuristic_id in heuristics_to_test:
                config = ExperimentConfig(
                    domain=domain,
                    problem=problem,
                    heuristic=heuristic_id,
                    agents=agents,
                    agent_files=agent_files,
                    timeout_seconds=timeout
                )
                configs.append(config)
        
        # Sort configs by agent count, then by domain/problem to ensure smallest problems run first
        configs = sorted(configs, key=lambda x: (len(x.agents), x.domain, x.problem, x.heuristic))
                
        logger.info(f"Generated {len(configs)} configurations from finalized working problems")
        
        # Show distribution
        agent_distribution = defaultdict(int)
        heuristic_distribution = defaultdict(int)
        domain_distribution = defaultdict(int)
        
        for config in configs:
            agent_distribution[len(config.agents)] += 1
            heuristic_distribution[self._get_heuristic_name(config.heuristic)] += 1
            domain_distribution[config.domain] += 1
        
        logger.info("Finalized sample distribution:")
        logger.info(f"  Agent counts: {dict(sorted(agent_distribution.items()))}")
        logger.info(f"  Heuristics: {dict(heuristic_distribution)}")
        logger.info(f"  Domains: {dict(domain_distribution)}")
        
        return configs

class ExperimentAnalyzer:
    """Analyzes and visualizes experiment results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
    
    def load_results(self) -> pd.DataFrame:
        """Load all experiment results into a DataFrame"""
        all_results_file = self.results_dir / "all_results.json"
        
        if not all_results_file.exists():
            logger.error(f"Results file not found: {all_results_file}")
            return pd.DataFrame()
        
        with open(all_results_file, 'r') as f:
            results_data = json.load(f)
        
        # Flatten the nested structure
        flattened_data = []
        for result in results_data:
            row = {
                'domain': result['config']['domain'],
                'problem': result['config']['problem'],
                'heuristic_id': result['config']['heuristic'],
                'agent_count': len(result['config']['agents']),
                'coverage': result['search']['coverage'],
                'wall_clock_time': result['search']['wall_clock_time'],
                'cpu_time': result['search']['cpu_time'],
                'peak_memory_mb': result['search']['peak_memory_mb'],
                'search_nodes': result['search']['search_nodes'],
                'plan_found': result['plan']['plan_found'],
                'plan_length': result['plan']['plan_length'],
                'makespan': result['plan']['makespan'],
                'error_message': result.get('error_message', ''),
            }
            
            # Add heuristic name
            heuristic_names = {1: "DTG", 2: "DTG+Landmarks", 3: "Inc_DTG+Landmarks", 4: "Centroids", 5: "MCS"}
            row['heuristic_name'] = heuristic_names.get(row['heuristic_id'], f"H{row['heuristic_id']}")
            
            # Add complexity category
            row['complexity'] = self._categorize_complexity(row['problem'], row['agent_count'])
            
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        logger.info(f"Loaded {len(df)} experiment results")
        return df
    
    def _categorize_complexity(self, problem_name: str, agent_count: int) -> str:
        """Categorize problem complexity"""
        import re
        numbers = re.findall(r'\d+', problem_name)
        problem_num = int(numbers[-1]) if numbers else 0
        
        if agent_count <= 2:
            if problem_num <= 5:
                return "SMALL"
            elif problem_num <= 15:
                return "MEDIUM"
            else:
                return "LARGE"
        elif agent_count <= 5:
            if problem_num <= 3:
                return "SMALL"
            elif problem_num <= 10:
                return "MEDIUM"
            else:
                return "LARGE"
        else:
            if problem_num <= 2:
                return "SMALL"
            elif problem_num <= 5:
                return "MEDIUM"
            else:
                return "LARGE"
    
    def generate_comprehensive_analysis(self, df: pd.DataFrame):
        """Generate comprehensive analysis and visualizations"""
        logger.info("Generating comprehensive analysis...")
        
        # Filter successful experiments
        successful_df = df[df['coverage'] == True].copy()
        
        if len(successful_df) == 0:
            logger.warning("No successful experiments to analyze")
            return
        
        # Generate all visualizations
        self.plot_heuristic_performance_by_domain(successful_df)
        self.plot_heuristic_performance_by_agent_count(successful_df)
        self.plot_scalability_analysis(successful_df)
        self.plot_performance_matrices(successful_df)
        self.plot_heuristic_comparison_analysis(successful_df)
        self.generate_statistical_summary(successful_df)
        
        logger.info(f"Analysis complete. Plots saved to {self.plots_dir}")
    
    def plot_heuristic_performance_by_domain(self, df: pd.DataFrame):
        """Plot heuristic performance breakdown by domain"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Success rate by domain
        success_by_domain = df.groupby(['domain', 'heuristic_name']).size().unstack(fill_value=0)
        success_rate = success_by_domain.div(success_by_domain.sum(axis=1), axis=0)
        
        sns.heatmap(success_rate, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,0])
        axes[0,0].set_title('Success Rate by Domain and Heuristic')
        axes[0,0].set_xlabel('Heuristic')
        axes[0,0].set_ylabel('Domain')
        
        # Average execution time by domain
        time_by_domain = df.groupby(['domain', 'heuristic_name'])['wall_clock_time'].mean().unstack()
        sns.heatmap(time_by_domain, annot=True, fmt='.2f', cmap='viridis_r', ax=axes[0,1])
        axes[0,1].set_title('Average Execution Time by Domain (seconds)')
        axes[0,1].set_xlabel('Heuristic')
        axes[0,1].set_ylabel('Domain')
        
        # Plan quality by domain
        quality_by_domain = df.groupby(['domain', 'heuristic_name'])['plan_length'].mean().unstack()
        sns.heatmap(quality_by_domain, annot=True, fmt='.1f', cmap='plasma_r', ax=axes[1,0])
        axes[1,0].set_title('Average Plan Length by Domain')
        axes[1,0].set_xlabel('Heuristic')
        axes[1,0].set_ylabel('Domain')
        
        # Memory usage by domain
        memory_by_domain = df.groupby(['domain', 'heuristic_name'])['peak_memory_mb'].mean().unstack()
        sns.heatmap(memory_by_domain, annot=True, fmt='.0f', cmap='Reds', ax=axes[1,1])
        axes[1,1].set_title('Average Peak Memory by Domain (MB)')
        axes[1,1].set_xlabel('Heuristic')
        axes[1,1].set_ylabel('Domain')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heuristic_performance_by_domain.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_heuristic_performance_by_agent_count(self, df: pd.DataFrame):
        """Plot heuristic performance breakdown by agent count"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Execution time scaling
        for heuristic in df['heuristic_name'].unique():
            heuristic_data = df[df['heuristic_name'] == heuristic]
            agent_times = heuristic_data.groupby('agent_count')['wall_clock_time'].agg(['mean', 'std'])
            
            axes[0,0].errorbar(agent_times.index, agent_times['mean'], 
                              yerr=agent_times['std'], label=heuristic, marker='o')
        
        axes[0,0].set_xlabel('Number of Agents')
        axes[0,0].set_ylabel('Execution Time (seconds)')
        axes[0,0].set_title('Execution Time Scaling by Agent Count')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # Memory scaling
        for heuristic in df['heuristic_name'].unique():
            heuristic_data = df[df['heuristic_name'] == heuristic]
            agent_memory = heuristic_data.groupby('agent_count')['peak_memory_mb'].agg(['mean', 'std'])
            
            axes[0,1].errorbar(agent_memory.index, agent_memory['mean'],
                              yerr=agent_memory['std'], label=heuristic, marker='s')
        
        axes[0,1].set_xlabel('Number of Agents')
        axes[0,1].set_ylabel('Peak Memory (MB)')
        axes[0,1].set_title('Memory Usage Scaling by Agent Count')
        axes[0,1].legend()
        
        # Plan quality scaling
        for heuristic in df['heuristic_name'].unique():
            heuristic_data = df[df['heuristic_name'] == heuristic]
            agent_quality = heuristic_data.groupby('agent_count')['plan_length'].agg(['mean', 'std'])
            
            axes[1,0].errorbar(agent_quality.index, agent_quality['mean'],
                              yerr=agent_quality['std'], label=heuristic, marker='^')
        
        axes[1,0].set_xlabel('Number of Agents')
        axes[1,0].set_ylabel('Average Plan Length')
        axes[1,0].set_title('Plan Quality by Agent Count')
        axes[1,0].legend()
        
        # Success rate by agent count
        success_by_agents = df.groupby(['agent_count', 'heuristic_name']).size().unstack(fill_value=0)
        success_by_agents.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_xlabel('Number of Agents')
        axes[1,1].set_ylabel('Number of Successful Experiments')
        axes[1,1].set_title('Success Count by Agent Count')
        axes[1,1].legend(title='Heuristic', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heuristic_performance_by_agent_count.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scalability_analysis(self, df: pd.DataFrame):
        """Generate detailed scalability analysis"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Performance vs Complexity
        complexity_order = ['SMALL', 'MEDIUM', 'LARGE']
        sns.boxplot(data=df, x='complexity', y='wall_clock_time', hue='heuristic_name', 
                   order=complexity_order, ax=axes[0])
        axes[0].set_title('Performance by Problem Complexity')
        axes[0].set_ylabel('Execution Time (seconds)')
        axes[0].set_yscale('log')
        
        # Heuristic Comparison by Agent Count
        heuristic_perf = df.groupby(['agent_count', 'heuristic_name'])['wall_clock_time'].mean().unstack()
        
        if not heuristic_perf.empty:
            # Plot all heuristics with different styles
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            styles = ['-', '--', '-.', ':', '-']
            
            for i, col in enumerate(heuristic_perf.columns):
                color = colors[i % len(colors)]
                style = styles[i % len(styles)]
                axes[1].plot(heuristic_perf.index, heuristic_perf[col], 
                           color=color, linestyle=style, linewidth=2, markersize=6, 
                           marker='o', label=col)
            
            axes[1].set_xlabel('Number of Agents')
            axes[1].set_ylabel('Average Execution Time (s)')
            axes[1].set_title('Heuristic Performance Comparison by Agent Count')
            axes[1].legend()
            axes[1].set_yscale('log')
        
        # Domain difficulty ranking
        domain_difficulty = df.groupby('domain')['wall_clock_time'].agg(['mean', 'std', 'count'])
        domain_difficulty = domain_difficulty.sort_values('mean')
        
        x_pos = range(len(domain_difficulty))
        axes[2].barh(x_pos, domain_difficulty['mean'], xerr=domain_difficulty['std'])
        axes[2].set_yticks(x_pos)
        axes[2].set_yticklabels(domain_difficulty.index)
        axes[2].set_xlabel('Average Execution Time (seconds)')
        axes[2].set_title('Domain Difficulty Ranking')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_matrices(self, df: pd.DataFrame):
        """Generate performance correlation matrices"""
        numeric_cols = ['wall_clock_time', 'cpu_time', 'peak_memory_mb', 'search_nodes', 
                       'plan_length', 'makespan', 'agent_count']
        
        # Filter numeric columns that exist and have data
        available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient numeric data for correlation analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall correlation matrix
        corr_matrix = df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Performance Metrics Correlation Matrix')
        
        # Best performing heuristic analysis
        best_heuristic = df.groupby('heuristic_name')['wall_clock_time'].mean().idxmin()
        df_best = df[df['heuristic_name'] == best_heuristic]
        
        if len(df_best) > 5:  # Need sufficient data points
            best_corr = df_best[available_cols].corr()
            sns.heatmap(best_corr, annot=True, cmap='coolwarm', center=0, ax=axes[1])
            axes[1].set_title(f'{best_heuristic} Heuristic - Metrics Correlation')
        else:
            axes[1].text(0.5, 0.5, 'Insufficient data\nfor detailed correlation analysis', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Detailed Heuristic Analysis')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_heuristic_comparison_analysis(self, df: pd.DataFrame):
        """Generate comprehensive heuristic comparison analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heuristic success rates by domain
        if len(df) > 0:
            success_rates = df.groupby(['heuristic_name', 'domain']).size().unstack(fill_value=0)
            success_rates.plot(kind='bar', ax=axes[0,0])
            axes[0,0].set_title('Success Count by Heuristic and Domain')
            axes[0,0].set_ylabel('Number of Successful Runs')
            axes[0,0].set_xlabel('Heuristic')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Performance ranking by domain
        avg_performance = df.groupby(['domain', 'heuristic_name'])['wall_clock_time'].mean().unstack()
        
        if not avg_performance.empty:
            # Normalize each domain's results for comparison
            normalized_perf = avg_performance.div(avg_performance.min(axis=1), axis=0)
            sns.heatmap(normalized_perf, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0,1])
            axes[0,1].set_title('Relative Performance by Domain\n(Lower is Better, 1.0 = Best)')
            axes[0,1].set_xlabel('Heuristic')
            axes[0,1].set_ylabel('Domain')
        
        # Overall heuristic performance distribution
        if len(df) > 0:
            heuristics = df['heuristic_name'].unique()
            for heuristic in heuristics:
                heuristic_data = df[df['heuristic_name'] == heuristic]['wall_clock_time']
                axes[1,0].hist(heuristic_data, bins=20, alpha=0.7, label=heuristic)
            
            axes[1,0].set_xlabel('Execution Time (seconds)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Execution Time Distribution by Heuristic')
            axes[1,0].legend()
            axes[1,0].set_yscale('log')
        
        # Heuristic performance summary statistics
        if len(df) > 0:
            performance_stats = df.groupby('heuristic_name')['wall_clock_time'].agg(['mean', 'std', 'count'])
            performance_stats = performance_stats.sort_values('mean')
            
            x_pos = range(len(performance_stats))
            axes[1,1].bar(x_pos, performance_stats['mean'], yerr=performance_stats['std'], 
                         capsize=5, alpha=0.7)
            axes[1,1].set_xticks(x_pos)
            axes[1,1].set_xticklabels(performance_stats.index, rotation=45)
            axes[1,1].set_ylabel('Average Execution Time (seconds)')
            axes[1,1].set_title('Overall Heuristic Performance Ranking')
            
            # Add count annotations
            for i, (idx, row) in enumerate(performance_stats.iterrows()):
                axes[1,1].text(i, row['mean'] + row['std'], f'n={row["count"]}', 
                              ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'heuristic_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_statistical_summary(self, df: pd.DataFrame):
        """Generate comprehensive statistical summary"""
        summary_file = self.results_dir / 'statistical_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("FMAP HEURISTIC PERFORMANCE ANALYSIS - STATISTICAL SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL EXPERIMENT STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total successful experiments: {len(df)}\n")
            f.write(f"Domains tested: {df['domain'].nunique()}\n")
            f.write(f"Heuristics tested: {df['heuristic_name'].nunique()}\n")
            f.write(f"Agent count range: {df['agent_count'].min()} - {df['agent_count'].max()}\n")
            f.write(f"Problems tested: {df['problem'].nunique()}\n\n")
            
            # Performance by heuristic
            f.write("PERFORMANCE BY HEURISTIC\n")
            f.write("-" * 25 + "\n")
            heuristic_stats = df.groupby('heuristic_name').agg({
                'wall_clock_time': ['count', 'mean', 'std', 'min', 'max'],
                'peak_memory_mb': ['mean', 'std'],
                'plan_length': ['mean', 'std']
            }).round(3)
            f.write(str(heuristic_stats))
            f.write("\n\n")
            
            # Domain analysis
            f.write("DOMAIN ANALYSIS\n")
            f.write("-" * 15 + "\n")
            domain_stats = df.groupby('domain').agg({
                'wall_clock_time': ['count', 'mean', 'std'],
                'agent_count': ['min', 'max', 'mean']
            }).round(3)
            f.write(str(domain_stats))
            f.write("\n\n")
            
            # Best performing heuristic analysis
            best_heuristic = df.groupby('heuristic_name')['wall_clock_time'].mean().idxmin()
            best_df = df[df['heuristic_name'] == best_heuristic]
            if len(best_df) > 0:
                f.write(f"BEST PERFORMING HEURISTIC: {best_heuristic.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Experiments: {len(best_df)}\n")
                f.write(f"Average execution time: {best_df['wall_clock_time'].mean():.3f} seconds\n")
                f.write(f"Std deviation: {best_df['wall_clock_time'].std():.3f} seconds\n")
                f.write(f"Average plan length: {best_df['plan_length'].mean():.3f}\n")
                f.write(f"Average memory usage: {best_df['peak_memory_mb'].mean():.1f} MB\n")
                f.write(f"Domains covered: {best_df['domain'].nunique()}\n")
                f.write(f"Agent count range: {best_df['agent_count'].min()} - {best_df['agent_count'].max()}\n\n")
            
            # Heuristic ranking
            f.write("HEURISTIC PERFORMANCE RANKING\n")
            f.write("-" * 30 + "\n")
            ranking = df.groupby('heuristic_name').agg({
                'wall_clock_time': ['mean', 'count']
            }).round(3)
            ranking.columns = ['avg_time', 'experiments']
            ranking = ranking.sort_values('avg_time')
            for i, (heuristic, row) in enumerate(ranking.iterrows(), 1):
                f.write(f"{i}. {heuristic}: {row['avg_time']:.3f}s (n={row['experiments']})\n")
        
        logger.info(f"Statistical summary saved to {summary_file}")

def main():
    """Main entry point"""
    # Check if we should run experiments or just analyze existing results
    if "--analyze-only" in sys.argv:
        logger.info("Running analysis on existing results...")
        analyzer = ExperimentAnalyzer()
        df = analyzer.load_results()
        if not df.empty:
            analyzer.generate_comprehensive_analysis(df)
        else:
            logger.error("No results found to analyze")
        return
    
    # Get FMAP jar path
    fmap_jar = "../FMAP.jar"  # Default path since we're in experiments/
    for i, arg in enumerate(sys.argv[1:]):
        if not arg.startswith("--") and arg.endswith(".jar"):
            fmap_jar = arg
            break
    
    if not os.path.exists(fmap_jar):
        logger.error(f"FMAP jar file not found: {fmap_jar}")
        sys.exit(1)
    
    runner = ExperimentRunner(fmap_jar_path=fmap_jar)
    
    # Generate experiment configurations
    if "--minimal" in sys.argv:
        # Run minimal test with known-working configurations
        test_configs = runner.get_minimal_test_configs()
        logger.info(f"Running MINIMAL {len(test_configs)} test experiments")
    elif "--strategic" in sys.argv:
        # Run strategically selected experiments for comprehensive analysis
        test_configs = runner.get_strategic_configs()
        logger.info(f"Running STRATEGIC {len(test_configs)} experiments")
    elif "--finalized" in sys.argv:
        # Run experiments on our finalized working sample set
        test_configs = runner.get_finalized_sample_configs()
        logger.info(f"Running FINALIZED SAMPLE {len(test_configs)} experiments")
    elif "--full" in sys.argv:
        # Run all experiments
        all_configs = runner.get_domain_configs()
        test_configs = all_configs
        logger.info(f"Running ALL {len(test_configs)} experiments")
    else:
        # Run subset for testing (stratified sampling)
        all_configs = runner.get_domain_configs()
        test_configs = []
        configs_by_domain = defaultdict(list)
        for config in all_configs:
            configs_by_domain[config.domain].append(config)
        
        # Sample from each domain - stratified sampling for balanced representation
        for domain, domain_configs in configs_by_domain.items():
            # Take 10-15 configs per domain for comprehensive coverage
            sample_size = min(15, len(domain_configs))
            # Sort by agent count and problem to get variety across the complexity spectrum
            sorted_configs = sorted(domain_configs, key=lambda x: (len(x.agents), x.problem, x.heuristic))
            step = max(1, len(sorted_configs) // sample_size)
            sampled = sorted_configs[::step][:sample_size]
            test_configs.extend(sampled)
        
        logger.info(f"Running {len(test_configs)} test experiments out of {len(all_configs)} total")
    
    # Run experiments
    results = runner.run_experiments(test_configs)
    
    # Print summary
    successful = sum(1 for r in results if r.search.coverage)
    logger.info(f"Experiment summary: {successful}/{len(results)} successful")
    
    # Generate analysis if we have results
    if successful > 0:
        logger.info("Generating analysis...")
        analyzer = ExperimentAnalyzer()
        df = analyzer.load_results()
        if not df.empty:
            analyzer.generate_comprehensive_analysis(df)

if __name__ == "__main__":
    main() 