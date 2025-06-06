#!/usr/bin/env python3
"""
FMAP Domain Testing and Holistic Experiment Automation

This script systematically tests all FMAP domains to identify working configurations,
populates agent files, and runs comprehensive experiments spanning 2-10 agents.
"""

import subprocess
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DomainTester:
    """Tests domain compatibility and populates agent files"""
    
    def __init__(self, domains_dir: str = "Domains", fmap_jar: str = "FMAP.jar", timeout: int = 20):
        self.domains_dir = Path(domains_dir)
        self.fmap_jar = fmap_jar
        self.timeout = timeout
        self.working_problems = []
        self.failed_problems = []
    
    def populate_agent_files(self):
        """Auto-populate agents.txt files for all domains"""
        logger.info("Populating agents.txt files across all domains...")
        created_count = 0
        
        for domain_dir in self.domains_dir.iterdir():
            if domain_dir.is_dir() and not domain_dir.name.startswith('.'):
                logger.info(f"Processing domain: {domain_dir.name}")
                
                for problem_dir in domain_dir.iterdir():
                    if problem_dir.is_dir():
                        agents = self._extract_agents_from_directory(problem_dir)
                        
                        if len(agents) >= 2:
                            agents_file = problem_dir / "agents.txt"
                            with open(agents_file, 'w') as f:
                                for agent in sorted(agents):
                                    f.write(f"{agent} 127.0.0.1\n")
                            
                            logger.info(f"  Created {problem_dir.name}/agents.txt with {len(agents)} agents: {agents}")
                            created_count += 1
                        elif len(agents) == 1:
                            logger.info(f"  Skipped {problem_dir.name} (only 1 agent)")
                        else:
                            logger.info(f"  Skipped {problem_dir.name} (no agents found)")
        
        logger.info(f"Total agents.txt files created: {created_count}")
    
    def _extract_agents_from_directory(self, problem_dir: Path) -> List[str]:
        """Extract agent names from problem files in directory"""
        agents = []
        domain_name = problem_dir.parent.name
        
        for file in problem_dir.iterdir():
            if file.name.startswith("Problem") and file.suffix == ".pddl":
                filename = file.name.replace("Problem", "").replace(".pddl", "")
                
                # Remove domain name variations
                for variant in [domain_name, domain_name.capitalize(), domain_name.upper()]:
                    filename = filename.replace(variant, "")
                
                # Clean up agent name
                agent_name = filename.strip("-_")
                
                if agent_name and agent_name not in agents:
                    agents.append(agent_name)
        
        return agents
    
    def test_all_domains(self):
        """Test all domains systematically"""
        logger.info("=== FMAP DOMAIN COMPATIBILITY TESTING ===")
        
        working_domains = []
        failed_domains = []
        
        domains_to_test = [
            "driverlog", "logistics", "rovers", "satellite", "elevators", 
            "ma-blocksworld", "openstacks", "woodworking", "zenotravel", "depots"
        ]
        
        for domain in domains_to_test:
            domain_path = self.domains_dir / domain
            if not domain_path.exists():
                logger.warning(f"Domain {domain} not found, skipping...")
                continue
            
            logger.info(f"=== TESTING DOMAIN: {domain} ===")
            success_count = 0
            test_count = 0
            
            # Test up to 8 problems per domain
            problem_dirs = [d for d in domain_path.iterdir() if d.is_dir()]
            problem_dirs = sorted(problem_dirs)[:8]  # Limit to first 8 for efficiency
            
            for problem_dir in problem_dirs:
                result = self._test_domain_problem(domain, problem_dir.name)
                
                if result != "skip":
                    test_count += 1
                    if result == "success":
                        success_count += 1
                
                # Early success detection - if we get 2 successes, this domain is working
                if success_count >= 2:
                    break
            
            # Classify domain
            if success_count > 0:
                working_domains.append(domain)
                logger.info(f"  RESULT: WORKING ({success_count}/{test_count} successful)")
            else:
                failed_domains.append(domain)
                logger.info(f"  RESULT: FAILED ({success_count}/{test_count} successful)")
        
        # Generate summary
        logger.info("=== DOMAIN TESTING SUMMARY ===")
        logger.info(f"Working domains ({len(working_domains)}): {working_domains}")
        logger.info(f"Failed domains ({len(failed_domains)}): {failed_domains}")
        
        # Analyze agent distribution
        agent_count_dist = defaultdict(int)
        for entry in self.working_problems:
            agent_count = entry['agent_count']
            agent_count_dist[agent_count] += 1
        
        logger.info("Working problems by agent count:")
        for count in sorted(agent_count_dist.keys()):
            logger.info(f"  {count} agents: {agent_count_dist[count]} problems")
        
        return working_domains, self.working_problems
    
    def _test_domain_problem(self, domain: str, problem: str) -> str:
        """Test a single domain/problem combination"""
        domain_path = self.domains_dir / domain / problem
        
        if not domain_path.exists():
            return "skip"
        
        # Change to problem directory
        original_cwd = os.getcwd()
        os.chdir(domain_path)
        
        try:
            # Check prerequisites
            if not (domain_path / "agents.txt").exists():
                return "skip"
            
            with open("agents.txt") as f:
                agent_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if len(agent_lines) < 2:
                return "skip"
            
            # Find domain file
            domain_file = None
            for f in domain_path.iterdir():
                if f.name.startswith("Domain") and f.suffix == ".pddl":
                    domain_file = f.name
                    break
            
            if not domain_file:
                return "skip"
            
            # Build FMAP command
            cmd = ["timeout", str(self.timeout), "java", "-jar", "../../../" + self.fmap_jar, "-h", "1"]
            valid_agents = 0
            
            for line in agent_lines:
                agent_name = line.split()[0]
                
                # Find problem file for this agent
                problem_file = None
                for f in domain_path.iterdir():
                    if f.name.startswith("Problem") and agent_name in f.name and f.suffix == ".pddl":
                        problem_file = f.name
                        break
                
                if problem_file:
                    cmd.extend([agent_name, domain_file, problem_file])
                    valid_agents += 1
            
            if valid_agents < 2:
                return "skip"
            
            cmd.append("agents.txt")
            
            # Execute test
            logger.info(f"TESTING: {domain}/{problem} ({valid_agents} agents)")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout + 5)
                
                if result.returncode == 0 and "Solution plan" in result.stdout:
                    logger.info(f"  ✓ SUCCESS: Found solution")
                    self.working_problems.append({
                        'domain': domain,
                        'problem': problem,
                        'agent_count': valid_agents,
                        'agents': [line.split()[0] for line in agent_lines]
                    })
                    return "success"
                elif "NullPointerException" in result.stderr or "NullPointerException" in result.stdout:
                    logger.info(f"  ✗ FAILED: NullPointerException")
                    return "fail"
                elif "Address already in use" in result.stderr:
                    logger.info(f"  ✗ FAILED: Port conflict")
                    return "fail"
                else:
                    logger.info(f"  ✗ FAILED: No solution found")
                    return "fail"
                    
            except subprocess.TimeoutExpired:
                logger.info(f"  ✗ TIMEOUT: No solution in {self.timeout}s")
                return "fail"
            
        finally:
            os.chdir(original_cwd)
    
    def save_working_problems(self, filename: str = "working_problems.json"):
        """Save working problems to file"""
        with open(filename, 'w') as f:
            json.dump(self.working_problems, f, indent=2)
        logger.info(f"Saved {len(self.working_problems)} working problems to {filename}")

def run_strategic_experiments(working_problems: List[Dict]):
    """Run strategic experiments based on working problems"""
    logger.info("=== RUNNING STRATEGIC EXPERIMENTS ===")
    
    # Clear previous results
    results_dir = Path("experiments/results")
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to experiments directory
    original_cwd = os.getcwd()
    os.chdir("experiments")
    
    try:
        # Update experiment runner with working problems
        logger.info("Updating experiment runner with validated working problems...")
        
        # Create strategic config based on working problems
        config_code = generate_strategic_config_code(working_problems)
        
        # Append to experiment runner
        with open("experiment_runner.py", "a") as f:
            f.write("\n" + config_code)
        
        # Run strategic experiments
        logger.info("Executing strategic experiments...")
        result = subprocess.run(["python3", "experiment_runner.py", "--strategic"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Strategic experiments completed successfully")
            
            # Generate analysis
            logger.info("Generating analysis...")
            analysis_result = subprocess.run(["python3", "experiment_runner.py", "--analyze-only"],
                                           capture_output=True, text=True)
            
            if analysis_result.returncode == 0:
                logger.info("Analysis completed successfully")
            else:
                logger.error(f"Analysis failed: {analysis_result.stderr}")
        else:
            logger.error(f"Strategic experiments failed: {result.stderr}")
    
    finally:
        os.chdir(original_cwd)

def generate_strategic_config_code(working_problems: List[Dict]) -> str:
    """Generate Python code for strategic experiment configuration"""
    
    # Group problems by agent count
    by_agent_count = defaultdict(list)
    for problem in working_problems:
        by_agent_count[problem['agent_count']].append(problem)
    
    code = """
    def get_validated_strategic_configs(self) -> List[ExperimentConfig]:
        \"\"\"Generate strategic configs using only validated working domains\"\"\"
        configs = []
        
        # Working problems from domain testing
        working_problems = {
    """
    
    for agent_count, problems in by_agent_count.items():
        code += f"            {agent_count}: {repr(problems)},\n"
    
    code += """        }
        
        # Target distribution across agent counts
        target_per_agent_count = {2: 10, 3: 8, 4: 6, 5: 4}
        
        for agent_count, target_count in target_per_agent_count.items():
            if agent_count in working_problems:
                problems = working_problems[agent_count][:target_count]
                
                for problem_info in problems:
                    domain = problem_info['domain']
                    problem = problem_info['problem']
                    agents = problem_info['agents']
                    
                    # Build agent files list
                    agent_files = []
                    for agent in agents:
                        domain_file = f"../Domains/{domain}/{problem}/Domain{domain.capitalize()}.pddl"
                        problem_file = f"../Domains/{domain}/{problem}/Problem{domain.capitalize()}{agent}.pddl"
                        agent_files.append((domain_file, problem_file))
                    
                    # Test all 5 heuristics
                    for heuristic_id in [1, 2, 3, 4, 5]:
                        config = ExperimentConfig(
                            domain=domain,
                            problem=problem,
                            heuristic=heuristic_id,
                            agents=agents,
                            agent_files=agent_files,
                            timeout_seconds=60
                        )
                        configs.append(config)
        
        logger.info(f"Generated {len(configs)} validated strategic configs")
        return configs
    """
    
    return code

def main():
    """Main automation pipeline"""
    logger.info("=== FMAP HOLISTIC EXPERIMENT AUTOMATION ===")
    
    # Phase 1: Domain Testing
    logger.info("Phase 1: Domain Testing and Agent File Population")
    
    tester = DomainTester()
    
    # Populate agent files
    tester.populate_agent_files()
    
    # Test domain compatibility
    working_domains, working_problems = tester.test_all_domains()
    
    # Save working problems
    tester.save_working_problems()
    
    if not working_problems:
        logger.error("No working problems found! Cannot proceed with experiments.")
        return
    
    # Phase 2: Strategic Experiments
    logger.info("Phase 2: Strategic Experiment Execution")
    run_strategic_experiments(working_problems)
    
    # Phase 3: Results Summary
    logger.info("=== FINAL RESULTS SUMMARY ===")
    
    results_file = Path("experiments/results/all_results.json")
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        successful = sum(1 for r in results if r['search']['coverage'])
        total = len(results)
        
        logger.info(f"Total experiments: {total}")
        logger.info(f"Successful: {successful} ({100*successful/total:.1f}%)")
        
        # Generate distribution analysis
        agent_dist = defaultdict(int)
        heuristic_dist = defaultdict(int)
        
        for r in results:
            if r['search']['coverage']:
                agent_dist[len(r['config']['agents'])] += 1
                heuristic_dist[r['config']['heuristic']] += 1
        
        logger.info(f"Agent count coverage: {dict(sorted(agent_dist.items()))}")
        logger.info(f"Heuristic coverage: {dict(sorted(heuristic_dist.items()))}")
        
        logger.info("Results available in:")
        logger.info("- experiments/results/statistical_summary.txt")
        logger.info("- experiments/results/plots/")
        logger.info("- experiments/results/all_results.json")
    else:
        logger.error("No results file found")

if __name__ == "__main__":
    main() 