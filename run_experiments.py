#!/usr/bin/env python3
import os
import subprocess
import json
import time
import re

class HeuristicExperimentRunner:
    def __init__(self):
        self.heuristics = {
            1: "DTG", 
            4: "Centroids",
            5: "MCS"
        }
        
        self.domains = ["driverlog", "logistics", "rovers", "satellite"]
        self.results = []
        self.timeout = 60  # 1 minute per experiment
        
    def find_problems(self, domain):
        """Find problems for a domain"""
        domain_path = f"Domains/{domain}"
        problems = []
        
        if os.path.exists(domain_path):
            for item in os.listdir(domain_path):
                if os.path.isdir(os.path.join(domain_path, item)) and item.startswith("Pfile"):
                    problems.append(item)
        
        return sorted(problems)[:3]  # First 3 problems per domain
    
    def run_experiment(self, domain, problem, heuristic_id):
        """Run single experiment"""
        print(f"Testing {domain}/{problem} with {self.heuristics[heuristic_id]}...")
        
        problem_path = f"Domains/{domain}/{problem}"
        
        # Find files
        files = os.listdir(problem_path)
        domain_file = next((f for f in files if 'domain' in f.lower() and f.endswith('.pddl')), None)
        problem_files = [f for f in files if 'problem' in f.lower() and f.endswith('.pddl')]
        agent_file = next((f for f in files if f in ['agents.txt', 'agent-list.txt']), None)
        
        if not domain_file or not problem_files:
            return None
            
        # Build command
        if agent_file and len(problem_files) > 1:
            # Multi-agent
            cmd = ["java", "-jar", "FMAP.jar"]
            for i, pfile in enumerate(problem_files[:2]):  # Max 2 agents
                cmd.extend([f"agent{i+1}", f"{problem_path}/{domain_file}", f"{problem_path}/{pfile}"])
            cmd.extend([f"{problem_path}/{agent_file}", "-h", str(heuristic_id)])
        else:
            # Single agent
            cmd = ["java", "-jar", "FMAP.jar", 
                   "-domain", f"{problem_path}/{domain_file}",
                   "-problem", f"{problem_path}/{problem_files[0]}",
                   "-h", str(heuristic_id)]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            execution_time = time.time() - start_time
            
            # Parse results
            output = result.stdout + result.stderr
            metrics = self.parse_output(output, execution_time)
            metrics.update({
                "domain": domain,
                "problem": problem,
                "heuristic": self.heuristics[heuristic_id],
                "agents": len(problem_files) if agent_file else 1,
                "success": "Solution plan" in output or "CoDMAP" in output
            })
            
            return metrics
            
        except subprocess.TimeoutExpired:
            return {
                "domain": domain,
                "problem": problem,
                "heuristic": self.heuristics[heuristic_id],
                "timeout": True,
                "execution_time": self.timeout
            }
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def parse_output(self, output, execution_time):
        """Parse experiment output"""
        lines = output.split('\n')
        
        plan_length = 0
        heuristic_evals = 0
        
        # Count plan steps
        for line in lines:
            if re.match(r'^\d+:', line.strip()):
                plan_length += 1
            if "Hdtg" in line:
                heuristic_evals += 1
        
        return {
            "execution_time": execution_time,
            "plan_length": plan_length,
            "heuristic_evaluations": heuristic_evals,
            "timeout": False
        }
    
    def run_all_experiments(self):
        """Run all experiments"""
        for domain in self.domains:
            problems = self.find_problems(domain)
            print(f"\n=== Domain: {domain} ===")
            
            for problem in problems:
                for heuristic_id in self.heuristics.keys():
                    result = self.run_experiment(domain, problem, heuristic_id)
                    if result:
                        self.results.append(result)
                        status = "✓" if result.get("success", False) else "✗"
                        print(f"  {status} {self.heuristics[heuristic_id]}: {result.get('plan_length', 0)} steps")
        
        return self.results

if __name__ == "__main__":
    runner = HeuristicExperimentRunner()
    results = runner.run_all_experiments()
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted {len(results)} experiments. Results saved to experiment_results.json") 