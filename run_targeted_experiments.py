#!/usr/bin/env python3
import os
import subprocess
import json
import time
import re
import signal

class TargetedExperimentRunner:
    def __init__(self):
        self.heuristics = {
            1: "DTG", 
            4: "Centroids",
            5: "MCS"
        }
        
        # Focus on working multi-agent problems
        self.test_cases = [
            {
                "domain": "driverlog", 
                "problem": "Pfile1",
                "agents": ["driver1", "driver2"],
                "domain_file": "DomainDriverlog.pddl",
                "problem_files": ["ProblemDriverlogdriver1.pddl", "ProblemDriverlogdriver2.pddl"],
                "agent_file": "agents.txt"
            }
        ]
        
        self.results = []
        self.timeout = 45  # 45 seconds per experiment
    
    def kill_existing_processes(self):
        """Kill any existing FMAP processes"""
        try:
            subprocess.run(["pkill", "-f", "java.*FMAP"], capture_output=True)
            time.sleep(2)
        except:
            pass
    
    def run_experiment(self, test_case, heuristic_id):
        """Run single experiment with proper process management"""
        self.kill_existing_processes()
        
        domain = test_case["domain"]
        problem = test_case["problem"]
        heuristic_name = self.heuristics[heuristic_id]
        
        print(f"Testing {domain}/{problem} with {heuristic_name}...")
        
        problem_path = f"Domains/{domain}/{problem}"
        
        # Build command for multi-agent setup
        cmd = ["java", "-jar", "FMAP.jar"]
        
        # Add each agent
        for i, (agent, problem_file) in enumerate(zip(test_case["agents"], test_case["problem_files"])):
            cmd.extend([
                agent,
                f"{problem_path}/{test_case['domain_file']}",
                f"{problem_path}/{problem_file}"
            ])
        
        # Add agent file and heuristic
        cmd.extend([
            f"{problem_path}/{test_case['agent_file']}",
            "-h", str(heuristic_id)
        ])
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Run with timeout and process management
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                preexec_fn=os.setsid  # Create new process group
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                output = stdout + stderr
                
            except subprocess.TimeoutExpired:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
                
                return {
                    "domain": domain,
                    "problem": problem,
                    "heuristic": heuristic_name,
                    "timeout": True,
                    "execution_time": self.timeout,
                    "success": False,
                    "plan_length": 0,
                    "heuristic_evaluations": 0
                }
            
            # Parse results
            metrics = self.parse_output(output, execution_time)
            metrics.update({
                "domain": domain,
                "problem": problem,
                "heuristic": heuristic_name,
                "agents": len(test_case["agents"]),
                "timeout": False
            })
            
            return metrics
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None
        finally:
            # Ensure cleanup
            self.kill_existing_processes()
    
    def parse_output(self, output, execution_time):
        """Parse experiment output for metrics"""
        lines = output.split('\n')
        
        plan_length = 0
        heuristic_evals = 0
        solution_found = False
        
        # Look for solution indicators
        if "Solution plan" in output or "CoDMAP" in output:
            solution_found = True
        
        # Count plan steps (lines starting with numbers followed by colon)
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+:', line):
                plan_length += 1
            # Count heuristic evaluations
            if "Hdtg" in line or "H=" in line:
                heuristic_evals += 1
        
        return {
            "execution_time": execution_time,
            "plan_length": plan_length,
            "heuristic_evaluations": heuristic_evals,
            "success": solution_found
        }
    
    def run_quick_comparison(self):
        """Run a quick comparison of all heuristics on working problems"""
        print("🚀 Running Targeted Heuristic Comparison...")
        
        for test_case in self.test_cases:
            print(f"\n=== Testing {test_case['domain']}/{test_case['problem']} ===")
            
            for heuristic_id in self.heuristics.keys():
                result = self.run_experiment(test_case, heuristic_id)
                if result:
                    self.results.append(result)
                    
                    if result.get("success", False):
                        print(f"  ✅ {result['heuristic']}: {result['plan_length']} steps in {result['execution_time']:.2f}s")
                    elif result.get("timeout", False):
                        print(f"  ⏱️  {result['heuristic']}: TIMEOUT ({self.timeout}s)")
                    else:
                        print(f"  ❌ {result['heuristic']}: No solution found")
                
                # Small delay between experiments
                time.sleep(3)
        
        return self.results
    
    def extend_experiments(self):
        """Extend to more problems for comprehensive analysis"""
        # Add more test cases from domains that we know work
        additional_cases = []
        
        # Check for more driverlog problems
        driverlog_path = "Domains/driverlog"
        for pfile in sorted(os.listdir(driverlog_path)):
            if pfile.startswith("Pfile") and pfile != "Pfile1":
                pfile_path = os.path.join(driverlog_path, pfile)
                if os.path.isdir(pfile_path):
                    files = os.listdir(pfile_path)
                    if "agents.txt" in files or "agent-list.txt" in files:
                        # Find domain and problem files
                        domain_file = next((f for f in files if "domain" in f.lower() and f.endswith('.pddl')), None)
                        problem_files = [f for f in files if "problem" in f.lower() and f.endswith('.pddl')]
                        agent_file = "agents.txt" if "agents.txt" in files else "agent-list.txt"
                        
                        if domain_file and len(problem_files) >= 2:
                            additional_cases.append({
                                "domain": "driverlog",
                                "problem": pfile,
                                "agents": [f"agent{i+1}" for i in range(len(problem_files))],
                                "domain_file": domain_file,
                                "problem_files": problem_files[:2],  # Limit to 2 agents
                                "agent_file": agent_file
                            })
                
                if len(additional_cases) >= 2:  # Limit to 3 more problems
                    break
        
        self.test_cases.extend(additional_cases)
        print(f"Extended to {len(self.test_cases)} test cases")

if __name__ == "__main__":
    runner = TargetedExperimentRunner()
    
    # First run quick comparison
    results = runner.run_quick_comparison()
    
    # If successful, extend to more problems
    if any(r.get("success", False) for r in results):
        print("\n📈 Initial tests successful! Extending experiments...")
        runner.extend_experiments()
        
        # Run extended experiments
        for test_case in runner.test_cases[1:]:  # Skip the first one we already did
            print(f"\n=== Testing {test_case['domain']}/{test_case['problem']} ===")
            
            for heuristic_id in runner.heuristics.keys():
                result = runner.run_experiment(test_case, heuristic_id)
                if result:
                    runner.results.append(result)
                    
                    if result.get("success", False):
                        print(f"  ✅ {result['heuristic']}: {result['plan_length']} steps in {result['execution_time']:.2f}s")
                    elif result.get("timeout", False):
                        print(f"  ⏱️  {result['heuristic']}: TIMEOUT")
                    else:
                        print(f"  ❌ {result['heuristic']}: No solution")
                
                time.sleep(2)
    
    # Save results
    with open('targeted_experiment_results.json', 'w') as f:
        json.dump(runner.results, f, indent=2)
    
    # Generate summary
    print(f"\n📊 EXPERIMENT SUMMARY")
    print(f"Total experiments: {len(runner.results)}")
    
    if runner.results:
        # Success rates
        print("\n🎯 SUCCESS RATES:")
        for heuristic in runner.heuristics.values():
            heur_results = [r for r in runner.results if r['heuristic'] == heuristic]
            if heur_results:
                success_count = sum(1 for r in heur_results if r.get('success', False))
                success_rate = success_count / len(heur_results) * 100
                print(f"  {heuristic}: {success_count}/{len(heur_results)} ({success_rate:.1f}%)")
        
        # Performance metrics for successful runs
        successful_results = [r for r in runner.results if r.get('success', False)]
        if successful_results:
            print("\n⚡ PERFORMANCE METRICS (Successful runs):")
            for heuristic in runner.heuristics.values():
                heur_success = [r for r in successful_results if r['heuristic'] == heuristic]
                if heur_success:
                    avg_time = sum(r['execution_time'] for r in heur_success) / len(heur_success)
                    avg_length = sum(r['plan_length'] for r in heur_success) / len(heur_success)
                    print(f"  {heuristic}: {avg_time:.2f}s avg time, {avg_length:.1f} avg plan length")
    
    print(f"\nResults saved to targeted_experiment_results.json") 