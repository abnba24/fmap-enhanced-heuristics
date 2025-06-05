#!/usr/bin/env python3
import os
import subprocess
import json
import time
import re
import signal

class ComprehensiveExperimentRunner:
    def __init__(self):
        self.heuristics = {
            0: "FF",
            1: "DTG", 
            2: "DTG+Landmarks",
            4: "Centroids",
            5: "MCS"
        }
        
        self.results = []
        self.timeout = 30  # 30 seconds per experiment
    
    def kill_existing_processes(self):
        """Kill any existing FMAP processes"""
        try:
            subprocess.run(["pkill", "-f", "java.*FMAP"], capture_output=True)
            time.sleep(1)
        except:
            pass
    
    def create_single_agent_problem(self, domain_path, problem_path):
        """Create a single-agent version of multi-agent problems"""
        # For single-agent testing, we can use just one of the problem files
        files = os.listdir(domain_path)
        domain_file = next((f for f in files if "domain" in f.lower() and f.endswith('.pddl')), None)
        problem_files = [f for f in files if "problem" in f.lower() and f.endswith('.pddl')]
        
        if domain_file and problem_files:
            return {
                "domain_file": os.path.join(domain_path, domain_file),
                "problem_file": os.path.join(domain_path, problem_files[0]),
                "type": "single_agent"
            }
        return None
    
    def run_single_agent_experiment(self, test_config, heuristic_id):
        """Run single-agent experiment using direct PDDL files"""
        self.kill_existing_processes()
        
        # Create a temporary agent list file for single agent
        agent_list_content = "agent1 127.0.0.1\n"
        temp_agent_file = "temp_agent.txt"
        
        with open(temp_agent_file, 'w') as f:
            f.write(agent_list_content)
        
        # Build command for single agent
        cmd = [
            "java", "-jar", "FMAP.jar",
            "agent1",
            test_config["domain_file"],
            test_config["problem_file"],
            temp_agent_file,
            "-h", str(heuristic_id)
        ]
        
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                preexec_fn=os.setsid
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                output = stdout + stderr
                
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
                
                return {
                    "timeout": True,
                    "execution_time": self.timeout,
                    "success": False,
                    "plan_length": 0,
                    "heuristic_evaluations": 0
                }
            
            # Parse results
            metrics = self.parse_output(output, execution_time)
            return metrics
            
        except Exception as e:
            print(f"Error running single-agent experiment: {e}")
            return None
        finally:
            # Cleanup
            try:
                os.remove(temp_agent_file)
            except:
                pass
            self.kill_existing_processes()
    
    def run_multi_agent_experiment(self, test_case, heuristic_id):
        """Run multi-agent experiment"""
        self.kill_existing_processes()
        
        problem_path = f"Domains/{test_case['domain']}/{test_case['problem']}"
        
        cmd = ["java", "-jar", "FMAP.jar"]
        
        for i, (agent, problem_file) in enumerate(zip(test_case["agents"], test_case["problem_files"])):
            cmd.extend([
                agent,
                f"{problem_path}/{test_case['domain_file']}",
                f"{problem_path}/{problem_file}"
            ])
        
        cmd.extend([
            f"{problem_path}/{test_case['agent_file']}",
            "-h", str(heuristic_id)
        ])
        
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                preexec_fn=os.setsid
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                output = stdout + stderr
                
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(1)
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    pass
                
                return {
                    "timeout": True,
                    "execution_time": self.timeout,
                    "success": False,
                    "plan_length": 0,
                    "heuristic_evaluations": 0
                }
            
            # Parse results
            metrics = self.parse_output(output, execution_time)
            return metrics
            
        except Exception as e:
            print(f"Error running multi-agent experiment: {e}")
            return None
        finally:
            self.kill_existing_processes()
    
    def parse_output(self, output, execution_time):
        """Parse experiment output for metrics"""
        lines = output.split('\n')
        
        plan_length = 0
        heuristic_evals = 0
        solution_found = False
        
        # Look for solution indicators
        if "Solution plan" in output or "CoDMAP" in output or "Solution found" in output:
            solution_found = True
        
        # Count plan steps
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+:', line):
                plan_length += 1
            if "Hdtg" in line or "H=" in line:
                heuristic_evals += 1
        
        return {
            "execution_time": execution_time,
            "plan_length": plan_length,
            "heuristic_evaluations": heuristic_evals,
            "success": solution_found,
            "timeout": False
        }
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across multiple scenarios"""
        print("🚀 Running Comprehensive Heuristic Analysis...")
        
        # Test configurations
        test_scenarios = []
        
        # 1. Single-agent scenarios (converted from multi-agent problems)
        single_agent_domains = ["driverlog", "logistics", "rovers"]
        for domain in single_agent_domains:
            domain_path = f"Domains/{domain}"
            if os.path.exists(domain_path):
                for pfile in sorted(os.listdir(domain_path))[:2]:  # First 2 problems
                    if pfile.startswith("Pfile"):
                        pfile_path = os.path.join(domain_path, pfile)
                        if os.path.isdir(pfile_path):
                            config = self.create_single_agent_problem(pfile_path, pfile_path)
                            if config:
                                test_scenarios.append({
                                    "name": f"{domain}/{pfile}",
                                    "domain": domain,
                                    "problem": pfile,
                                    "type": "single_agent",
                                    "config": config
                                })
        
        # 2. Multi-agent scenarios (known working ones)
        multi_agent_scenarios = [
            {
                "name": "driverlog/Pfile1",
                "domain": "driverlog", 
                "problem": "Pfile1",
                "type": "multi_agent",
                "config": {
                    "agents": ["driver1", "driver2"],
                    "domain_file": "DomainDriverlog.pddl",
                    "problem_files": ["ProblemDriverlogdriver1.pddl", "ProblemDriverlogdriver2.pddl"],
                    "agent_file": "agents.txt"
                }
            }
        ]
        
        test_scenarios.extend(multi_agent_scenarios)
        
        print(f"Testing {len(test_scenarios)} scenarios with {len(self.heuristics)} heuristics")
        
        # Run experiments
        for scenario in test_scenarios:
            print(f"\n=== Testing {scenario['name']} ({scenario['type']}) ===")
            
            for heuristic_id in self.heuristics.keys():
                heuristic_name = self.heuristics[heuristic_id]
                print(f"  Testing {heuristic_name}...")
                
                if scenario['type'] == 'single_agent':
                    result = self.run_single_agent_experiment(scenario['config'], heuristic_id)
                else:
                    result = self.run_multi_agent_experiment(scenario['config'], heuristic_id)
                
                if result:
                    result.update({
                        "scenario": scenario['name'],
                        "domain": scenario['domain'],
                        "problem": scenario['problem'],
                        "scenario_type": scenario['type'],
                        "heuristic_id": heuristic_id,
                        "heuristic": heuristic_name
                    })
                    
                    self.results.append(result)
                    
                    if result.get("success", False):
                        print(f"    ✅ Success: {result['plan_length']} steps in {result['execution_time']:.2f}s")
                    elif result.get("timeout", False):
                        print(f"    ⏱️  Timeout after {self.timeout}s")
                    else:
                        print(f"    ❌ Failed")
                
                time.sleep(1)  # Small delay between experiments
        
        return self.results

if __name__ == "__main__":
    runner = ComprehensiveExperimentRunner()
    results = runner.run_comprehensive_analysis()
    
    # Save results
    with open('comprehensive_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    print(f"\n📊 COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"Total experiments: {len(results)}")
    
    if results:
        # Success rates by heuristic
        print("\n🎯 SUCCESS RATES BY HEURISTIC:")
        for heuristic in runner.heuristics.values():
            heur_results = [r for r in results if r['heuristic'] == heuristic]
            if heur_results:
                success_count = sum(1 for r in heur_results if r.get('success', False))
                success_rate = success_count / len(heur_results) * 100
                print(f"  {heuristic}: {success_count}/{len(heur_results)} ({success_rate:.1f}%)")
        
        # Success rates by scenario type
        print("\n🔍 SUCCESS RATES BY SCENARIO TYPE:")
        for scenario_type in ['single_agent', 'multi_agent']:
            type_results = [r for r in results if r['scenario_type'] == scenario_type]
            if type_results:
                success_count = sum(1 for r in type_results if r.get('success', False))
                success_rate = success_count / len(type_results) * 100
                print(f"  {scenario_type}: {success_count}/{len(type_results)} ({success_rate:.1f}%)")
        
        # Performance metrics for successful runs
        successful_results = [r for r in results if r.get('success', False)]
        if successful_results:
            print("\n⚡ PERFORMANCE METRICS (Successful runs):")
            for heuristic in runner.heuristics.values():
                heur_success = [r for r in successful_results if r['heuristic'] == heuristic]
                if heur_success:
                    avg_time = sum(r['execution_time'] for r in heur_success) / len(heur_success)
                    avg_length = sum(r['plan_length'] for r in heur_success) / len(heur_success)
                    print(f"  {heuristic}: {avg_time:.2f}s avg time, {avg_length:.1f} avg plan length")
    
    print(f"\nResults saved to comprehensive_experiment_results.json") 