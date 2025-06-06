# FMAP DOMAIN TESTING AND HOLISTIC EXPERIMENT AUTOMATION PROMPT

## MISSION OBJECTIVE
Systematically test all FMAP domains to identify working multi-agent configurations, populate accurate agent files, create representative sample sets spanning 2-10 agents, and execute comprehensive heuristic comparison experiments.

## PHASE 1: DOMAIN COMPATIBILITY TESTING

### Step 1.1: Domain Discovery and Inventory
```bash
# Navigate to Domains directory
cd /Users/abrahamadeniyi/Desktop/altorler-fmap-2ce663469695/Domains

# Create comprehensive domain inventory
for domain in */; do
    echo "=== DOMAIN: ${domain%/} ==="
    for problem in "$domain"*/; do
        if [ -d "$problem" ]; then
            agent_count=$(ls "$problem"Problem*.pddl 2>/dev/null | wc -l)
            echo "  ${problem##*/}: $agent_count agents"
        fi
    done
done
```

### Step 1.2: Agent File Population
For each problem directory:
1. **Extract agent names** from problem filenames (Pattern: `Problem<Domain><Agent>.pddl`)
2. **Create agents.txt** with format: `<agent_name> 127.0.0.1`
3. **Verify agent-to-file mapping** is correct

```python
# Auto-populate agents.txt files
import os
from pathlib import Path

def populate_agent_files():
    domains_dir = Path("Domains")
    for domain_dir in domains_dir.iterdir():
        if domain_dir.is_dir():
            for problem_dir in domain_dir.iterdir():
                if problem_dir.is_dir():
                    # Extract agents from problem files
                    agents = []
                    for file in problem_dir.iterdir():
                        if file.name.startswith("Problem") and file.suffix == ".pddl":
                            # Extract agent name from filename
                            filename = file.name.replace("Problem", "").replace(".pddl", "")
                            domain_name = domain_dir.name
                            agent_name = filename.replace(domain_name, "").replace(domain_name.capitalize(), "")
                            agent_name = agent_name.strip("-_")
                            if agent_name and agent_name not in agents:
                                agents.append(agent_name)
                    
                    # Create agents.txt
                    if len(agents) >= 2:
                        agents_file = problem_dir / "agents.txt"
                        with open(agents_file, 'w') as f:
                            for agent in sorted(agents):
                                f.write(f"{agent} 127.0.0.1\n")
                        print(f"Created {agents_file} with {len(agents)} agents: {agents}")
```

### Step 1.3: Systematic Domain Testing
Test each domain with representative multi-agent problems:

```bash
#!/bin/bash
# Domain Testing Script

WORKING_DOMAINS=()
FAILED_DOMAINS=()
TIMEOUT=30

test_domain_problem() {
    local domain=$1
    local problem=$2
    local domain_path="Domains/$domain/$problem"
    
    if [ ! -d "$domain_path" ]; then
        echo "SKIP: $domain/$problem (directory not found)"
        return 2
    fi
    
    cd "$domain_path"
    
    # Check if agents.txt exists
    if [ ! -f "agents.txt" ]; then
        echo "SKIP: $domain/$problem (no agents.txt)"
        cd - > /dev/null
        return 2
    fi
    
    # Count agents
    agent_count=$(wc -l < agents.txt)
    if [ $agent_count -lt 2 ]; then
        echo "SKIP: $domain/$problem (insufficient agents: $agent_count)"
        cd - > /dev/null
        return 2
    fi
    
    # Build FMAP command
    cmd="timeout $TIMEOUT java -jar ../../../FMAP.jar -h 1"
    
    # Add agent specifications
    while IFS= read -r line; do
        if [[ ! $line =~ ^[[:space:]]*$ ]] && [[ ! $line =~ ^# ]]; then
            agent_name=$(echo "$line" | cut -d' ' -f1)
            domain_file="Domain${domain^}.pddl"
            problem_file="Problem${domain^}${agent_name}.pddl"
            
            if [ -f "$domain_file" ] && [ -f "$problem_file" ]; then
                cmd="$cmd $agent_name $domain_file $problem_file"
            else
                echo "SKIP: $domain/$problem (missing files for $agent_name)"
                cd - > /dev/null
                return 2
            fi
        fi
    done < agents.txt
    
    cmd="$cmd agents.txt"
    
    # Execute test
    echo "TESTING: $domain/$problem ($agent_count agents)"
    result=$(eval "$cmd" 2>&1)
    exit_code=$?
    
    cd - > /dev/null
    
    # Analyze result
    if [ $exit_code -eq 0 ] && echo "$result" | grep -q "Solution plan"; then
        echo "SUCCESS: $domain/$problem"
        return 0
    elif echo "$result" | grep -q "NullPointerException"; then
        echo "FAILED: $domain/$problem (NullPointerException)"
        return 1
    elif echo "$result" | grep -q "Address already in use"; then
        echo "FAILED: $domain/$problem (Port conflict)"
        return 1
    elif [ $exit_code -eq 124 ]; then
        echo "TIMEOUT: $domain/$problem"
        return 1
    else
        echo "FAILED: $domain/$problem (Unknown error)"
        return 1
    fi
}

# Test representative problems from each domain
DOMAINS=("driverlog" "logistics" "rovers" "satellite" "elevators" "ma-blocksworld" "openstacks" "woodworking" "zenotravel" "depots")

for domain in "${DOMAINS[@]}"; do
    echo "=== TESTING DOMAIN: $domain ==="
    
    # Test multiple problems to get coverage
    success_count=0
    problem_count=0
    
    for problem in Domains/$domain/*/; do
        if [ -d "$problem" ]; then
            problem_name=$(basename "$problem")
            test_domain_problem "$domain" "$problem_name"
            result=$?
            
            ((problem_count++))
            if [ $result -eq 0 ]; then
                ((success_count++))
            fi
            
            # Test up to 5 problems per domain for efficiency
            if [ $problem_count -ge 5 ]; then
                break
            fi
        fi
    done
    
    # Classify domain
    if [ $success_count -gt 0 ]; then
        WORKING_DOMAINS+=("$domain")
        echo "DOMAIN $domain: WORKING ($success_count/$problem_count successful)"
    else
        FAILED_DOMAINS+=("$domain")
        echo "DOMAIN $domain: FAILED (0/$problem_count successful)"
    fi
    
    echo ""
done

echo "=== DOMAIN TESTING SUMMARY ==="
echo "Working domains: ${WORKING_DOMAINS[*]}"
echo "Failed domains: ${FAILED_DOMAINS[*]}"
```

## PHASE 2: SAMPLE SET CREATION

### Step 2.1: Agent Count Distribution Analysis
```python
# Analyze agent distribution across working domains
def analyze_agent_distribution():
    working_domains = ["driverlog", "rovers", "satellite"]  # Update based on testing
    agent_distribution = {}
    
    for domain in working_domains:
        domain_path = Path(f"Domains/{domain}")
        for problem_dir in domain_path.iterdir():
            if problem_dir.is_dir():
                agents_file = problem_dir / "agents.txt"
                if agents_file.exists():
                    with open(agents_file) as f:
                        agent_count = len([line for line in f if line.strip() and not line.startswith('#')])
                    
                    if agent_count not in agent_distribution:
                        agent_distribution[agent_count] = []
                    agent_distribution[agent_count].append(f"{domain}/{problem_dir.name}")
    
    return agent_distribution
```

### Step 2.2: Strategic Sample Selection
Create representative sample spanning 2-10 agents:

```python
def create_strategic_sample():
    """
    Create holistic sample set with:
    - Balanced representation across agent counts (2-10)
    - Multiple domains for robustness
    - All 5 heuristics tested on each configuration
    - Complexity variation (small/medium/large problems)
    """
    
    target_agent_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    sample_configs = []
    
    # For each target agent count
    for agent_count in target_agent_counts:
        # Find working problems with this agent count
        matching_problems = find_problems_with_agent_count(agent_count)
        
        if matching_problems:
            # Select representative problems (different domains, complexities)
            selected = select_representative_problems(matching_problems, count=3)
            
            # Test all 5 heuristics on each selected problem
            for problem in selected:
                for heuristic_id in [1, 2, 3, 4, 5]:
                    config = ExperimentConfig(
                        domain=problem['domain'],
                        problem=problem['problem'],
                        heuristic=heuristic_id,
                        agents=problem['agents'],
                        agent_files=problem['agent_files']
                    )
                    sample_configs.append(config)
    
    return sample_configs
```

## PHASE 3: EXPERIMENT EXECUTION

### Step 3.1: Pre-flight Validation
```bash
# Validate all sample configurations before running experiments
python3 -c "
from experiment_runner import ExperimentRunner
runner = ExperimentRunner()
configs = runner.get_strategic_configs()
print(f'Total configurations: {len(configs)}')

# Validate each config
valid_configs = []
for config in configs:
    if runner.validate_config(config):
        valid_configs.append(config)
    else:
        print(f'INVALID: {config.domain}/{config.problem}')

print(f'Valid configurations: {len(valid_configs)}')
"
```

### Step 3.2: Comprehensive Experiment Execution
```bash
# Clear previous results
rm -rf experiments/results/*

# Run strategic experiments with comprehensive coverage
cd experiments
python3 experiment_runner.py --strategic

# Verify results
echo "Experiment Summary:"
python3 -c "
import json
with open('results/all_results.json') as f:
    results = json.load(f)
successful = sum(1 for r in results if r['search']['coverage'])
total = len(results)
print(f'Success rate: {successful}/{total} ({100*successful/total:.1f}%)')

# Agent count distribution
from collections import defaultdict
agent_dist = defaultdict(int)
heuristic_dist = defaultdict(int)
for r in results:
    if r['search']['coverage']:
        agent_dist[len(r['config']['agents'])] += 1
        heuristic_dist[r['config']['heuristic']] += 1

print('Agent count distribution:', dict(agent_dist))
print('Heuristic distribution:', dict(heuristic_dist))
"
```

### Step 3.3: Analysis and Visualization
```bash
# Generate comprehensive analysis
python3 experiment_runner.py --analyze-only

# Verify plots were generated
ls -la results/plots/

echo "Analysis complete. Results available in:"
echo "- results/statistical_summary.txt"
echo "- results/plots/*.png"
echo "- results/all_results.json"
```

## PHASE 4: QUALITY ASSURANCE

### Step 4.1: Results Validation
- Verify all target agent counts (2-10) are represented
- Confirm all 5 heuristics have data points
- Check for balanced domain representation
- Validate statistical significance of comparisons

### Step 4.2: Success Criteria
- **Minimum 50% overall success rate**
- **At least 3 different domains working**
- **Agent count coverage: 2-8 agents minimum**
- **All 5 heuristics tested at least 10 times each**
- **Statistical plots generated successfully**

## EXPECTED OUTCOMES

1. **Domain Compatibility Map**: Clear identification of which domains work vs fail
2. **Comprehensive Dataset**: 100-200 successful experiments across agent counts/heuristics
3. **Performance Analysis**: Statistical ranking of heuristics with confidence intervals
4. **Scalability Insights**: Performance trends across agent counts
5. **Publication-Ready Visualizations**: Professional plots showing heuristic comparisons

## EXECUTION COMMAND

```bash
# Execute full automation pipeline
bash domain_testing_script.sh
python3 populate_agent_files.py
python3 experiment_runner.py --strategic
python3 experiment_runner.py --analyze-only
```

This systematic approach will provide robust, comprehensive data for heuristic comparison across the full spectrum of multi-agent planning scenarios. 