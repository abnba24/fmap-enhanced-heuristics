# FMAP Command Line Statistics Guide

## 🎯 Getting GUI-Style Statistics from Command Line

The `simple_fmap_stats.py` wrapper provides detailed performance statistics similar to what you see in FMAP's GUI trace window, but from command line execution.

## 📊 Available Statistics

The wrapper captures and displays:

### ⏱️ Timing Breakdown
- **Planning (expansion) time**: Time spent expanding search nodes
- **Evaluation time**: Time spent evaluating heuristic functions  
- **Communication time**: Time spent on agent coordination
- **Grounding time**: Time spent grounding PDDL into internal representation
- **Total FMAP time**: Total time reported by FMAP
- **Wall clock time**: Actual execution time measured externally

### 🔍 Search Statistics
- **Node expansions**: Number of search nodes expanded
- **Heuristic evaluations**: Number of heuristic function calls
- **Average branching factor**: Estimated branching factor of search tree
- **Discarded plans**: Number of plans discarded during search

### 💾 Memory Usage
- **Peak memory usage**: Maximum memory consumption during execution

### 📋 Plan Quality (if solution found)
- **Plan length**: Number of actions in solution
- **Makespan**: Temporal length of parallel plan
- **Parallel actions**: Number of concurrent actions
- **Concurrency index**: Ratio of parallel to total actions

### 📡 Communication
- **Number of messages**: Total messages exchanged between agents

## 🚀 Usage

### Basic Usage
```bash
python3 simple_fmap_stats.py -- [FMAP arguments]
```

### Examples

**Two-agent driverlog problem with DTG heuristic:**
```bash
python3 simple_fmap_stats.py -- \
  driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl \
  driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl \
  Domains/driverlog/Pfile1/agents.txt -h 1
```

**Four-agent ma-blocksworld problem with Centroids heuristic:**
```bash
python3 simple_fmap_stats.py -- \
  r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl \
  r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl \
  r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl \
  r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl \
  Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h 4
```

### Command Line Options

```bash
python3 simple_fmap_stats.py [OPTIONS] -- [FMAP arguments]

Options:
  --verbose, -v         Show verbose output (FMAP stdout/stderr)
  --output FILE, -o     Save statistics to JSON file
  --timeout SECONDS     Set timeout (default: 300 seconds)
```

### Save Results to File
```bash
python3 simple_fmap_stats.py --output results.json -- [FMAP arguments]
```

### View Verbose Output
```bash
python3 simple_fmap_stats.py --verbose -- [FMAP arguments]
```

## 📈 Sample Output

```
======================================================================
📊 FMAP EXECUTION STATISTICS (GUI-Style Output)
======================================================================
✅ SOLUTION FOUND

⏱️  TIMING BREAKDOWN:
   Planning (expansion) time: 0.458 sec.
   Evaluation time:          0.131 sec.
   Communication time:       0.065 sec.
   Grounding time:           45 ms.
   Total FMAP time:          0.721 sec.
   Wall clock time:          0.721 sec.

🔍 SEARCH STATISTICS:
   Node expansions:          5
   Heuristic evaluations:    5
   Average branching factor: 0.625
   Discarded plans:          0

💾 MEMORY USAGE:
   Peak memory usage:        128 MB

📋 PLAN QUALITY:
   Plan length:              8 actions
   Makespan:                 3.0
   Parallel actions:         0
   Concurrency index:        0.000

📡 COMMUNICATION:
   Number of messages:       89

Planning completed in 0.721 sec.
======================================================================
```

## 🔧 How It Works

1. **External Timing**: Measures wall clock time around FMAP execution
2. **Output Parsing**: Extracts available metrics from FMAP's stdout/stderr
3. **Memory Monitoring**: Uses system tools to track memory usage
4. **Estimation**: Calculates derived metrics (branching factor, timing breakdown)
5. **Formatting**: Presents results in GUI-style format

## 🎯 Comparison with GUI Mode

| Statistic | GUI Mode | Command Line Wrapper |
|-----------|----------|---------------------|
| Planning time | ✅ Exact | ✅ Exact (from FMAP) |
| Evaluation time | ✅ Exact | ⚠️ Estimated (20% of planning time) |
| Communication time | ✅ Exact | ⚠️ Estimated (10% of planning time) |
| Branching factor | ✅ Exact | ⚠️ Estimated (evaluations/depth) |
| Memory usage | ✅ Exact | ✅ System measurement |
| Plan length | ✅ Exact | ✅ Exact (from FMAP) |
| Messages | ✅ Exact | ✅ Exact (from FMAP) |
| Node expansions | ✅ Exact | ⚠️ Estimated (≈ heuristic evaluations) |

## 💡 Tips

1. **Port Conflicts**: If you get "Address already in use" errors, wait a moment between runs
2. **Timeout**: Increase timeout for complex problems: `--timeout 1800`
3. **Debugging**: Use `--verbose` to see raw FMAP output
4. **Automation**: Save to JSON for analysis: `--output stats.json`

## 🔍 Troubleshooting

**"Address already in use" error:**
- Wait 10-15 seconds between FMAP runs
- Kill any hanging FMAP processes: `pkill -f FMAP.jar`

**No statistics captured:**
- Check that FMAP.jar exists in current directory
- Verify domain/problem files exist
- Use `--verbose` to see error messages

**Memory statistics showing 0:**
- This is normal on some systems due to security restrictions
- The wrapper will still capture other statistics

## 📚 Related Scripts

- `test_stats_example.py` - Demonstration with sample output
- `fmap_stats_wrapper.py` - Advanced version (requires psutil)
- `run_targeted_experiments.py` - Batch experiment runner with statistics 