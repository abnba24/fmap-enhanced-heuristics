# 🤖 Enhanced FMAP: Multi-Agent Planning with Advanced Heuristics

[![Java](https://img.shields.io/badge/Java-8+-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-green.svg)](LICENSE.md)

**Enhanced FMAP** is an advanced multi-agent planning system based on the original FMAP platform, featuring **two new state-of-the-art heuristics** and a comprehensive experimental framework for research in distributed planning.

## ✨ Key Enhancements

### 🧠 **New Heuristics Implemented**

| Heuristic | ID | Description | Optimization Target |
|-----------|----|-----------| ------------------|
| **Centroids** | `h=4` | Minimizes mean cost to goals: `μ{ĥ(s, Gi)}` | **Expected performance** |
| **MCS** | `h=5` | Minimizes max cost to goals: `max{ĥ(s, Gi)}` | **Worst-case robustness** |

### 📊 **Comprehensive Analysis Framework**
- **Command-line statistics extraction** (GUI-equivalent metrics)
- **Automated experimental pipelines** with statistical significance testing
- **Publication-ready visualizations** and performance comparisons
- **Multi-domain testing suite** across 9+ planning domains

### 🔧 **Developer Tools**
- **PyAutoGUI automation** for large-scale GUI experiments
- **Python analysis scripts** with pandas/matplotlib integration
- **Batch processing capabilities** for systematic evaluation

## 🚀 Quick Start

### **Run with New Heuristics**

```bash
# Centroids heuristic (minimizes mean cost to goals)
java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl \
  Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl \
  driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl \
  Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl \
  Domains/driverlog/Pfile1/agents.txt -h 4 -gui

# MCS heuristic (minimizes max cost to goals)  
java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl \
  Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl \
  driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl \
  Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl \
  Domains/driverlog/Pfile1/agents.txt -h 5 -gui
```

### **Get GUI-Style Statistics from Command Line**

```bash
python3 simple_fmap_stats.py -- \
  driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl \
  Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl \
  driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl \
  Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl \
  Domains/driverlog/Pfile1/agents.txt -h 4
```

## 📈 Performance Results

### **Comparative Analysis** (Driverlog Pfile1)

| Heuristic | Time | Evaluations | Memory | Plan Quality | Success |
|-----------|------|-------------|--------|--------------|---------|
| **DTG** | 0.555s | 19 | 81 MB | 7 actions | ✅ |
| **Centroids** | 1.461s | 328 | 131 MB | 7 actions | ✅ |
| **MCS** | 1.069s | 217 | 126 MB | 7 actions | ✅ |

**Key Findings**:
- ✅ All heuristics find **optimal solutions** (identical plan quality)
- 🎯 **DTG**: Fastest execution, minimal search  
- 🧮 **Centroids**: Most thorough exploration, comprehensive goal analysis
- ⚖️ **MCS**: Balanced performance, robust across scenarios

## 🏗️ Architecture

### **Heuristic Integration**

The new heuristics integrate seamlessly with FMAP's existing architecture:

```java
// HeuristicFactory.java - Registration
public static final int CENTROIDS = 4;      // Centroids heuristic
public static final int MCS = 5;           // MCS heuristic

// HeuristicFactoryImp.java - Instantiation  
case CENTROIDS:
    h = new CentroidsHeuristic(comm, gTask, pf);
    break;
case MCS:
    h = new MCSHeuristic(comm, gTask, pf);
    break;
```

### **Multi-Agent Support**

Both heuristics support:
- ✅ **Single-agent** and **multi-agent** planning modes
- ✅ **Privacy-preserving** distributed search
- ✅ **DTG-based** path cost computation
- ✅ **Thread-safe** evaluation for concurrent planning

## 🧪 Experimental Framework

### **Automated Testing**

```bash
# Run comprehensive heuristic comparison
cd experiments
source venv/bin/activate  
python run_experiments.py --quick

# Generate statistical analysis
python analyze_results.py

# Create publication-quality plots
python visualizer.py
```

### **Domains Tested**

| Domain | Characteristics | Agent Types | Complexity |
|--------|-----------------|-------------|------------|
| **Driverlog** | Transportation, coordination | 2-3 drivers | Easy-Hard |
| **MA-Blocksworld** | Manipulation, dependencies | 2-4 robots | Medium-Hard |
| **Elevators** | Resource scheduling | 3+ elevators | Easy-Medium |
| **Logistics** | Multi-modal transport | 2-4 agents | Medium-Hard |
| **Rovers** | Exploration, collection | 2-4 rovers | Easy-Hard |

## 📚 Documentation

- **[Complete User Guide](FMAP_STATS_GUIDE.md)** - Command-line statistics and usage
- **[Experimental Framework](EXPERIMENT_SUMMARY.md)** - Research applications and setup
- **[GUI Automation](AUTOMATION_AGENT_PROMPT.md)** - Large-scale testing protocols
- **[Heuristic Analysis](heuristic_analysis_report.md)** - Performance evaluation results

## 🔬 Research Applications

This enhanced FMAP enables:

### **Heuristic Research**
- 📊 **Comparative analysis** of planning heuristics
- 📈 **Performance trade-off studies** (speed vs. quality)
- 🧠 **Multi-agent coordination** efficiency analysis
- 📋 **Statistical significance testing** with effect sizes

### **Planning Applications**
- 🚛 **Transportation** and logistics optimization
- 🤖 **Multi-robot** coordination and task allocation  
- 🏢 **Resource scheduling** and conflict resolution
- 🛰️ **Distributed systems** planning and coordination

## 🛠️ Development

### **Project Evolution**

```
FMAP_original.jar → FMAP_new.jar → FMAP_updated.jar → FMAP_final.jar → FMAP.jar
     ↓                ↓              ↓                  ↓              ↓
   Baseline     First attempt    Bug fixes      Stable version   Active dev
```

### **Key Contributions**

1. **✅ Centroids Heuristic Implementation** - Mean-cost optimization for balanced multi-goal planning
2. **✅ MCS Heuristic Implementation** - Worst-case optimization for robust planning  
3. **✅ Statistical Analysis Framework** - Comprehensive performance evaluation tools
4. **✅ Command-Line Integration** - GUI-equivalent statistics from batch execution
5. **✅ Automation Pipeline** - Large-scale experimental capabilities

## 📦 Installation

### **Prerequisites**
- **Java 8+** (for FMAP execution)
- **Python 3.7+** (for analysis tools)
- **Git** (for repository management)

### **Setup**
```bash
git clone https://github.com/yourusername/fmap-enhanced-heuristics.git
cd fmap-enhanced-heuristics

# Test installation
java -jar FMAP.jar --help

# Setup Python environment (optional)
python3 -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🧠 **New heuristic functions** for multi-agent planning
- 📊 **Enhanced analysis tools** and visualizations  
- 🔧 **Performance optimizations** and scaling improvements
- 📚 **Documentation** and example applications

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see [LICENSE.md](LICENSE.md) for details.

## 🏆 Acknowledgments

- **Original FMAP Team** - Universitat Politècnica de València
- **Research Foundation** - Pozanco et al. "Finding Centroids and Minimum Covering States in Planning" (ICAPS 2019)
- **Multi-Agent Planning Community** - For inspiration and validation

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue or contact the development team.

---

**⭐ If this project helps your research, please consider giving it a star!** 