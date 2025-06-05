# 🧪 Comprehensive FMAP GUI Heuristic Experiment Plan

## 🎯 Experimental Objectives

1. **Compare all 6 heuristics** across different domains
2. **Analyze impact of problem complexity** on heuristic performance  
3. **Evaluate multi-agent coordination efficiency**
4. **Assess scalability** with increasing agent count
5. **Identify domain-specific heuristic advantages**

## 📊 Heuristics Under Test

| ID | Heuristic | Description |
|----|-----------|-------------|
| 0 | FF | Fast-Forward heuristic |
| 1 | DTG | Domain Transition Graph |
| 2 | DTG + Landmarks | DTG with landmark analysis (default) |
| 3 | Inc. DTG + Landmarks | Incremental DTG + Landmarks |
| 4 | Centroids | Fixed Centroids heuristic |
| 5 | MCS | Min. Covering States |

## 🗂️ Experimental Test Suite

### **Tier 1: Core Multi-Agent Domains (PRIORITY)**

#### **1. Driverlog Domain** 🚛
**Characteristics**: Transportation, coordination, resource sharing
- **Pfile1** (2 agents) - `EASY` baseline
- **Pfile2** (2 agents) - `MEDIUM` complexity  
- **Pfile3** (2 agents) - `MEDIUM` complexity
- **Pfile5** (3 agents) - `HARD` coordination challenge

**GUI Commands:**
```bash
# Pfile1 (2 agents)
java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h [0-5] -gui

# Pfile5 (3 agents) 
java -jar FMAP.jar driver1 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver2.pddl driver3 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver3.pddl Domains/driverlog/Pfile5/agent-list.txt -h [0-5] -gui
```

#### **2. MA-Blocksworld Domain** 🧱
**Characteristics**: Manipulation, dependencies, coordination
- **Pfile4-2** (2 agents) - `EASY` robot coordination
- **Pfile6-2** (4 agents) - `MEDIUM` complex coordination
- **Pfile8-2** (4 agents) - `HARD` complex dependencies
- **Pfile12-1** (4+ agents) - `VERY HARD` scalability test

**GUI Commands:**
```bash
# Pfile6-2 (4 agents - we have agent-list.txt ready)
java -jar FMAP.jar r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h [0-5] -gui
```

#### **3. Elevators Domain** 🏢
**Characteristics**: Resource conflicts, scheduling, optimization
- **Pfile1** (3 agents) - `EASY` elevator coordination
- **Pfile5** (3+ agents) - `MEDIUM` traffic management
- **Pfile10** (3+ agents) - `HARD` complex scheduling

**GUI Commands:**
```bash
# Pfile1 (3 elevators)
java -jar FMAP.jar fast0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsfast0.pddl slow0-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow0-0.pddl slow1-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow1-0.pddl Domains/elevators/Pfile1/agent-list.txt -h [0-5] -gui
```

### **Tier 2: Extended Analysis Domains**

#### **4. Logistics Domain** 📦
**Characteristics**: Multi-modal transport, complex routing
- **Pfile1** (2+ agents) - `EASY`
- **Pfile5** (3+ agents) - `MEDIUM`
- **Pfile10** (4+ agents) - `HARD`

#### **5. Rovers Domain** 🤖
**Characteristics**: Exploration, resource collection, coordination
- **Pfile1** (2 rovers) - `EASY`
- **Pfile5** (3+ rovers) - `MEDIUM`
- **Pfile10** (4+ rovers) - `HARD`

#### **6. Satellite Domain** 🛰️
**Characteristics**: Scheduling, resource optimization, temporal planning
- **Pfile01** (multiple satellites) - `EASY`
- **Pfile05** (complex scheduling) - `MEDIUM` 
- **Pfile10** (high coordination) - `HARD`

## 🔬 Experimental Protocol

### **Step 1: Setup Agent List Files**
Create missing agent list files for each test case:
```bash
# Example for Pfile5 (3 agents)
echo "driver1 127.0.0.1" > Domains/driverlog/Pfile5/agent-list.txt
echo "driver2 127.0.0.1" >> Domains/driverlog/Pfile5/agent-list.txt  
echo "driver3 127.0.0.1" >> Domains/driverlog/Pfile5/agent-list.txt
```

### **Step 2: Systematic Testing Matrix**

For **each problem** × **each heuristic (0-5)**:

1. **Launch GUI**: `java -jar FMAP.jar [agents...] [agent-list] -h X -gui`
2. **Record Statistics** from trace window:
   - Planning (expansion) time
   - Evaluation time  
   - Communication time
   - Average branching factor
   - Discarded plans
   - Used memory
   - Plan length
   - Number of messages
   - Total time

### **Step 3: Data Collection Template**

| Domain | Problem | Agents | Heuristic | Time | Evaluations | Memory | Plan Length | Messages | Success |
|--------|---------|--------|-----------|------|-------------|--------|-------------|----------|---------|
| driverlog | Pfile1 | 2 | DTG(1) | 0.088s | 19 | 133MB | 7 | 89 | ✅ |
| driverlog | Pfile1 | 2 | Centroids(4) | 1.258s | 328 | 145MB | 7 | 127 | ✅ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 📈 Analysis Metrics

### **Primary Performance Indicators**
1. **Success Rate** - % of problems solved
2. **Planning Time** - Total time to find solution
3. **Search Efficiency** - Evaluations per second
4. **Memory Usage** - Peak memory consumption
5. **Solution Quality** - Plan length and makespan

### **Heuristic-Specific Analysis**
1. **DTG vs Centroids** - Compare our fixed implementation
2. **Landmark Integration** - Effect of landmarks (h=2,3)
3. **Domain Sensitivity** - Which heuristics work best per domain
4. **Scalability** - Performance vs number of agents

### **Coordination Analysis**
1. **Message Complexity** - Communication overhead
2. **Branching Factor** - Search space pruning effectiveness
3. **Plan Quality** - Optimality and concurrency

## 🎯 Expected Experimental Outcomes

### **Hypothesis 1: Domain Specificity**
- **Driverlog**: DTG should perform well (transportation graphs)
- **Blocksworld**: Landmarks should help (dependency analysis)
- **Elevators**: MCS might excel (state covering)

### **Hypothesis 2: Centroids Performance**
- Fixed Centroids should show **competitive performance**
- Should excel in **coordination-heavy scenarios**
- May have **higher search overhead** but **better solution quality**

### **Hypothesis 3: Scalability Patterns**
- **DTG**: Fast but may struggle with complexity
- **DTG+Landmarks**: Better on complex problems
- **MCS**: Most robust across different scales

## 🚀 Quick Start Commands

### **Immediate Testing (Copy-Paste Ready)**

**Test 1: Driverlog Pfile1 - All Heuristics**
```bash
# DTG
java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h 1 -gui

# Centroids (Fixed)
java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h 4 -gui

# MCS
java -jar FMAP.jar driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h 5 -gui
```

**Test 2: MA-Blocksworld Pfile6-2 - Key Heuristics**
```bash
# DTG + Landmarks (default)
java -jar FMAP.jar r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h 2 -gui

# Centroids (Fixed)
java -jar FMAP.jar r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h 4 -gui
```

## 📝 Results Documentation

Create detailed experiment logs with:
1. **Screenshot captures** of GUI trace windows
2. **Timing measurements** for each heuristic
3. **Plan quality analysis** comparing solutions
4. **Error cases** where heuristics fail or timeout
5. **Memory usage patterns** across different domains

This comprehensive experimental plan will provide robust data for analyzing heuristic performance in multi-agent planning scenarios! 