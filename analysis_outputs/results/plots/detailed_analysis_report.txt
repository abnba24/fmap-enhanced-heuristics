# FMAP Heuristic Comparison Analysis Report

## Executive Summary

- **Total Experiments**: 65
- **Successful Experiments**: 42
- **Overall Success Rate**: 64.6%
- **Domains Tested**: 5
- **Heuristics Compared**: 5

## Heuristic Performance Ranking

| Rank | Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |
|------|-----------|--------------|------------- |-----------------|-------------|
| 1 | Inc_DTG_Only | 84.6% | 5.79 | 16.6 | 13 |
| 2 | DTG_Only | 84.6% | 14.00 | 14.0 | 13 |
| 3 | DTG+Landmarks | 84.6% | 5.74 | 16.6 | 13 |
| 4 | FF_Heuristic | 38.5% | 325.56 | 12.0 | 13 |
| 5 | Inc_DTG+Landmarks | 30.8% | 83.80 | 10.0 | 13 |

## Detailed Heuristic Analysis

### Inc_DTG_Only

- **Total Experiments**: 13
- **Successful**: 11
- **Success Rate**: 84.6%
- **Average Execution Time**: 5.787 seconds
- **Median Execution Time**: 2.029 seconds
- **Average Memory Usage**: 161.4 MB
- **Average Plan Length**: 16.6 actions
- **Average Search Nodes**: 76
- **Domains Tested**: 5
- **Problems Tested**: 8

### DTG_Only

- **Total Experiments**: 13
- **Successful**: 11
- **Success Rate**: 84.6%
- **Average Execution Time**: 13.999 seconds
- **Median Execution Time**: 1.325 seconds
- **Average Memory Usage**: 185.7 MB
- **Average Plan Length**: 14.0 actions
- **Average Search Nodes**: 156
- **Domains Tested**: 5
- **Problems Tested**: 8

### DTG+Landmarks

- **Total Experiments**: 13
- **Successful**: 11
- **Success Rate**: 84.6%
- **Average Execution Time**: 5.741 seconds
- **Median Execution Time**: 2.073 seconds
- **Average Memory Usage**: 161.9 MB
- **Average Plan Length**: 16.6 actions
- **Average Search Nodes**: 76
- **Domains Tested**: 5
- **Problems Tested**: 8

### FF_Heuristic

- **Total Experiments**: 13
- **Successful**: 5
- **Success Rate**: 38.5%
- **Average Execution Time**: 325.564 seconds
- **Median Execution Time**: 7.262 seconds
- **Average Memory Usage**: 523.5 MB
- **Average Plan Length**: 12.0 actions
- **Average Search Nodes**: 51612
- **Domains Tested**: 5
- **Problems Tested**: 8

### Inc_DTG+Landmarks

- **Total Experiments**: 13
- **Successful**: 4
- **Success Rate**: 30.8%
- **Average Execution Time**: 83.805 seconds
- **Median Execution Time**: 21.949 seconds
- **Average Memory Usage**: 269.7 MB
- **Average Plan Length**: 10.0 actions
- **Average Search Nodes**: 24140
- **Domains Tested**: 5
- **Problems Tested**: 8

## Domain-Specific Performance

### Depots Domain

| Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |
|-----------|--------------|------------- |-----------------|-------------|
| Inc_DTG_Only | 100.0% | 43.86 | 28.0 | 1 |
| DTG+Landmarks | 100.0% | 43.49 | 28.0 | 1 |
| DTG_Only | 100.0% | 138.54 | 0.0 | 1 |
| FF_Heuristic | 0.0% | N/A | N/A | 1 |
| Inc_DTG+Landmarks | 0.0% | N/A | N/A | 1 |

### Driverlog Domain

| Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |
|-----------|--------------|------------- |-----------------|-------------|
| DTG+Landmarks | 100.0% | 1.69 | 16.0 | 4 |
| DTG_Only | 100.0% | 1.48 | 15.5 | 4 |
| Inc_DTG_Only | 100.0% | 1.70 | 16.0 | 4 |
| Inc_DTG+Landmarks | 50.0% | 145.66 | 13.0 | 4 |
| FF_Heuristic | 50.0% | 131.93 | 13.0 | 4 |

### Elevators Domain

| Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |
|-----------|--------------|------------- |-----------------|-------------|
| Inc_DTG_Only | 100.0% | 3.37 | 20.5 | 2 |
| DTG_Only | 100.0% | 2.05 | 20.0 | 2 |
| DTG+Landmarks | 100.0% | 3.33 | 20.5 | 2 |
| FF_Heuristic | 50.0% | 1354.53 | 18.0 | 2 |
| Inc_DTG+Landmarks | 0.0% | N/A | N/A | 2 |

### Openstacks Domain

| Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |
|-----------|--------------|------------- |-----------------|-------------|
| Inc_DTG_Only | 0.0% | N/A | N/A | 2 |
| DTG_Only | 0.0% | N/A | N/A | 2 |
| FF_Heuristic | 0.0% | N/A | N/A | 2 |
| DTG+Landmarks | 0.0% | N/A | N/A | 2 |
| Inc_DTG+Landmarks | 0.0% | N/A | N/A | 2 |

### Zenotravel Domain

| Heuristic | Success Rate | Avg Time (s) | Avg Plan Length | Experiments |
|-----------|--------------|------------- |-----------------|-------------|
| DTG_Only | 100.0% | 1.36 | 13.0 | 4 |
| Inc_DTG_Only | 100.0% | 1.57 | 12.5 | 4 |
| DTG+Landmarks | 100.0% | 1.56 | 12.5 | 4 |
| FF_Heuristic | 50.0% | 4.72 | 8.0 | 4 |
| Inc_DTG+Landmarks | 50.0% | 21.95 | 7.0 | 4 |

## Key Insights

1. **Best Overall Heuristic**: Inc_DTG_Only with 84.6% success rate

2. **Fastest Heuristic**: DTG+Landmarks with average time of 5.741 seconds

3. **Most Successful Domain**: Driverlog (80.0% average success rate)
4. **Most Challenging Domain**: Openstacks (0.0% average success rate)

5. **Complexity Observations**:
   - High-complexity domains (>5 agents): depots
   - Simple domains (≤3 agents): driverlog, elevators, openstacks, zenotravel

## Recommendations

1. **For General Use**: Consider using the top-performing heuristics with high success rates
2. **For Speed-Critical Applications**: Use the fastest successful heuristics
3. **For Complex Problems**: Test multiple heuristics as performance varies by domain
4. **Future Research**: Focus on improving performance for challenging domains
