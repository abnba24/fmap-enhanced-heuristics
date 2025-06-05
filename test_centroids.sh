#!/bin/bash

echo "Testing FMAP Centroids Heuristic"
echo "================================="

# Kill any existing FMAP processes
pkill -f "java.*FMAP" 2>/dev/null
sleep 3

cd Domains/driverlog/Pfile1

echo "Testing with DTG heuristic (baseline)..."
timeout 60s java -jar ../../../FMAP.jar driver1 DomainDriverlog.pddl ProblemDriverlogdriver1.pddl driver2 DomainDriverlog.pddl ProblemDriverlogdriver2.pddl agent-list.txt -h 1 > dtg_result.txt 2>&1 &
DTG_PID=$!

sleep 45
kill $DTG_PID 2>/dev/null

echo ""
echo "Testing with Centroids heuristic..."
sleep 5
timeout 60s java -jar ../../../FMAP.jar driver1 DomainDriverlog.pddl ProblemDriverlogdriver1.pddl driver2 DomainDriverlog.pddl ProblemDriverlogdriver2.pddl agent-list.txt -h 4 > centroids_result.txt 2>&1 &
CENT_PID=$!

sleep 45
kill $CENT_PID 2>/dev/null

echo ""
echo "Results:"
echo "--------"
if [ -f dtg_result.txt ]; then
    echo "DTG result file size: $(wc -l < dtg_result.txt) lines"
    grep -i "solution\|makespan\|actions\|time" dtg_result.txt | head -5
fi

if [ -f centroids_result.txt ]; then
    echo "Centroids result file size: $(wc -l < centroids_result.txt) lines"  
    grep -i "solution\|makespan\|actions\|time" centroids_result.txt | head -5
fi

echo ""
echo "Done. Check dtg_result.txt and centroids_result.txt for full output." 