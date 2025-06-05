#!/bin/bash

echo "=== FMAP Centroids Heuristic Test (Port Conflict Fix) ==="
echo "========================================================"

# Check if port 38000 is in use
echo "Step 1: Checking port 38000 usage..."
CURSOR_PID=$(lsof -ti :38000)
if [ ! -z "$CURSOR_PID" ]; then
    echo "Port 38000 is being used by process $CURSOR_PID (likely Cursor)"
    echo "This needs to be freed for FMAP agents to communicate."
    echo ""
    echo "WARNING: This will temporarily stop Cursor's language server."
    echo "Cursor will restart it automatically, but you may lose some IDE features temporarily."
    echo ""
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Test aborted. Port conflict remains."
        exit 1
    fi
    
    echo "Killing process using port 38000..."
    kill $CURSOR_PID
    sleep 2
    
    # Verify port is free
    NEW_PID=$(lsof -ti :38000)
    if [ ! -z "$NEW_PID" ]; then
        echo "ERROR: Port 38000 still in use by process $NEW_PID"
        exit 1
    fi
    echo "Port 38000 is now free!"
else
    echo "Port 38000 is available!"
fi

echo ""
echo "Step 2: Testing FMAP with Centroids heuristic..."
cd Domains/driverlog/Pfile1

echo "Running DTG heuristic (baseline)..."
timeout 60s java -jar ../../../FMAP.jar driver1 DomainDriverlog.pddl ProblemDriverlogdriver1.pddl driver2 DomainDriverlog.pddl ProblemDriverlogdriver2.pddl agents.txt -h 1 > dtg_test_result.txt 2>&1 &
DTG_PID=$!

# Wait and capture result
sleep 45
kill $DTG_PID 2>/dev/null
wait $DTG_PID 2>/dev/null

echo ""
echo "DTG heuristic results:"
if grep -q "Solution found" dtg_test_result.txt; then
    echo "✅ DTG heuristic: SUCCESS"
    MAKESPAN_DTG=$(grep -E "Makespan|makespan" dtg_test_result.txt | tail -1)
    echo "   Makespan: $MAKESPAN_DTG"
else
    echo "❌ DTG heuristic: FAILED"
    echo "   Error log:"
    tail -5 dtg_test_result.txt | sed 's/^/   /'
fi

echo ""
echo "Running Centroids heuristic..."
sleep 5
timeout 60s java -jar ../../../FMAP.jar driver1 DomainDriverlog.pddl ProblemDriverlogdriver1.pddl driver2 DomainDriverlog.pddl ProblemDriverlogdriver2.pddl agents.txt -h 4 > centroids_test_result.txt 2>&1 &
CENT_PID=$!

# Wait and capture result
sleep 45
kill $CENT_PID 2>/dev/null
wait $CENT_PID 2>/dev/null

echo ""
echo "Centroids heuristic results:"
if grep -q "Solution found" centroids_test_result.txt; then
    echo "✅ Centroids heuristic: SUCCESS"
    MAKESPAN_CENT=$(grep -E "Makespan|makespan" centroids_test_result.txt | tail -1)
    echo "   Makespan: $MAKESPAN_CENT"
else
    echo "❌ Centroids heuristic: FAILED"
    echo "   Error log:"
    tail -5 centroids_test_result.txt | sed 's/^/   /'
fi

echo ""
echo "=== Test Complete ==="
echo "Result files created:"
echo "  - dtg_test_result.txt"
echo "  - centroids_test_result.txt"
echo ""
echo "NOTE: Cursor's language server was temporarily stopped."
echo "It should restart automatically when you use Cursor again." 