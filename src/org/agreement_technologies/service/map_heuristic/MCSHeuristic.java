/* 
 * Copyright (C) 2017 Universitat Politècnica de València
 *
 * This file is part of FMAP.
 *
 * FMAP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * FMAP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with FMAP. If not, see <http://www.gnu.org/licenses/>.
 */
package org.agreement_technologies.service.map_heuristic;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import org.agreement_technologies.common.map_communication.AgentCommunication;
import org.agreement_technologies.common.map_grounding.GroundedTask;
import org.agreement_technologies.common.map_grounding.GroundedVar;
import org.agreement_technologies.common.map_grounding.GroundedCond;
import org.agreement_technologies.common.map_heuristic.HPlan;
import org.agreement_technologies.common.map_heuristic.Heuristic;
import org.agreement_technologies.common.map_planner.PlannerFactory;
import org.agreement_technologies.common.map_dtg.DTGSet;
import org.agreement_technologies.service.map_dtg.DTGSetImp;
import org.agreement_technologies.common.map_dtg.DTG;

/**
 * Minimum Covering States (MCS) heuristic function evaluator. This heuristic 
 * computes the maximum (worst-case) cost to any possible goal, minimizing
 * the worst-case cost of reaching any goal from the current state.
 * 
 * Based on: Pozanco et al. "Finding Centroids and Minimum Covering States in Planning" (ICAPS 2019)
 * Formula: f_mcs(s) = max{ĥ(s, Gi)} = max{ĥ(s, G1), ĥ(s, G2), ..., ĥ(s, Gn)}
 *
 * @author FMAP Extension
 * @version %I%, %G%
 * @since 1.0
 */
public class MCSHeuristic implements Heuristic {

    protected GroundedTask groundedTask;        // Grounded task
    protected AgentCommunication comm;          // Communication utility
    protected PlannerFactory pf;                // Planner factory
    protected DTGSet dtgs;                      // Domain Transition Graphs
    
    private ArrayList<GoalCondition> goals;     // List of all possible goals
    private ArrayList<Goal> dtgGoals;           // DTG-style goals for computation
    private static final int INFINITE = (Integer.MAX_VALUE) / 3;

    /**
     * Constructs a Minimum Covering States (MCS) heuristic evaluator.
     *
     * @param comm Communication utility
     * @param gTask Grounded task
     * @param pf Planner factory
     * @since 1.0
     */
    public MCSHeuristic(AgentCommunication comm, GroundedTask gTask, PlannerFactory pf) {
        this.groundedTask = gTask;
        this.comm = comm;
        this.pf = pf;
        
        // Initialize DTG set for path cost computation
        this.dtgs = new DTGSetImp(gTask);
        this.dtgs.distributeDTGs(comm, gTask);
        
        // Initialize goals from the task
        this.goals = HeuristicToolkit.computeTaskGoals(comm, gTask);
        this.dtgGoals = new ArrayList<>();
        
        // Convert goals to DTG format for computation
        for (GoalCondition g : goals) {
            GroundedVar var = null;
            for (GroundedVar v : gTask.getVars()) {
                if (v.toString().equals(g.varName)) {
                    var = v;
                    break;
                }
            }
            if (var != null) {
                Goal ng = new Goal(gTask.createGroundedCondition(GroundedCond.EQUAL, var, g.value), 0);
                dtgGoals.add(ng);
            }
        }
    }

    /**
     * Beginning of the heuristic evaluation stage.
     *
     * @param basePlan Base plan, whose successors will be evaluated
     * @since 1.0
     */
    @Override
    public void startEvaluation(HPlan basePlan) {
        // No special initialization needed for MCS
    }

    /**
     * Heuristic evaluation of a plan. Computes the maximum cost to any goal.
     * The resulting value is stored inside the plan (see setH method in Plan interface).
     *
     * @param p Plan to evaluate
     * @param threadIndex Thread index, for multi-threading purposes
     * @since 1.0
     */
    @Override
    public void evaluatePlan(HPlan p, int threadIndex) {
        if (p.isSolution()) {
            p.setH(0, threadIndex);
            return;
        }

        // If we have no goals, set heuristic to 0
        if (dtgGoals.isEmpty()) {
            p.setH(0, threadIndex);
            return;
        }

        // ACTUAL MCS IMPLEMENTATION
        // Calculate the maximum cost to any goal: max(s) = max{h(s, G1), h(s, G2), ..., h(s, Gn)}
        
        // Compute current state
        int[] stepOrder = p.linearization();
        HashMap<String, String> state;
        HashMap<String, ArrayList<String>> newValues = new HashMap<>();
        
        if (comm.numAgents() == 1) {
            // Single-agent mode
            state = p.computeState(stepOrder, pf);
            evaluateMCS_MonoAgent(p, state, newValues, threadIndex);
        } else {
            // Multi-agent mode
            HashMap<String, ArrayList<String>> multiState = p.computeMultiState(stepOrder, pf);
            evaluateMCS_MultiAgent(p, multiState, threadIndex);
        }
    }
    
    /**
     * Evaluates MCS heuristic in single-agent mode.
     */
    private void evaluateMCS_MonoAgent(HPlan p, HashMap<String, String> state, 
                                      HashMap<String, ArrayList<String>> newValues, int threadIndex) {
        int maxCost = 0;
        
        for (Goal goal : dtgGoals) {
            int goalCost = computeSingleGoalCost_Mono(goal, state, newValues, threadIndex);
            
            if (goalCost >= INFINITE) {
                // If any goal is unreachable, MCS is infinite
                p.setH(INFINITE, threadIndex);
                return;
            }
            
            // Track maximum cost across all goals
            if (goalCost > maxCost) {
                maxCost = goalCost;
            }
        }
        
        // Set the MCS heuristic value (maximum cost to any goal)
        p.setH(maxCost, threadIndex);
    }
    
    /**
     * Evaluates MCS heuristic in multi-agent mode.
     */
    private void evaluateMCS_MultiAgent(HPlan p, HashMap<String, ArrayList<String>> varValues, int threadIndex) {
        int maxCost = 0;
        
        for (Goal goal : dtgGoals) {
            int goalCost = computeSingleGoalCost_Multi(goal, varValues);
            
            if (goalCost >= INFINITE || goalCost < 0) {
                // If any goal is unreachable, use penalty but continue to find worst case
                goalCost = 100; // Use a high penalty instead of infinite
            }
            
            // Track maximum cost across all goals
            if (goalCost > maxCost) {
                maxCost = goalCost;
            }
        }
        
        // Set the MCS heuristic value (maximum cost to any goal)
        p.setH(maxCost, threadIndex);
    }
    
    /**
     * Computes the cost to reach a single goal in single-agent mode.
     * Uses DTG path cost computation for accurate estimation.
     */
    private int computeSingleGoalCost_Mono(Goal goal, HashMap<String, String> state, 
                                          HashMap<String, ArrayList<String>> newValues, int threadIndex) {
        String varName = goal.varName;
        String targetValue = goal.varValue;
        
        // Check if goal is already satisfied
        if (holdsMono(varName, targetValue, state, newValues)) {
            return 0;
        }
        
        // Use DTG to compute path cost to this specific goal
        DTG dtg = dtgs.getDTG(varName);
        String initValue = selectInitialValueMono(varName, targetValue, dtg, state, newValues);
        
        // Compute actual DTG path cost
        int pathCost = dtg.pathCost(initValue, targetValue, state, newValues, threadIndex);
        
        if (pathCost < 0 || pathCost >= INFINITE) {
            return INFINITE;
        }
        
        return pathCost;
    }
    
    /**
     * Computes the cost to reach a single goal in multi-agent mode.
     */
    private int computeSingleGoalCost_Multi(Goal goal, HashMap<String, ArrayList<String>> varValues) {
        String varName = goal.varName;
        String targetValue = goal.varValue;
        
        // Check if goal is already satisfied
        if (holdsMulti(varName, targetValue, varValues)) {
            return 0;
        }
        
        // Use DTG to compute path cost to this specific goal
        DTG dtg = dtgs.getDTG(varName);
        String initValue = selectInitialValueMulti(varName, targetValue, dtg, varValues);
        
        // Compute actual DTG path cost
        int pathCost = dtg.pathCostMulti(initValue, targetValue);
        
        return pathCost;
    }
    
    /**
     * Selects the best initial value for a transition in single-agent mode.
     */
    private String selectInitialValueMono(String varName, String endValue, DTG dtg,
            HashMap<String, String> state, HashMap<String, ArrayList<String>> newValues) {
        String bestValue = state.get(varName);
        int bestCost = dtg.pathCost(bestValue, endValue, state, newValues, 0);
        ArrayList<String> valueList = newValues.get(varName);
        if (valueList != null) {
            for (String value : valueList) {
                int cost = dtg.pathCost(value, endValue, state, newValues, 0);
                if (cost != -1 && cost < bestCost) {
                    bestCost = cost;
                    bestValue = value;
                }
            }
        }
        return bestValue;
    }
    
    /**
     * Selects the best initial value for a transition in multi-agent mode.
     */
    private String selectInitialValueMulti(String varName, String endValue, DTG dtg,
            HashMap<String, ArrayList<String>> varValues) {
        ArrayList<String> values = varValues.get(varName);
        if (values == null || values.isEmpty()) {
            return "?";
        }
        
        String bestValue = values.get(0);
        int bestCost = dtg.pathCostMulti(bestValue, endValue);
        
        for (String value : values) {
            int cost = dtg.pathCostMulti(value, endValue);
            if (cost < bestCost) {
                bestCost = cost;
                bestValue = value;
            }
        }
        return bestValue;
    }
    
    /**
     * Checks if a condition holds in single-agent mode.
     */
    private static boolean holdsMono(String varName, String value, HashMap<String, String> state,
            HashMap<String, ArrayList<String>> newValues) {
        String v = state.get(varName);
        if (v != null && v.equals(value)) {
            return true;
        }
        ArrayList<String> values = newValues.get(varName);
        if (values == null) {
            return false;
        }
        return values.contains(value);
    }
    
    /**
     * Checks if a condition holds in multi-agent mode.
     */
    private static boolean holdsMulti(String varName, String value, HashMap<String, ArrayList<String>> varValues) {
        ArrayList<String> values = varValues.get(varName);
        return values != null && values.contains(value);
    }

    /**
     * Multi-heuristic evaluation of a plan.
     *
     * @param p Plan to evaluate
     * @param threadIndex Thread index, for multi-threading purposes
     * @param achievedLandmarks List of already achieved landmarks
     * @since 1.0
     */
    @Override
    public void evaluatePlan(HPlan p, int threadIndex, ArrayList<Integer> achievedLandmarks) {
        // MCS doesn't use landmarks by default, so use standard evaluation
        evaluatePlan(p, threadIndex);
    }

    /**
     * Heuristically evaluates the cost of reaching the agent's private goals.
     *
     * @param p Plan to evaluate
     * @param threadIndex Thread index, for multi-threading purposes
     * @since 1.0
     */
    @Override
    public void evaluatePlanPrivacy(HPlan p, int threadIndex) {
        // For simplicity, set private goal costs to 0
        // In a full implementation, you would apply MCS logic to private goals too
        for (int i = 0; i < groundedTask.getPreferences().size(); i++) {
            p.setHPriv(0, i);
        }
    }

    /**
     * Synchronization step after the distributed heuristic evaluation.
     *
     * @since 1.0
     */
    @Override
    public void waitEndEvaluation() {
        // No synchronization needed for MCS
    }

    /**
     * Gets information about a given topic.
     *
     * @param infoFlag Topic to get information about.
     * @return Object with the requested information
     * @since 1.0
     */
    @Override
    public Object getInformation(int infoFlag) {
        return null; // No special information provided
    }

    /**
     * Checks if the current heuristic evaluator supports multi-threading.
     *
     * @return <code>true</code>, if multi-threading evaluation is available.
     * <code>false</code>, otherwise
     * @since 1.0
     */
    @Override
    public boolean supportsMultiThreading() {
        return true;
    }

    /**
     * Checks if the current heuristic evaluator requires an additional stage
     * for landmarks evaluation.
     *
     * @return <code>true</code>, if a landmarks evaluation stage is required.
     * <code>false</code>, otherwise
     * @since 1.0
     */
    @Override
    public boolean requiresHLandStage() {
        return false;
    }

    /**
     * Returns the total number of global (public) landmarks.
     * 
     * @return Total number of global (public) landmarks
     * @since 1.0
     */
    @Override
    public int numGlobalLandmarks() {
        return 0; // MCS doesn't use landmarks
    }

    /**
     * Landmark analysis.
     *
     * @param plan Current plan
     * @param achievedLandmarks Already achieved landmarks
     * @return List of newly achieved landmarks
     * @since 1.0
     */
    @Override
    public ArrayList<Integer> checkNewLandmarks(HPlan plan, BitSet achievedLandmarks) {
        return new ArrayList<>(); // MCS doesn't use landmarks
    }

    /**
     * Goal class defines (sub)goals for the priority queue of open goals.
     * 
     * @author FMAP Extension
     * @since 1.0
     */
    private static class Goal implements Comparable<Goal> {

        String varName;     // Variable name
        String varValue;    // Value for the variable
        int distance;       // Path distance to reach this goal

        /**
         * Creates a new goal.
         *
         * @param goal Grounded condition
         * @param distance Path distance
         * @since 1.0
         */
        public Goal(GroundedCond goal, int distance) {
            this(goal.getVar().toString(), goal.getValue(), distance);
        }

        /**
         * Creates a new goal.
         *
         * @param varName Variable name
         * @param varValue Value for the variable
         * @param distance Path distance to reach this goal
         * @since 1.0
         */
        public Goal(String varName, String varValue, int distance) {
            this.varName = varName;
            this.varValue = varValue;
            this.distance = distance;
        }

        /**
         * Compares two goals.
         *
         * @param g Another goal to compare with this one
         * @return Value less than zero if the distance to this goal is smaller;
         * Value greater than zero if the distance to the other goal is smaller;
         * Zero, otherwise
         * @since 1.0
         */
        @Override
        public int compareTo(Goal g) {
            return g.distance - distance;
        }

        /**
         * Returns a description of this goal.
         *
         * @return Description of this goal
         * @since 1.0
         */
        @Override
        public String toString() {
            return varName + "=" + varValue + "(" + distance + ")";
        }

        /**
         * Gets a hash code for this goal.
         *
         * @return Hash code
         * @since 1.0
         */
        @Override
        public int hashCode() {
            return (varName + "=" + varValue).hashCode();
        }

        /**
         * Check if two goals are equal.
         *
         * @param x Another goal to compare with this one.
         * @return <code>true</code>, if both goals have the same variable name
         * and value; <code>false</code>, otherwise
         * @since 1.0
         */
        @Override
        public boolean equals(Object x) {
            Goal g = (Goal) x;
            return varName.equals(g.varName) && varValue.equals(g.varValue);
        }
    }
} 