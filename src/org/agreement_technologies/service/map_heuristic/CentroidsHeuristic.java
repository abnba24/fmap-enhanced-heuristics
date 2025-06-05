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
import org.agreement_technologies.common.map_grounding.GroundedCond;
import org.agreement_technologies.common.map_grounding.GroundedTask;
import org.agreement_technologies.common.map_grounding.GroundedVar;
import org.agreement_technologies.common.map_heuristic.HPlan;
import org.agreement_technologies.common.map_heuristic.Heuristic;
import org.agreement_technologies.common.map_planner.PlannerFactory;
import org.agreement_technologies.common.map_dtg.DTG;
import org.agreement_technologies.common.map_dtg.DTGSet;
import org.agreement_technologies.service.map_dtg.DTGSetImp;

/**
 * Centroids heuristic function evaluator.
 * 
 * This heuristic computes the mean (average) cost to all possible goals,
 * representing the expected cost from the current state to any goal.
 * The centroids approach minimizes the maximum expected distance to goal states.
 * 
 * Formula: h_centroids(s) = μ{h(s, g_i)} = (1/|G|) * Σ h(s, g_i) for all goals g_i
 * 
 * @author FMAP Extension Team
 * @version %I%, %G%
 * @since 1.0
 */
public class CentroidsHeuristic implements Heuristic {

    private static final int INFINITE = 999999;  // Infinite cost
    protected GroundedTask groundedTask;        // Grounded task
    protected AgentCommunication comm;          // Communication utility
    protected PlannerFactory pf;                // Planner factory
    protected DTGSet dtgs;                      // Domain Transition Graphs
    protected ArrayList<Goal> goals;            // Task goals

    /**
     * Constructs a Centroids heuristic evaluator.
     *
     * @param comm Communication utility
     * @param gTask Grounded task
     * @param pf Planner factory
     * @since 1.0
     */
    public CentroidsHeuristic(AgentCommunication comm, GroundedTask gTask, PlannerFactory pf) {
        this.groundedTask = gTask;
        this.comm = comm;
        this.pf = pf;
        
        // Initialize DTG set for path cost computation
        this.dtgs = new DTGSetImp(gTask);
        this.dtgs.distributeDTGs(comm, gTask);
        
        // Initialize goals from the task
        this.goals = new ArrayList<>();
        ArrayList<GoalCondition> gc = HeuristicToolkit.computeTaskGoals(comm, gTask);
        for (GoalCondition g : gc) {
            GroundedVar var = null;
            for (GroundedVar v : gTask.getVars()) {
                if (v.toString().equals(g.varName)) {
                    var = v;
                    break;
                }
            }
            if (var != null) {
                Goal ng = new Goal(gTask.createGroundedCondition(GroundedCond.EQUAL, var, g.value), 0);
                goals.add(ng);
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
        // No special initialization needed for centroids heuristic
    }

    /**
     * Heuristic evaluation of a plan using Centroids approach.
     * Computes the mean (average) cost to all goals using DTG path costs.
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
        if (goals.isEmpty()) {
            p.setH(0, threadIndex);
            return;
        }

        // Check if this is single-agent or multi-agent mode
        if (comm.numAgents() == 1) {
            evaluateCentroids_MonoAgent(p, threadIndex);
        } else {
            evaluateCentroids_MultiAgent(p, threadIndex);
        }
    }
    
    /**
     * Evaluates Centroids heuristic in single-agent mode.
     * Computes the mean cost to all goals using DTG path costs.
     */
    private void evaluateCentroids_MonoAgent(HPlan p, int threadIndex) {
        int totalOrder[] = p.linearization();
        HashMap<String, String> state = p.computeState(totalOrder, pf);
        HashMap<String, ArrayList<String>> newValues = new HashMap<>();
        
        int totalCost = 0;
        int validGoals = 0;
        
        for (Goal goal : goals) {
            int goalCost = computeSingleGoalCost_Mono(goal, state, newValues, threadIndex);
            
            if (goalCost < INFINITE && goalCost >= 0) {
                totalCost += goalCost;
                validGoals++;
            }
        }
        
        // Calculate mean cost (centroids heuristic)
        int centroidsValue;
        if (validGoals > 0) {
            centroidsValue = totalCost / validGoals; // Integer division for mean
        } else {
            centroidsValue = INFINITE; // No reachable goals
        }
        
        p.setH(centroidsValue, threadIndex);
    }
    
    /**
     * Evaluates Centroids heuristic in multi-agent mode.
     */
    private void evaluateCentroids_MultiAgent(HPlan p, int threadIndex) {
        int totalOrder[] = p.linearization();
        HashMap<String, ArrayList<String>> varValues = p.computeMultiState(totalOrder, pf);
        
        int totalCost = 0;
        int validGoals = 0;
        
        for (Goal goal : goals) {
            int goalCost = computeSingleGoalCost_Multi(goal, varValues);
            
            if (goalCost < INFINITE && goalCost >= 0) {
                totalCost += goalCost;
                validGoals++;
            }
        }
        
        // Calculate mean cost (centroids heuristic)
        int centroidsValue;
        if (validGoals > 0) {
            centroidsValue = totalCost / validGoals; // Integer division for mean
        } else {
            centroidsValue = INFINITE; // No reachable goals
        }
        
        p.setH(centroidsValue, threadIndex);
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
     * Checks if a condition holds in single-agent mode.
     */
    private boolean holdsMono(String varName, String value, HashMap<String, String> state, 
                             HashMap<String, ArrayList<String>> newValues) {
        if (value.equals(state.get(varName))) {
            return true;
        }
        ArrayList<String> achievedValues = newValues.get(varName);
        return achievedValues != null && achievedValues.contains(value);
    }
    
    /**
     * Checks if a condition holds in multi-agent mode.
     */
    private boolean holdsMulti(String varName, String value, HashMap<String, ArrayList<String>> varValues) {
        ArrayList<String> values = varValues.get(varName);
        return values != null && values.contains(value);
    }
    
    /**
     * Selects the best initial value for a transition in single-agent mode.
     * Based on DTGHeuristic.selectInitialValueMono implementation.
     */
    private String selectInitialValueMono(String varName, String targetValue, DTG dtg, 
                                         HashMap<String, String> state, HashMap<String, ArrayList<String>> newValues) {
        String currentValue = state.get(varName);
        ArrayList<String> achievedValues = newValues.get(varName);
        
        if (currentValue != null) {
            return currentValue;
        } else if (achievedValues != null && !achievedValues.isEmpty()) {
            // Find the best value from achieved values based on path cost
            String bestValue = achievedValues.get(0);
            int bestCost = dtg.pathCost(bestValue, targetValue, state, newValues, 0);
            for (String value : achievedValues) {
                int cost = dtg.pathCost(value, targetValue, state, newValues, 0);
                if (cost >= 0 && cost < bestCost) {
                    bestCost = cost;
                    bestValue = value;
                }
            }
            return bestValue;
        } else {
            return "?"; // Unknown initial value
        }
    }
    
    /**
     * Selects the best initial value for a transition in multi-agent mode.
     * Based on DTGHeuristic.selectInitialValueMulti implementation.
     */
    private String selectInitialValueMulti(String varName, String targetValue, DTG dtg, 
                                          HashMap<String, ArrayList<String>> varValues) {
        ArrayList<String> values = varValues.get(varName);
        
        if (values == null || values.isEmpty()) {
            return "?";
        }
        
        // Find the best value based on path cost (following DTGHeuristic pattern)
        String bestValue = null;
        int bestCost = -1;
        for (String value : values) {
            if (bestValue == null) {
                bestCost = dtg.pathCostMulti(value, targetValue);
                bestValue = value;
            } else {
                int cost = dtg.pathCostMulti(value, targetValue);
                if (cost != -1 && cost < bestCost) {
                    bestCost = cost;
                    bestValue = value;
                }
            }
        }
        return bestValue != null ? bestValue : "?";
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
        evaluatePlan(p, threadIndex);
    }

    /**
     * Synchronization step after the distributed heuristic evaluation.
     *
     * @since 1.0
     */
    @Override
    public void waitEndEvaluation() {
        // No synchronization needed for centroids heuristic
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
        return 0;
    }

    /**
     * Returns the new landmarks achieved in this plan.
     * 
     * @param plan Plan to check
     * @param achievedLandmarks Already achieved landmarks
     * @return List of indexes of the new achieved landmarks
     * @since 1.0
     */
    @Override
    public ArrayList<Integer> checkNewLandmarks(HPlan plan, BitSet achievedLandmarks) {
        return null;
    }
    
    /**
     * Goal class for centroids computation.
     * Represents a planning goal with variable and target value.
     */
    private static class Goal {
        String varName;     // Variable name
        String varValue;    // Target value for the variable
        
        /**
         * Creates a new goal from a grounded condition.
         *
         * @param goal Grounded condition representing the goal
         * @param distance Unused (for compatibility)
         * @since 1.0
         */
        public Goal(GroundedCond goal, int distance) {
            this.varName = goal.getVar().toString();
            this.varValue = goal.getValue();
        }
        
        /**
         * Returns a description of this goal.
         *
         * @return Goal description
         * @since 1.0
         */
        @Override
        public String toString() {
            return varName + "=" + varValue;
        }
    }
} 