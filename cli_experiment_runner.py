#!/usr/bin/env python3
"""Run FMAP experiments via CLI and generate results."""

import shlex
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from simple_fmap_stats import SimpleFMAPStats

HEURISTICS = {
    0: "FF",
    1: "DTG",
    2: "DTG+Landmarks",
    3: "Inc.DTG+Landmarks",
    4: "Centroids",
    5: "MCS",
}

EXPERIMENTS = [
    {
        "domain": "driverlog",
        "problem": "Pfile1",
        "agents": 2,
        "complexity": "EASY",
        "cmd": "driver1 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile1/DomainDriverlog.pddl Domains/driverlog/Pfile1/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile1/agents.txt -h {H}",
    },
    {
        "domain": "driverlog",
        "problem": "Pfile2",
        "agents": 2,
        "complexity": "MEDIUM",
        "cmd": "driver1 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile2/DomainDriverlog.pddl Domains/driverlog/Pfile2/ProblemDriverlogdriver2.pddl Domains/driverlog/Pfile2/agent-list.txt -h {H}",
    },
    {
        "domain": "driverlog",
        "problem": "Pfile5",
        "agents": 3,
        "complexity": "HARD",
        "cmd": "driver1 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver1.pddl driver2 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver2.pddl driver3 Domains/driverlog/Pfile5/DomainDriverlog.pddl Domains/driverlog/Pfile5/ProblemDriverlogdriver3.pddl Domains/driverlog/Pfile5/agent-list.txt -h {H}",
    },
    {
        "domain": "ma-blocksworld",
        "problem": "Pfile6-2",
        "agents": 4,
        "complexity": "MEDIUM",
        "cmd": "r0 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr0.pddl r1 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr1.pddl r2 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr2.pddl r3 Domains/ma-blocksworld/Pfile6-2/DomainMaBlocksworld.pddl Domains/ma-blocksworld/Pfile6-2/ProblemMaBlocksr3.pddl Domains/ma-blocksworld/Pfile6-2/agent-list.txt -h {H}",
    },
    {
        "domain": "elevators",
        "problem": "Pfile1",
        "agents": 3,
        "complexity": "EASY",
        "cmd": "fast0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsfast0.pddl slow0-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow0-0.pddl slow1-0 Domains/elevators/Pfile1/DomainElevators.pddl Domains/elevators/Pfile1/ProblemElevatorsslow1-0.pddl Domains/elevators/Pfile1/agent-list.txt -h {H}",
    },
]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "domain",
    "problem",
    "agent_count",
    "complexity",
    "heuristic_id",
    "heuristic_name",
    "timestamp",
    "solution_found",
    "wall_clock_time_sec",
    "peak_memory_mb",
    "plan_length",
    "num_messages",
    "heuristic_evaluations",
    "estimated_branching_factor",
    "search_efficiency",
    "memory_per_agent",
    "messages_per_second",
]

def run_experiment(exp, h_id):
    cmd = exp["cmd"].format(H=h_id)
    args = shlex.split(cmd)
    collector = SimpleFMAPStats()
    stats = collector.run_fmap_with_stats(args)
    row = {
        "domain": exp["domain"],
        "problem": exp["problem"],
        "agent_count": exp["agents"],
        "complexity": exp["complexity"],
        "heuristic_id": h_id,
        "heuristic_name": HEURISTICS[h_id],
        "timestamp": datetime.now().isoformat(),
        "solution_found": stats.get("solution_found", False),
        "wall_clock_time_sec": stats.get("wall_clock_time_sec", 0.0),
        "peak_memory_mb": stats.get("peak_memory_mb", 0.0),
        "plan_length": stats.get("plan_length", 0),
        "num_messages": stats.get("num_messages", 0),
        "heuristic_evaluations": stats.get("heuristic_evaluations", 0),
        "estimated_branching_factor": stats.get("estimated_branching_factor", 0.0),
    }
    if row["wall_clock_time_sec"] > 0:
        row["search_efficiency"] = row["plan_length"] / row["wall_clock_time_sec"]
        row["messages_per_second"] = row["num_messages"] / row["wall_clock_time_sec"]
    else:
        row["search_efficiency"] = 0.0
        row["messages_per_second"] = 0.0
    row["memory_per_agent"] = row["peak_memory_mb"] / row["agent_count"]
    return row

def main():
    results = []
    for exp in EXPERIMENTS:
        for h_id in HEURISTICS.keys():
            print(f"Running {exp['domain']} {exp['problem']} h={h_id} ({HEURISTICS[h_id]})")
            row = run_experiment(exp, h_id)
            results.append(row)
    df = pd.DataFrame(results, columns=COLUMNS)
    df.to_csv("experiment_results.csv", index=False)

    # summary
    summary = df.groupby("heuristic_name")["solution_found"].mean().reset_index()
    summary.to_csv("performance_summary.csv", index=False)

    # heatmap
    pivot = df.pivot_table(values="solution_found", index="heuristic_name", columns="domain", aggfunc="mean")
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap="RdYlGn", fmt=".2f")
    plt.title("Success Rate by Domain")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "success_rate_heatmap.png")

    # execution time boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="heuristic_name", y="wall_clock_time_sec")
    plt.yscale("log")
    plt.title("Execution Time Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "execution_time_comparison.png")

    # memory usage barplot
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x="heuristic_name", y="peak_memory_mb")
    plt.title("Peak Memory Usage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "memory_usage_comparison.png")

    print("Experiments completed. Results saved.")

if __name__ == "__main__":
    main()
