#!/usr/bin/env python3
"""Plot FMAP experiment results from CSV file"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_CSV = "experiment_results.csv"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(csv_path: str = RESULTS_CSV) -> pd.DataFrame:
    """Load experiment results CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def create_summary_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics per heuristic."""
    summary = df.groupby("heuristic_name").agg(
        success_rate=("solution_found", "mean"),
        avg_wall_time=("wall_clock_time_sec", "mean"),
        avg_memory=("peak_memory_mb", "mean"),
        avg_plan_length=("plan_length", "mean"),
        avg_messages=("num_messages", "mean"),
    ).round(3)
    summary.to_csv(OUTPUT_DIR / "summary_table.csv")
    return summary


def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        values="solution_found", index="heuristic_name", columns="domain", aggfunc="mean"
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="RdYlGn", fmt=".2f")
    plt.title("Success Rate by Domain")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_rate_heatmap.png")
    plt.close()


def plot_execution_time(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="heuristic_name", y="wall_clock_time_sec")
    plt.yscale("log")
    plt.title("Execution Time Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "execution_time_boxplot.png")
    plt.close()


def plot_memory_usage(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="heuristic_name", y="peak_memory_mb")
    plt.title("Peak Memory Usage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "memory_usage_barplot.png")
    plt.close()


def main() -> None:
    df = load_data()
    summary = create_summary_tables(df)
    print(summary)

    plot_heatmap(df)
    plot_execution_time(df)
    plot_memory_usage(df)
    print(f"Plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
