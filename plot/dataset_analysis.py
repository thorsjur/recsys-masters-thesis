"""Dataset temporal analysis visualization.

Generates visualizations of dataset temporal properties from experiment configs:
- Interaction volume over time with window overlays
- Time-of-day/week patterns
- User activity and item popularity distributions
"""

import argparse
from typing import Dict, Any, Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd

from plot.common import (
    get_output_dir,
    save_figure,
    print_header,
    load_experiment,
    collect_windows,
    get_time_range,
    run_cli,
)
from util.dataset_analysis import (
    load_temporal_interaction_data,
    compute_temporal_statistics,
    plot_interactions_over_time,
    plot_time_of_day_pattern,
    plot_user_item_distributions,
)


def analyze_dataset_from_experiment(
    experiment_id: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    dataset_path: str = "datasets/atomic_files",
    output_dir: Optional[str] = None,
    start_timestamp: Optional[float] = None,
    figsize: tuple = (12, 6),
    log_scale: bool = False,
) -> Dict[str, Any]:
    """Generate dataset analysis from experiment configuration.

    Returns:
        Dictionary with analysis results and statistics
    """
    results, first = load_experiment(experiment_id, jsonl_path, require_temporal=True)

    dataset_name = first.get("run_info", {}).get("dataset", "unknown")
    granularity = first["window_info"].get("granularity", "hour")
    all_windows = collect_windows(results)
    min_unit, max_unit = get_time_range(all_windows)

    print(f"Loading {dataset_name} data from {granularity} {min_unit} to {max_unit}...")

    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1),
    )

    print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    if start_timestamp is None:
        start_timestamp = float(df["timestamp"].min())

    stats = compute_temporal_statistics(df, granularity, start_timestamp)
    out_dir = get_output_dir(output_dir)
    output_paths = []

    # Interactions over time
    plot_interactions_over_time(df, granularity, start_timestamp, all_windows)
    plt.suptitle(f"{dataset_name} - Interactions Over Time", fontsize=12, fontweight="bold")
    path1 = out_dir / f"{experiment_id}_interactions_timeline.pdf"
    save_figure(plt.gcf(), path1)
    output_paths.append(path1)

    # Time pattern (hourly or day-of-week)
    fig2, ax2 = plt.subplots(figsize=figsize)
    if granularity == "hour":
        plot_time_of_day_pattern(df, start_timestamp, ax=ax2)
        fig2.suptitle(f"{dataset_name} - Hourly Interaction Pattern", fontsize=12, fontweight="bold")
    else:
        _plot_day_of_week(df, ax2, dataset_name)
    path2 = out_dir / f"{experiment_id}_time_pattern.pdf"
    save_figure(fig2, path2)
    output_paths.append(path2)

    # User/Item distributions
    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
    plot_user_item_distributions(df, ax=(ax_left, ax_right), log_scale=log_scale)
    suffix = " (Log-Log)" if log_scale else ""
    fig3.suptitle(f"{dataset_name} - User Activity & Item Popularity{suffix}", fontsize=12, fontweight="bold")
    path3 = out_dir / f"{experiment_id}_distributions.pdf"
    save_figure(fig3, path3)
    output_paths.append(path3)

    # Print summary
    _print_summary(experiment_id, dataset_name, granularity, min_unit, max_unit, stats, len(all_windows))
    print(f"\nGenerated {len(output_paths)} plots in: {out_dir}\n")

    return {"statistics": stats, "dataframe": df, "output_paths": [str(p) for p in output_paths]}


def _plot_day_of_week(df: pd.DataFrame, ax: Axes, dataset_name: str) -> None:
    """Plot day-of-week interaction pattern."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["day_of_week"] = df["datetime"].dt.dayofweek  # type: ignore[union-attr]
    dow_counts = df.groupby("day_of_week").size()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.bar(range(7), [dow_counts.get(i, 0) for i in range(7)], color="#2E86AB", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Day of Week", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Interactions", fontsize=11, fontweight="bold")
    ax.set_title("Interaction Pattern by Day of Week", fontsize=12, fontweight="bold")
    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.gcf().suptitle(f"{dataset_name} - Day of Week Pattern", fontsize=12, fontweight="bold")


def _print_summary(
    exp_id: str, dataset: str, granularity: str, min_u: int, max_u: int, stats: Dict, n_windows: int
) -> None:
    """Print dataset analysis summary."""
    print_header(f"Dataset Analysis Summary - {exp_id}")
    print(f"Dataset: {dataset}")
    print(f"Granularity: {granularity}")
    print(f"Time range: {min_u} to {max_u} ({max_u - min_u + 1} {granularity}s)")
    print(f"Start datetime: {stats['start_datetime']}")
    print(f"\nInteractions (positive/clicked only): {stats['total_interactions']:,}")

    ua = stats["user_activity"]
    print(f"\nUsers: {stats['unique_users']:,}")
    print(f"  Mean: {ua['mean']:.1f} | Median: {ua['median']:.1f} | Range: {ua['min']}-{ua['max']}")

    ip = stats["item_popularity"]
    print(f"\nItems: {stats['unique_items']:,}")
    print(f"  Mean: {ip['mean']:.1f} | Median: {ip['median']:.1f} | Range: {ip['min']}-{ip['max']}")

    ipu = stats["interactions_per_unit"]
    print(f"\nInteractions per {granularity}:")
    print(f"  Mean: {ipu['mean']:.1f} | Std: {ipu['std']:.1f} | Range: {ipu['min']}-{ipu['max']}")
    print(f"\nWindows analyzed: {n_windows}")


def _run(args: argparse.Namespace) -> None:
    analyze_dataset_from_experiment(
        experiment_id=args.experiment_id,
        jsonl_path=args.jsonl_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        start_timestamp=args.start_timestamp,
        figsize=tuple(args.figsize),
        log_scale=args.log_scale,
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset temporal properties from experiments")

    parser.add_argument(
        "--experiment-id", required=True, help="Experiment ID to analyze (must be temporal experiment)"
    )
    parser.add_argument(
        "--jsonl-path", default="output/results/experiments.jsonl", help="Path to experiments.jsonl file"
    )
    parser.add_argument("--dataset-path", default="datasets/atomic_files", help="Path to dataset directory")
    parser.add_argument("--output-dir", "-o", help="Output directory for PDFs")
    parser.add_argument("--start-timestamp", type=float, help="Start timestamp (Unix)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 6], help="Figure size")
    parser.add_argument("--log-scale", action="store_true", help="Use log-log scale for distributions")

    run_cli(_run, parser)


if __name__ == "__main__":
    main()
