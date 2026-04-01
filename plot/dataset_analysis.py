
import argparse
from typing import Any, Dict, Optional
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_analysis.atomic_file import find_item_file, load_item_dataframe
from data_analysis.dataset_summary_table import save_dataset_summary_table, summarize_interaction_dataframe
from plot.common import (
    collect_windows,
    get_output_dir,
    get_time_range,
    load_experiment,
    print_header,
    run_cli,
    save_figure,
)
from plot.dataset_interactions_timeline import plot_interactions_timeline
from plot.dataset_time_pattern import plot_time_pattern
from plot.dataset_user_item_distributions import plot_user_item_distributions
from util.dataset_analysis import compute_temporal_statistics, load_temporal_interaction_data


def analyze_dataset_from_experiment(
    experiment_id: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    dataset_path: str = "data/atomic_files",
    output_dir: Optional[str] = None,
    start_timestamp: Optional[float] = None,
    figsize: tuple[float, float] = (12, 6),
    log_scale: bool = False,
    primary_color: str = "#2E86AB",
    bucket_hours: int = 24,
) -> Dict[str, Any]:
    results, first = load_experiment(experiment_id, jsonl_path, require_temporal=True)

    dataset_name = first.get("run_info", {}).get("dataset", "unknown")
    granularity = first["window_info"].get("granularity", "hour")
    windows = collect_windows(results)
    min_unit, max_unit = get_time_range(windows)

    print(f"Loading {dataset_name} data from {granularity} {min_unit} to {max_unit}...")

    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1),
    )

    if df.empty:
        raise ValueError(
            f"No interactions found for dataset '{dataset_name}' ({granularity} {min_unit}-{max_unit}). "
            "Check that the atomic files exist and that the dataset name matches the directory."
        )

    print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    if start_timestamp is None:
        start_timestamp = float(df["timestamp"].min())

    stats = compute_temporal_statistics(df, granularity, start_timestamp)
    out_dir = get_output_dir(output_dir)
    output_paths = []
    table_paths = []

    fig1, ax1 = plt.subplots(figsize=figsize)
    plot_interactions_timeline(df, ax1, dataset_name, bucket_hours, primary_color)
    path1 = out_dir / f"{experiment_id}_interactions_timeline.pdf"
    save_figure(fig1, path1)
    output_paths.append(path1)

    fig2, ax2 = plt.subplots(figsize=figsize)
    plot_time_pattern(df, ax2, dataset_name, granularity, bucket_hours, primary_color)
    path2 = out_dir / f"{experiment_id}_time_pattern.pdf"
    save_figure(fig2, path2)
    output_paths.append(path2)

    fig3, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
    plot_user_item_distributions(df, (axes[0], axes[1]), dataset_name, log_scale, primary_color)
    path3 = out_dir / f"{experiment_id}_distributions.pdf"
    save_figure(fig3, path3)
    output_paths.append(path3)

    item_df = _maybe_load_item_dataframe(dataset_name, dataset_path)
    summary_table = save_dataset_summary_table(
        _build_summary_table(dataset_name, df, item_df),
        out_dir / f"{experiment_id}_dataset_summary.csv",
    )
    table_paths.append(summary_table)

    _print_summary(experiment_id, dataset_name, granularity, min_unit, max_unit, stats, len(windows))
    print(f"\nGenerated {len(output_paths)} plots and {len(table_paths)} table in: {out_dir}\n")

    return {
        "statistics": stats,
        "dataframe": df,
        "output_paths": [str(path) for path in output_paths],
        "table_paths": [str(path) for path in table_paths],
    }


def _print_summary(
    exp_id: str,
    dataset: str,
    granularity: str,
    min_unit: int,
    max_unit: int,
    stats: Dict[str, Any],
    n_windows: int,
) -> None:
    print_header(f"Dataset Analysis Summary - {exp_id}")
    print(f"Dataset: {dataset}")
    print(f"Granularity: {granularity}")
    print(f"Time range: {min_unit} to {max_unit} ({max_unit - min_unit + 1} {granularity}s)")
    print(f"Start datetime: {stats['start_datetime']}")
    print(f"\nInteractions (positive/clicked only): {stats['total_interactions']:,}")

    user_activity = stats["user_activity"]
    print(f"\nUsers: {stats['unique_users']:,}")
    print(
        f"  Mean: {user_activity['mean']:.1f} | "
        f"Median: {user_activity['median']:.1f} | "
        f"Range: {user_activity['min']}-{user_activity['max']}"
    )

    item_popularity = stats["item_popularity"]
    print(f"\nItems: {stats['unique_items']:,}")
    print(
        f"  Mean: {item_popularity['mean']:.1f} | "
        f"Median: {item_popularity['median']:.1f} | "
        f"Range: {item_popularity['min']}-{item_popularity['max']}"
    )

    per_unit = stats["interactions_per_unit"]
    print(f"\nInteractions per {granularity}:")
    print(f"  Mean: {per_unit['mean']:.1f} | Std: {per_unit['std']:.1f} | Range: {per_unit['min']}-{per_unit['max']}")
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
        primary_color=args.primary_color,
        bucket_hours=args.bucket_hours,
    )


def _build_summary_table(dataset_name: str, df: pd.DataFrame, item_df: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame([summarize_interaction_dataframe(dataset_name, df, item_df)])


def _maybe_load_item_dataframe(dataset_name: str, dataset_path: str) -> pd.DataFrame | None:
    try:
        item_path = find_item_file(dataset_name, dataset_path)
    except FileNotFoundError:
        return None
    return load_item_dataframe(item_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset temporal properties from experiments")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to analyze (must be temporal experiment)")
    parser.add_argument("--jsonl-path", default="output/results/experiments.jsonl", help="Path to experiments.jsonl file")
    parser.add_argument("--dataset-path", default="data/atomic_files", help="Path to dataset directory")
    parser.add_argument("--output-dir", "-o", help="Output directory for PDFs")
    parser.add_argument("--start-timestamp", type=float, help="Start timestamp (Unix)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 6], help="Figure size")
    parser.add_argument("--log-scale", action="store_true", help="Use log-log scale for distributions")
    parser.add_argument("--primary-color", default="#2E86AB", help="Primary color for histogram bars")
    parser.add_argument("--hist-color", dest="primary_color", help=argparse.SUPPRESS)
    parser.add_argument("--bucket-hours", type=int, default=24, help="Bucket size in hours for histogram plots")
    run_cli(_run, parser)


if __name__ == "__main__":
    main()
