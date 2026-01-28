"""Sliding window validation and statistics visualization.

Validates that the sliding window methodology functioned as intended by analyzing:
- Data distribution across windows
- Cold-start ratios (new items/users in test)
- Train-test overlap statistics
- Volume stability assessment
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot.common import get_output_dir, print_header, run_cli
from util.experiment_data import load_experiment_results
from util.dataset_analysis import load_temporal_interaction_data
from util.window_validation import compute_all_window_statistics, generate_validation_report, export_statistics_table
from util.window_plots import plot_window_data_distribution, plot_cold_start_ratios


def validate_sliding_windows(
    experiment_id: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    dataset_path: str = "data/atomic_files",
    output_dir: Optional[str] = None,
    generate_plots: bool = True,
    export_tables: bool = True,
) -> dict:
    """Generate comprehensive validation for sliding window methodology.

    Args:
        experiment_id: Experiment ID to validate
        jsonl_path: Path to experiments.jsonl file
        dataset_path: Path to dataset directory
        output_dir: Output directory (default: plot/output/)
        generate_plots: Whether to generate plots
        export_tables: Whether to export statistics tables

    Returns:
        Dictionary with validation results and output paths
    """
    results = load_experiment_results(jsonl_path, experiment_id)
    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")

    first_result = results[0]
    window_info = first_result.get("window_info")
    if not window_info:
        raise ValueError(f"Experiment {experiment_id} has no window_info - not a temporal experiment")

    dataset_name = first_result.get("run_info", {}).get("dataset", "unknown")
    granularity = window_info.get("granularity", "hour")

    # Collect unique windows
    all_windows = _collect_unique_windows(results)
    print(f"Validating {len(all_windows)} windows from experiment {experiment_id}...")

    # Load interaction data
    min_unit = min(w.get("start_unit", 0) for w in all_windows)
    max_unit = max(w.get("end_unit", 0) for w in all_windows)
    print(f"Loading {dataset_name} data ({granularity} {min_unit} to {max_unit})...")

    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1),
    )
    print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

    # Compute and generate outputs
    print("Computing window statistics...")
    stats_df = compute_all_window_statistics(df, all_windows, granularity)

    out_dir = Path(output_dir) if output_dir else get_output_dir()
    output_paths = _generate_outputs(
        stats_df, experiment_id, dataset_name, granularity, out_dir, generate_plots, export_tables
    )

    print_header("Validation Complete")
    return {
        "statistics": stats_df,
        "output_paths": output_paths,
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "num_windows": len(all_windows),
    }


def _collect_unique_windows(results: list) -> list:
    """Deduplicate windows by window_number (multiple runs per window)."""
    windows_by_num = {}
    for result in results:
        w_info = result.get("window_info", {})
        if w_info:
            win_num = w_info.get("window_number")
            if win_num is not None and win_num not in windows_by_num:
                windows_by_num[win_num] = w_info
    return [windows_by_num[k] for k in sorted(windows_by_num.keys())]


def _generate_outputs(
    stats_df,
    experiment_id: str,
    dataset_name: str,
    granularity: str,
    out_dir: Path,
    generate_plots: bool,
    export_tables: bool,
) -> dict:
    """Generate all output files: report, tables, and plots."""
    output_paths = {}

    # Text report
    print("\nGenerating validation report...")
    report = generate_validation_report(stats_df, experiment_id, dataset_name, granularity)
    print("\n" + report)

    report_path = out_dir / f"{experiment_id}_window_validation.txt"
    report_path.write_text(report)
    print(f"Saved: {report_path}")
    output_paths["report"] = str(report_path)

    # Tables
    if export_tables:
        print("\nExporting statistics tables...")
        for fmt, ext in [("latex", ".tex"), ("csv", ".csv"), ("markdown", ".md")]:
            path = out_dir / f"{experiment_id}_window_stats{ext}"
            export_statistics_table(stats_df, str(path), format=fmt)
            print(f"Saved {fmt}: {path}")
            output_paths[f"{fmt}_table"] = str(path)

    # Plots
    if generate_plots:
        print("\nGenerating validation plots...")
        for plot_fn, name in [
            (plot_window_data_distribution, "data_distribution"),
            (plot_cold_start_ratios, "cold_start"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_fn(stats_df, ax=ax)
            path = out_dir / f"{experiment_id}_{name}.pdf"
            fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")
            output_paths[f"{name}_plot"] = str(path)

    return output_paths


def _run(args: argparse.Namespace) -> None:
    validate_sliding_windows(
        experiment_id=args.experiment_id,
        jsonl_path=args.jsonl_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        export_tables=not args.no_tables,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate sliding window methodology and generate statistics")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to validate")
    parser.add_argument("--jsonl-path", default="output/results/experiments.jsonl")
    parser.add_argument("--dataset-path", default="data/atomic_files")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-tables", action="store_true", help="Skip table export")

    run_cli(_run, parser)


if __name__ == "__main__":
    main()
