"""
Grand mean comparison visualization for recommendation models.

This module provides functions to visualize the grand mean performance across
all temporal windows and experimental runs, showing overall model performance
with variance. Each model's performance is averaged across all N data points
(windows × runs per window).
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib

from util.constants import SEPARATOR

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from util.experiment_data import (
    load_experiment_results,
)


def compute_grand_mean_statistics(
    results: List[Dict[str, Any]], metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute grand mean statistics across all windows and runs.

    Args:
        results: List of experiment results from load_experiment_results
        metrics: List of metrics to compute (default: common metrics)

    Returns:
        Dictionary with grand mean statistics including:
        - grand_mean: Mean across all data points
        - grand_std: Standard deviation across all data points
        - grand_sem: Standard error of the mean
        - n_points: Total number of data points
        - n_windows: Number of unique windows
        - n_runs_per_window: Average runs per window
    """
    if not results:
        raise ValueError("No results provided")

    # Get metadata from first result
    first_result = results[0]
    dataset_name = first_result.get("run_info", {}).get("dataset", "unknown")
    model_name = first_result.get("run_info", {}).get("model", "unknown")
    experiment_id = first_result.get("experiment_id", "unknown")

    # Determine which metrics to use
    if metrics is None:
        # Try to get from test_results
        sample_metrics = first_result.get("test_results", {})
        if sample_metrics:
            # Use common metrics if available
            common_metrics = ["ndcg@10", "recall@10", "hit@10", "mrr@10"]
            metrics = [m for m in common_metrics if m in sample_metrics]
            if not metrics:
                # Fall back to first 4 metrics
                metrics = list(sample_metrics.keys())[:4]
        else:
            raise ValueError("No test_results found in results")

    # Collect all values for each metric
    all_values = {metric: [] for metric in metrics}
    window_numbers = set()

    for result in results:
        test_res = result.get("test_results", {})
        if test_res:
            for metric in metrics:
                if metric in test_res:
                    all_values[metric].append(test_res[metric])

            # Track window numbers if available
            window_info = result.get("window_info", {})
            if window_info:
                window_numbers.add(window_info.get("window_number"))

    # Compute statistics
    statistics = {}
    for metric in metrics:
        values = np.array(all_values[metric])
        statistics[metric] = {
            "grand_mean": np.mean(values),
            "grand_std": np.std(values, ddof=1),  # Sample std
            "grand_sem": np.std(values, ddof=1) / np.sqrt(len(values)),
            "grand_var": np.var(values, ddof=1),
            "n_points": len(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
        }

    # Compute runs per window
    n_windows = len(window_numbers) if window_numbers else 1
    total_points = len(all_values[metrics[0]])
    runs_per_window = total_points / n_windows if n_windows > 0 else total_points

    return {
        "metrics": statistics,
        "metadata": {
            "experiment_id": experiment_id,
            "model": model_name,
            "dataset": dataset_name,
            "n_windows": n_windows,
            "n_runs_per_window": runs_per_window,
            "total_points": total_points,
        },
    }


def plot_grand_mean_comparison(
    experiment_ids: List[str],
    jsonl_path: str = "output/results/experiments.jsonl",
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_error_bars: bool = True,
    error_type: str = "std",
    figsize: Tuple[int, int] = (14, 8),
) -> Tuple[plt.Figure, List[Dict]]:
    """
    Plot grand mean comparison across multiple models.

    Args:
        experiment_ids: List of experiment IDs to compare
        jsonl_path: Path to experiments.jsonl file
        metrics: List of metrics to plot (default: ndcg@10, recall@10, hit@10, mrr@10)
        output_path: Path to save PDF
        show_error_bars: Whether to show error bars
        error_type: Type of error bar ('std', 'sem', or 'ci95')
        figsize: Figure size (width, height)

    Returns:
        Tuple of (figure, list of statistics dictionaries)
    """
    # Load and compute statistics for all experiments
    all_stats = []
    for exp_id in experiment_ids:
        results = load_experiment_results(jsonl_path, exp_id)
        if not results:
            raise ValueError(f"No results found for experiment_id: {exp_id}")

        stats = compute_grand_mean_statistics(results, metrics)
        all_stats.append(stats)

    # Get metrics to plot (use first experiment's metrics)
    plot_metrics = list(all_stats[0]["metrics"].keys())

    # Set up plot
    n_metrics = len(plot_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette
    color_palette = ["#2E86AB", "#F18F01", "#A23B72", "#06A77D", "#D4AF37", "#9B59B6", "#E74C3C", "#16A085"]

    # Plot each metric
    for metric_idx, metric in enumerate(plot_metrics):
        ax = axes[metric_idx]

        # Prepare data
        model_names = [stats["metadata"]["model"] for stats in all_stats]
        means = [stats["metrics"][metric]["grand_mean"] for stats in all_stats]
        stds = [stats["metrics"][metric]["grand_std"] for stats in all_stats]
        sems = [stats["metrics"][metric]["grand_sem"] for stats in all_stats]
        n_points = [stats["metadata"]["total_points"] for stats in all_stats]

        # Calculate error bars
        if error_type == "std":
            errors = stds
            error_label = "±1 SD"
        elif error_type == "sem":
            errors = sems
            error_label = "±1 SEM"
        elif error_type == "ci95":
            # 95% confidence interval: mean ± 1.96 * SEM
            errors = [1.96 * sem for sem in sems]
            error_label = "95% CI"
        else:
            raise ValueError(f"Unknown error_type: {error_type}")

        x = np.arange(len(model_names))
        width = 0.6

        # Create bars
        bars = ax.bar(
            x,
            means,
            width,
            color=[color_palette[i % len(color_palette)] for i in range(len(model_names))],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        # Add error bars
        if show_error_bars:
            ax.errorbar(
                x, means, yerr=errors, fmt="none", color="black", capsize=5, capthick=1.5, linewidth=1.5, alpha=0.7
            )

        # Add value labels on bars
        for i, (bar, mean, n) in enumerate(zip(bars, means, n_points)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{mean:.4f}\n(N={n})",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        # Formatting
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_title(f"{metric.upper()} - Grand Mean Across All Windows & Runs", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Add error bar legend
        if show_error_bars:
            ax.text(
                0.02,
                0.98,
                f"Error bars: {error_label}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Set y-axis to start from 0 or slightly below minimum
        y_min = min(means) - max(errors) if show_error_bars else min(means)
        y_max = max(means) + max(errors) if show_error_bars else max(means)
        y_padding = (y_max - y_min) * 0.15
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    dataset_name = all_stats[0]["metadata"]["dataset"]
    n_windows = all_stats[0]["metadata"]["n_windows"]
    runs_per_window = all_stats[0]["metadata"]["n_runs_per_window"]
    total_points = all_stats[0]["metadata"]["total_points"]

    main_title = (
        f"Grand Mean Performance Comparison on {dataset_name}\n"
        f"Averaged across {n_windows} windows × {runs_per_window:.0f} runs = "
        f"N={total_points} data points per model"
    )
    fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout(rect=(0, 0.02, 1, 0.94))

    # Save to PDF
    if output_path is None:
        output_dir = Path("plot/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"grand_mean_comparison_{'_'.join(experiment_ids)}.pdf"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print statistics table
    print(f"\n{SEPARATOR}")
    print(f"Grand Mean Performance - {dataset_name}")
    print(f"{SEPARATOR}")
    print(f"Data points per model: N = {n_windows} windows × {runs_per_window:.0f} runs = {total_points}")
    print(f"{SEPARATOR}\n")

    # Table header
    header = f"{'Model':<20} | {'Metric':<12} | {'Mean':<8} | {'Std':<8} | {'SEM':<8} | {'Min':<8} | {'Max':<8}"
    print(header)
    print("-" * 90)

    # Print data for each model
    for stats in all_stats:
        model = stats["metadata"]["model"]
        for metric_idx, metric in enumerate(plot_metrics):
            metric_stats = stats["metrics"][metric]

            model_label = model if metric_idx == 0 else ""

            row = (
                f"{model_label:<20} | {metric:<12} | "
                f"{metric_stats['grand_mean']:>8.4f} | "
                f"{metric_stats['grand_std']:>8.4f} | "
                f"{metric_stats['grand_sem']:>8.4f} | "
                f"{metric_stats['min']:>8.4f} | "
                f"{metric_stats['max']:>8.4f}"
            )
            print(row)

        if stats != all_stats[-1]:
            print("-" * 90)

    print(SEPARATOR)

    # Statistical comparison
    if len(all_stats) > 1:
        print(f"\n{SEPARATOR}")
        print("Pairwise Comparisons")
        print(f"{SEPARATOR}\n")

        for metric in plot_metrics:
            print(f"{metric.upper()}:")

            # Sort models by performance
            model_means = [(stats["metadata"]["model"], stats["metrics"][metric]["grand_mean"]) for stats in all_stats]
            model_means.sort(key=lambda x: x[1], reverse=True)

            for rank, (model, mean) in enumerate(model_means, 1):
                print(f"  {rank}. {model:<20} {mean:.4f}")

            # Calculate relative differences from best
            best_mean = model_means[0][1]
            print(f"\n  Relative to best ({model_means[0][0]}):")
            for model, mean in model_means[1:]:
                rel_diff = ((mean - best_mean) / best_mean) * 100
                print(f"    {model:<20} {rel_diff:>+6.2f}%")
            print()

        print(SEPARATOR)

    print()

    return fig, all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compare grand mean performance across models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare multiple models
  python -m plot.grand_mean_comparison --experiment-id exp_tfidf exp_fasttext exp_pop exp_random
  
  # Custom metrics
  python -m plot.grand_mean_comparison --experiment-id exp001 exp002 \\
      --metrics ndcg@10 recall@20
  
  # Show 95% confidence intervals instead of standard deviation
  python -m plot.grand_mean_comparison --experiment-id exp001 exp002 \\
      --error-type ci95
  
  # Custom output path
  python -m plot.grand_mean_comparison --experiment-id exp001 exp002 \\
      --output my_comparison.pdf

Purpose:
  This script computes and visualizes the grand mean performance averaged
  across all temporal windows and all experimental runs. For example, with
  9 windows and 3 runs per window, each model has N=27 data points.
  
  The plot shows:
  - Bar chart of mean performance for each model
  - Error bars (standard deviation, SEM, or 95% CI)
  - Sample size (N) for each model
  - Statistical comparison table
        """,
    )

    parser.add_argument(
        "--experiment-id", nargs="+", required=True, help="Experiment IDs to compare (must be 2 or more)"
    )
    parser.add_argument(
        "--jsonl-path", default="output/results/experiments.jsonl", help="Path to experiments.jsonl file"
    )
    parser.add_argument("--metrics", nargs="+", help="Metrics to plot (default: ndcg@10 recall@10 hit@10 mrr@10)")
    parser.add_argument("--output", "-o", help="Output PDF path")
    parser.add_argument(
        "--error-type",
        choices=["std", "sem", "ci95"],
        default="std",
        help="Type of error bars: std (standard deviation), sem (standard error), ci95 (95%% CI)",
    )
    parser.add_argument("--no-error-bars", action="store_true", help="Hide error bars")
    parser.add_argument("--figsize", nargs=2, type=float, default=[14, 8], help="Figure size (width height)")

    args = parser.parse_args()

    if len(args.experiment_id) < 2:
        print("Error: At least 2 experiment IDs are required for comparison")
        exit(1)

    try:
        plot_grand_mean_comparison(
            experiment_ids=args.experiment_id,
            jsonl_path=args.jsonl_path,
            metrics=args.metrics,
            output_path=args.output,
            show_error_bars=not args.no_error_bars,
            error_type=args.error_type,
            figsize=tuple(args.figsize),
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure {args.jsonl_path} exists")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
