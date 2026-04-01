"""Grand mean comparison visualization for recommendation models.

Computes and visualizes grand mean performance averaged across all temporal
windows and experimental runs, showing overall model performance with variance.
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib

from util.constants import DEFAULT_METRICS

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.plot.common import AXIS_LABEL_SIZE, COLORS, LEGEND_FONT_SIZE, PLOT_TITLE_SIZE, get_output_dir, print_header, run_cli
from util.experiment_data import load_experiment_results


def compute_grand_mean_statistics(
    results: List[Dict[str, Any]], metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute grand mean statistics across all windows and runs.

    Returns:
        Dictionary with grand_mean, grand_std, grand_sem, n_points, etc.
    """
    if not results:
        raise ValueError("No results provided")

    first = results[0]
    dataset_name = first.get("run_info", {}).get("dataset", "unknown")
    model_name = first.get("run_info", {}).get("model", "unknown")
    experiment_id = first.get("experiment_id", "unknown")

    # Determine metrics
    if metrics is None:
        sample = first.get("test_results", {})
        if not sample:
            raise ValueError("No test_results found in results")
        metrics = [m for m in DEFAULT_METRICS if m in sample] or list(sample.keys())[:4]
    else:
        # Assert all metrics exist
        sample = first.get("test_results", {})
        for m in metrics:
            if m not in sample:
                raise ValueError(f"Metric '{m}' not found in test_results. Available: {list(sample.keys())}")

    # Collect values
    all_values: Dict[str, List[float]] = {m: [] for m in metrics}
    window_numbers: set[int] = set()

    for result in results:
        test_res = result.get("test_results", {})
        if test_res:
            for m in metrics:
                if m in test_res:
                    all_values[m].append(test_res[m])
            w_info = result.get("window_info", {})
            if w_info and w_info.get("window_number") is not None:
                window_numbers.add(w_info["window_number"])

    # Compute statistics
    statistics = {}
    for m in metrics:
        vals = np.array(all_values[m])
        n = len(vals)
        statistics[m] = {
            "grand_mean": float(np.mean(vals)),
            "grand_std": float(np.std(vals, ddof=1)),
            "grand_sem": float(np.std(vals, ddof=1) / np.sqrt(n)),
            "n_points": n,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    n_windows = len(window_numbers) or 1
    total = len(all_values[metrics[0]])

    return {
        "metrics": statistics,
        "metadata": {
            "experiment_id": experiment_id,
            "model": model_name,
            "dataset": dataset_name,
            "n_windows": n_windows,
            "n_runs_per_window": total / n_windows,
            "total_points": total,
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
) -> Tuple[Figure, List[Dict[str, Any]]]:
    """Plot grand mean comparison across multiple models.

    Returns:
        Tuple of (figure, list of statistics dictionaries)
    """
    # Load and compute statistics
    all_stats = []
    for exp_id in experiment_ids:
        results = load_experiment_results(jsonl_path, exp_id)
        if not results:
            raise ValueError(f"No results found for experiment_id: {exp_id}")
        all_stats.append(compute_grand_mean_statistics(results, metrics))

    plot_metrics = list(all_stats[0]["metrics"].keys())
    n_metrics = len(plot_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = [axes] if n_rows * n_cols == 1 else axes.flatten()

    for metric_idx, metric in enumerate(plot_metrics):
        ax = axes[metric_idx]
        _plot_metric_bars(ax, all_stats, metric, show_error_bars, error_type)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Title
    meta = all_stats[0]["metadata"]
    title = (
        f"Grand Mean Performance on {meta['dataset']}\n"
        f"Averaged across {meta['n_windows']} windows × {meta['n_runs_per_window']:.0f} runs = "
        f"N={meta['total_points']} per model"
    )
    fig.suptitle(title, fontsize=PLOT_TITLE_SIZE, fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0, 0.02, 1, 0.94))

    # Save
    out_dir = get_output_dir()
    save_path = Path(output_path) if output_path else out_dir / f"grand_mean_{'_'.join(experiment_ids)}.pdf"
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    _print_stats_table(all_stats, plot_metrics)

    return fig, all_stats


def _plot_metric_bars(
    ax: Axes,
    all_stats: List[Dict[str, Any]],
    metric: str,
    show_error_bars: bool,
    error_type: str,
) -> None:
    """Plot bar chart for a single metric."""
    models = [s["metadata"]["model"] for s in all_stats]
    means = [s["metrics"][metric]["grand_mean"] for s in all_stats]
    stds = [s["metrics"][metric]["grand_std"] for s in all_stats]
    sems = [s["metrics"][metric]["grand_sem"] for s in all_stats]
    n_pts = [s["metadata"]["total_points"] for s in all_stats]

    errors, label = _get_error_values(stds, sems, error_type)
    x = np.arange(len(models))

    bars = ax.bar(
        x,
        means,
        0.6,
        color=[COLORS[i % len(COLORS)] for i in range(len(models))],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
    )

    if show_error_bars:
        ax.errorbar(x, means, yerr=errors, fmt="none", color="black", capsize=5, capthick=1.5, alpha=0.7)
        ax.text(
            0.02,
            0.98,
            f"Error: {label}",
            transform=ax.transAxes,
            fontsize=LEGEND_FONT_SIZE - 1,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    for bar, mean, n in zip(bars, means, n_pts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.4f}\n(N={n})",
            ha="center",
            va="bottom",
            fontsize=LEGEND_FONT_SIZE - 1,
        )

    ax.set_ylabel(metric.upper(), fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"{metric.upper()} - Grand Mean", fontsize=PLOT_TITLE_SIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Y-axis limits
    err_max = max(errors) if show_error_bars else 0
    y_min, y_max = min(means) - err_max, max(means) + err_max
    pad = (y_max - y_min) * 0.15
    ax.set_ylim(max(0, y_min - pad), y_max + pad)


def _get_error_values(stds: List[float], sems: List[float], error_type: str) -> Tuple[List[float], str]:
    """Get error bar values based on type."""
    if error_type == "std":
        return stds, "±1 SD"
    elif error_type == "sem":
        return sems, "±1 SEM"
    elif error_type == "ci95":
        return [1.96 * s for s in sems], "95% CI"
    raise ValueError(f"Unknown error_type: {error_type}")


def _print_stats_table(all_stats: List[Dict[str, Any]], metrics: List[str]) -> None:
    """Print statistics table."""
    meta = all_stats[0]["metadata"]
    print_header(f"Grand Mean Performance - {meta['dataset']}")
    print(f"N = {meta['n_windows']} windows × {meta['n_runs_per_window']:.0f} runs = {meta['total_points']}")

    print(f"\n{'Model':<20} | {'Metric':<12} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 80)

    for stats in all_stats:
        model = stats["metadata"]["model"]
        for i, m in enumerate(metrics):
            s = stats["metrics"][m]
            lbl = model if i == 0 else ""
            print(
                f"{lbl:<20} | {m:<12} | {s['grand_mean']:>8.4f} | {s['grand_std']:>8.4f} | {s['min']:>8.4f} | {s['max']:>8.4f}"
            )
        if stats != all_stats[-1]:
            print("-" * 80)

    # Rankings
    if len(all_stats) > 1:
        print("\nRankings by metric:")
        for m in metrics:
            ranked = sorted(
                [(s["metadata"]["model"], s["metrics"][m]["grand_mean"]) for s in all_stats],
                key=lambda x: x[1],
                reverse=True,
            )
            print(f"  {m.upper()}: {', '.join(f'{r[0]}={r[1]:.4f}' for r in ranked)}")


def _run(args: argparse.Namespace) -> None:
    if len(args.experiment_id) < 2:
        print("Error: At least 2 experiment IDs required")
        exit(1)
    plot_grand_mean_comparison(
        experiment_ids=args.experiment_id,
        jsonl_path=args.jsonl_path,
        metrics=args.metrics,
        output_path=args.output,
        show_error_bars=not args.no_error_bars,
        error_type=args.error_type,
        figsize=tuple(args.figsize),
    )


def main():
    parser = argparse.ArgumentParser(description="Compare grand mean performance across models")
    parser.add_argument("--experiment-id", nargs="+", required=True, help="Experiment IDs to compare")
    parser.add_argument("--jsonl-path", default="output/results/experiments.jsonl")
    parser.add_argument("--metrics", nargs="+", help="Metrics to plot")
    parser.add_argument("--output", "-o", help="Output PDF path")
    parser.add_argument("--error-type", choices=["std", "sem", "ci95"], default="std")
    parser.add_argument("--no-error-bars", action="store_true")
    parser.add_argument("--figsize", nargs=2, type=float, default=[14, 8])

    run_cli(_run, parser)


if __name__ == "__main__":
    main()
