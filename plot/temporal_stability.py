"""Temporal stability visualization for recommendation models.

Plots model performance across sliding windows to analyze stability over time.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from plot.common import get_output_dir, print_header, run_cli, COLORS
from util.experiment_data import load_experiment_results, extract_temporal_metrics, compute_temporal_stability_stats


def plot_temporal_stability(
    experiment_id: str | List[str],
    jsonl_path: str = "output/results/experiments.jsonl",
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_std: bool = True,
    show_individual_runs: bool = False,
    figsize: tuple = (14, 8),
) -> tuple[Figure, List[Dict[str, Any]]]:
    """Plot temporal stability of one or more models across sliding windows.

    Returns:
        Tuple of (figure, list of experiment data)
    """
    experiment_ids = [experiment_id] if isinstance(experiment_id, str) else experiment_id
    all_data = [extract_temporal_metrics(load_experiment_results(jsonl_path, eid), metrics) for eid in experiment_ids]

    if len(all_data) > 1:
        _validate_compatible_configs(all_data, experiment_ids)

    metadata = all_data[0]["metadata"]
    plot_metrics = all_data[0]["metrics"]
    if not plot_metrics:
        raise ValueError("No metrics found in experiment data")

    n_metrics = len(plot_metrics)
    ncols = min(n_metrics, 2)
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(plot_metrics):
        _plot_metric_lines(axes_flat[idx], all_data, metric, show_std, show_individual_runs)

    # Title
    _add_figure_titles(fig, all_data, metadata, experiment_ids)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))

    # Save
    out_dir = get_output_dir()
    if output_path:
        save_path = Path(output_path)
    elif len(all_data) == 1:
        save_path = out_dir / f"{experiment_ids[0]}_temporal_stability.pdf"
    else:
        save_path = out_dir / f"comparison_{'_'.join(experiment_ids)}_temporal.pdf"

    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    _print_stability_stats(all_data, plot_metrics)

    return fig, all_data


def _validate_compatible_configs(all_data: List[Dict], experiment_ids: List[str]) -> None:
    """Ensure all experiments have compatible window configurations."""
    ref = all_data[0]["metadata"]
    for i, data in enumerate(all_data[1:], 1):
        meta = data["metadata"]
        if (
            meta["window_size"] != ref["window_size"]
            or meta["window_stride"] != ref["window_stride"]
            or meta["granularity"] != ref["granularity"]
        ):
            raise ValueError(f"Experiment '{experiment_ids[i]}' has incompatible window configuration")


def _plot_metric_lines(
    ax: Axes,
    all_data: List[Dict[str, Any]],
    metric: str,
    show_std: bool,
    show_individual_runs: bool,
) -> None:
    """Plot lines for a single metric across all experiments."""
    metadata = all_data[0]["metadata"]
    granularity = metadata["granularity"]
    use_temporal = granularity != "unknown"

    for exp_idx, data in enumerate(all_data):
        windows = data["windows"]
        window_nums = sorted(windows.keys())

        # X-axis values
        if use_temporal:
            x_vals = [windows[w]["info"].get("end_unit", 0) for w in window_nums]
        else:
            x_vals = list(window_nums)

        means = [windows[w]["mean"][metric] for w in window_nums]
        stds = [windows[w]["std"][metric] for w in window_nums]

        color = COLORS[exp_idx % len(COLORS)]
        label = data["metadata"]["model"] if len(all_data) > 1 else "Mean"

        ax.plot(x_vals, means, color=color, linewidth=2.5, marker="o", markersize=6, label=label)

        if show_std and any(s > 0 for s in stds):
            means_arr, stds_arr = np.array(means), np.array(stds)
            ax.fill_between(x_vals, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15, color=color)

        if show_individual_runs:
            for i, w in enumerate(window_nums):
                vals = windows[w]["values"][metric]
                ax.scatter([x_vals[i]] * len(vals), vals, alpha=0.3, s=20, color=color)

    # Formatting
    gran_label = granularity.capitalize() if use_temporal else "Window"
    ax.set_xlabel(f"Time ({gran_label}s)", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric.upper(), fontsize=11, fontweight="bold")
    ax.set_title(f"{metric.upper()} Over Time", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)


def _add_figure_titles(fig: Figure, all_data: List[Dict], metadata: Dict, exp_ids: List[str]) -> None:
    """Add title and subtitle to figure."""
    n_win, runs = metadata["total_windows"], metadata["runs_per_window"]

    if len(all_data) == 1:
        title = (
            f"Temporal Stability: {metadata['model']} on {metadata['dataset']} ({n_win} windows, {runs} runs/window)"
        )
        subtitle = metadata.get("description") or metadata["experiment_id"]
    else:
        models = ", ".join(d["metadata"]["model"] for d in all_data)
        title = f"Temporal Stability Comparison on {metadata['dataset']} ({n_win} windows, {runs} runs/window)"
        subtitle = f"Models: {models}"

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=9, style="italic", color="#555")

    config = f"Window: {metadata['window_size']} {metadata['granularity']}s (stride: {metadata['window_stride']})"
    fig.text(0.5, 0.02, config, ha="center", fontsize=9, color="#666")


def _print_stability_stats(all_data: List[Dict], metrics: List[str]) -> None:
    """Print stability statistics for each experiment."""
    title = "Temporal Stability Analysis" if len(all_data) == 1 else "Temporal Stability Comparison"
    print_header(title)

    for data in all_data:
        meta = data["metadata"]
        windows = data["windows"]

        if len(all_data) > 1:
            print(f"\n[{meta['model']}] - {meta['experiment_id']}")

        print(f"Model: {meta['model']} | Dataset: {meta['dataset']}")
        print(f"Windows: {meta['total_windows']} | Runs/window: {meta['runs_per_window']}")

        stats = compute_temporal_stability_stats(windows, metrics)
        for m in metrics:
            s = stats[m]
            print(f"\n  {m.upper()}: Mean={s['mean']:.4f}, Std={s['std']:.4f}, CV={s['cv']:.2f}%")


def _run(args: argparse.Namespace) -> None:
    experiment_id = args.experiment_id[0] if len(args.experiment_id) == 1 else args.experiment_id
    plot_temporal_stability(
        experiment_id=experiment_id,
        jsonl_path=args.jsonl_path,
        metrics=args.metrics,
        output_path=args.output,
        show_std=not args.no_std,
        show_individual_runs=args.show_runs,
        figsize=tuple(args.figsize),
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize temporal stability of recommendation models")
    parser.add_argument("--experiment-id", nargs="+", required=True, help="Experiment ID(s) to visualize")
    parser.add_argument("--jsonl-path", default="output/results/experiments.jsonl")
    parser.add_argument("--metrics", nargs="+", help="Metrics to plot")
    parser.add_argument("--output", "-o", help="Output PDF path")
    parser.add_argument("--show-runs", action="store_true", help="Show individual run points")
    parser.add_argument("--no-std", action="store_true", help="Hide standard deviation bands")
    parser.add_argument("--figsize", nargs=2, type=float, default=[14, 8])

    run_cli(_run, parser)


if __name__ == "__main__":
    main()
