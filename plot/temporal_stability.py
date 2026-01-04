import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import matplotlib

from util.constants import SEPARATOR

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from util.experiment_data import load_experiment_results, extract_temporal_metrics, compute_temporal_stability_stats


def plot_temporal_stability(
    experiment_id: str | List[str],
    jsonl_path: str = "output/results/experiments.jsonl",
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_std: bool = True,
    show_individual_runs: bool = False,
    figsize: tuple = (14, 8),
):
    """
    Plot temporal stability of one or more models across sliding windows.

    Args:
        experiment_id: Single experiment ID or list of experiment IDs to compare
        jsonl_path: Path to experiments.jsonl file
        metrics: List of metrics to plot (default: ndcg@10, recall@10, mrr@1, hit@5)
        output_path: Path to save PDF (default: plot/output/{experiment_id}_temporal_stability.pdf)
        show_std: Whether to show standard deviation bands
        show_individual_runs: Whether to show individual run points
        figsize: Figure size (width, height)
    """
    # Normalize to list
    experiment_ids = [experiment_id] if isinstance(experiment_id, str) else experiment_id

    # Load and process data for all experiments
    all_data = []
    for exp_id in experiment_ids:
        results = load_experiment_results(jsonl_path, exp_id)
        data = extract_temporal_metrics(results, metrics)
        all_data.append(data)

    # Validate that all experiments have compatible window configurations
    if len(all_data) > 1:
        ref_metadata = all_data[0]["metadata"]
        for i, data in enumerate(all_data[1:], 1):
            meta = data["metadata"]
            if (
                meta["window_size"] != ref_metadata["window_size"]
                or meta["window_stride"] != ref_metadata["window_stride"]
                or meta["granularity"] != ref_metadata["granularity"]
            ):
                raise ValueError(
                    f"Experiment '{experiment_ids[i]}' has incompatible window configuration. "
                    f"All experiments must have the same window size, stride, and granularity."
                )

    # Use first experiment's metadata for plot info
    metadata = all_data[0]["metadata"]
    plot_metrics = all_data[0]["metrics"]

    # Ensure metrics is not None
    if plot_metrics is None:
        raise ValueError("No metrics found in experiment data")

    # Set up plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Color palette for multiple experiments
    color_palette = ["#2E86AB", "#F18F01", "#A23B72", "#06A77D", "#D4AF37", "#9B59B6", "#E74C3C", "#16A085", "#F39C12"]

    # Line styles for additional variety
    line_styles = ["-", "--", "-.", ":"]

    # Plot each metric
    for idx, metric in enumerate(plot_metrics):
        ax = axes[idx]

        # Determine if we're using temporal x-axis
        granularity_label = (
            metadata["granularity"].capitalize() if metadata["granularity"] != "unknown" else "Time Unit"
        )
        use_temporal_x = metadata["granularity"] != "unknown"

        # Plot each experiment
        for exp_idx, data in enumerate(all_data):
            windows = data["windows"]
            exp_metadata = data["metadata"]

            # Get window numbers for this experiment
            window_numbers = sorted(windows.keys())

            # Get x-values and labels
            if use_temporal_x:
                x_values = []
                x_labels = []
                for w in window_numbers:
                    info = windows[w]["info"]
                    test_end = info.get("end_unit", 0)
                    x_values.append(test_end)
                    test_range = info.get("test_range", "")
                    x_labels.append(test_range)
            else:
                x_values = window_numbers
                x_labels = [str(w) for w in window_numbers]

            # Extract means and stds
            means = [windows[w]["mean"][metric] for w in window_numbers]
            stds = [windows[w]["std"][metric] for w in window_numbers]

            # Select color and line style
            color = color_palette[exp_idx % len(color_palette)]
            linestyle = line_styles[exp_idx % len(line_styles)] if len(all_data) > len(color_palette) else "-"

            # Create label
            label = exp_metadata["model"] if len(all_data) > 1 else "Mean"

            # Plot mean line
            ax.plot(
                x_values, means, color=color, linewidth=2.5, linestyle=linestyle, marker="o", markersize=6, label=label
            )

            # Plot standard deviation band
            if show_std and any(s > 0 for s in stds):
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.fill_between(x_values, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15, color=color)

            # Plot individual runs
            if show_individual_runs:
                for i, w in enumerate(window_numbers):
                    values = windows[w]["values"][metric]
                    ax.scatter([x_values[i]] * len(values), values, alpha=0.3, s=20, color=color)

        # Formatting (applied after all experiments plotted)
        ax.set_xlabel(f"Time ({granularity_label}s)", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_title(f"{metric.upper()} Over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=9)

        # Set x-tick labels using last experiment's x-values (should be same for all)
        if use_temporal_x and "x_values" in locals() and "x_labels" in locals():
            # Use union of all x-values if experiments have different windows
            all_x_values = set()
            all_x_labels = {}
            for data in all_data:
                windows = data["windows"]
                for w in sorted(windows.keys()):
                    info = windows[w]["info"]
                    x_val = info.get("end_unit", 0)
                    all_x_values.add(x_val)
                    all_x_labels[x_val] = info.get("test_range", "")

            x_values_sorted = sorted(all_x_values)
            x_labels_sorted = [all_x_labels[x] for x in x_values_sorted]

            ax.set_xticks(x_values_sorted)
            ax.set_xticklabels(x_labels_sorted, rotation=45, ha="right", fontsize=8)

            # Only show every other label if too many
            if len(x_values_sorted) > 10:
                for i, label in enumerate(ax.get_xticklabels()):
                    if i % 2 != 0:
                        label.set_visible(False)

    # Add overall title with metadata
    if len(all_data) == 1:
        # Single experiment
        main_title = f"Temporal Stability: {metadata['model']} on {metadata['dataset']} ({metadata['total_windows']} windows, {metadata['runs_per_window']} runs/window)"
        fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.98)

        # Add experiment info as subtitle
        subtitle = (
            f"{metadata['experiment_id']}: {metadata['description']}"
            if metadata["description"]
            else metadata["experiment_id"]
        )
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=9, style="italic", color="#555")
    else:
        # Multiple experiments - comparison
        model_names = [d["metadata"]["model"] for d in all_data]
        main_title = f"Temporal Stability Comparison on {metadata['dataset']} ({metadata['total_windows']} windows, {metadata['runs_per_window']} runs/window)"
        fig.suptitle(main_title, fontsize=13, fontweight="bold", y=0.98)

        # Add model names as subtitle
        subtitle = f"Models: {', '.join(model_names)}"
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=9, style="italic", color="#555")

    # Add window configuration info
    config_text = (
        f"Window: {metadata['window_size']} {metadata['granularity']}s "
        f"(stride: {metadata['window_stride']} {metadata['granularity']}s)"
    )
    fig.text(0.5, 0.02, config_text, ha="center", fontsize=9, color="#666")

    plt.tight_layout(rect=(0, 0.03, 1, 0.90))

    # Save to PDF
    if output_path is None:
        output_dir = Path("plot/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(all_data) == 1:
            save_path = output_dir / f"{experiment_ids[0]}_temporal_stability.pdf"
        else:
            save_path = output_dir / f"comparison_{'_'.join(experiment_ids)}_temporal_stability.pdf"
    else:
        save_path = Path(output_path)

    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

    # Calculate and print stability metrics for each experiment
    print(f"\n{SEPARATOR}")
    if len(all_data) == 1:
        print(f"Temporal Stability Analysis - {metadata['experiment_id']}")
    else:
        print(f"Temporal Stability Comparison")
    print(SEPARATOR)

    for exp_idx, data in enumerate(all_data):
        exp_metadata = data["metadata"]
        windows = data["windows"]

        if len(all_data) > 1:
            print(f"\n[{exp_metadata['model']}] - {exp_metadata['experiment_id']}")
            print(f"{'-'*70}")

        print(f"Model: {exp_metadata['model']} | Dataset: {exp_metadata['dataset']}")
        print(f"Windows: {exp_metadata['total_windows']} | Runs per window: {exp_metadata['runs_per_window']}")

        stability_stats = compute_temporal_stability_stats(windows, plot_metrics)

        for metric in plot_metrics:
            stats = stability_stats[metric]
            print(f"\n  {metric.upper()}:")
            print(f"    Mean across windows: {stats['mean']:.4f}")
            print(f"    Std across windows:  {stats['std']:.4f}")
            print(f"    Coefficient of Variation: {stats['cv']:.2f}%")
            print(f"    Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
            print(f"    Range: {stats['range']:.4f}")

        if exp_idx < len(all_data) - 1:
            print(f"\n{'-'*70}")

    print(f"{SEPARATOR}\n")

    return fig, all_data


def main():
    parser = argparse.ArgumentParser(
        description="Visualize temporal stability of recommendation models",
    )

    parser.add_argument(
        "--experiment-id",
        nargs="+",
        required=True,
        help="Experiment ID(s) to visualize. Multiple IDs will be compared.",
    )
    parser.add_argument(
        "--jsonl-path", default="output/results/experiments.jsonl", help="Path to experiments.jsonl file"
    )
    parser.add_argument("--metrics", nargs="+", help="Metrics to plot (default: ndcg@5 ndcg@10 mrr@5 hit@5)")
    parser.add_argument(
        "--output", "-o", help="Output PDF path (default: plot/output/{experiment_id}_temporal_stability.pdf)"
    )
    parser.add_argument("--show-runs", action="store_true", help="Show individual run points")
    parser.add_argument(
        "--show-std", action="store_true", default=True, help="Show standard deviation bands (default: True)"
    )
    parser.add_argument("--no-std", action="store_true", help="Hide standard deviation bands")
    parser.add_argument("--figsize", nargs=2, type=float, default=[14, 8], help="Figure size (width height)")

    args = parser.parse_args()

    # Handle single vs multiple experiment IDs
    experiment_id = args.experiment_id[0] if len(args.experiment_id) == 1 else args.experiment_id

    try:
        plot_temporal_stability(
            experiment_id=experiment_id,
            jsonl_path=args.jsonl_path,
            metrics=args.metrics,
            output_path=args.output,
            show_std=not args.no_std,
            show_individual_runs=args.show_runs,
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
