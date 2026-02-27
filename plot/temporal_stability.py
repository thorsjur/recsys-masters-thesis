"""Temporal stability visualization for recommendation models.

Plots model performance across sliding windows to analyze stability over time.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from plot.common import get_output_dir, print_header, run_cli, COLORS
from util.experiment_data import load_experiment_results, extract_temporal_metrics, compute_temporal_stability_stats

# Marker cycle for distinguishing multiple experiments
MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Line style cycle for additional visual distinction
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

COLOR_PALETTES = {
    "default": COLORS,
    "muted": ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB", "#E8A667", "#92C6FF"],
    "vibrant": ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653", "#A8DADC", "#1D3557"],
    "pastel": ["#A8D8EA", "#AA96DA", "#FCBAD3", "#FFFFD2", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#E2F0CB"],
    "colorblind": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"],
}


def plot_temporal_stability(
    experiment_id: str | List[str],
    jsonl_path: str = "output/results/experiments.jsonl",
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    show_std: bool = True,
    show_individual_runs: bool = False,
    figsize: tuple = (14, 8),
    line_width: float = 2.5,
    marker_size: float = 6.0,
    vary_markers: bool = False,
    vary_line_styles: bool = False,
    font_size: float = 11.0,
    title_size: float = 13.0,
    legend_size: float = 9.0,
    legend_loc: str = "best",
    legend_alpha: float = 0.8,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    grid_style: str = "--",
    std_alpha: float = 0.15,
    show_trend_line: bool = False,
    dpi: int = 300,
    output_format: str = "pdf",
    color_palette: str = "default",
    custom_title: Optional[str] = None,
    no_title: bool = False,
    dark_mode: bool = False,
    x_label: Optional[str] = None,
    y_label_suffix: Optional[str] = None,
) -> tuple[Figure, List[Dict[str, Any]]]:
    """Plot temporal stability of one or more models across sliding windows.

    Returns:
        Tuple of (figure, list of experiment data)
    """
    experiment_ids = [experiment_id] if isinstance(experiment_id, str) else experiment_id
    all_data = [extract_temporal_metrics(load_experiment_results(jsonl_path, eid), metrics) for eid in experiment_ids]

    # Commented out to allow comparison of different window configs
    # if len(all_data) > 1:
    #     _validate_compatible_configs(all_data, experiment_ids)

    metadata = all_data[0]["metadata"]
    plot_metrics = all_data[0]["metrics"]
    if not plot_metrics:
        raise ValueError("No metrics found in experiment data")

    colors = COLOR_PALETTES.get(color_palette, COLOR_PALETTES["default"])

    if dark_mode:
        plt.style.use("dark_background")

    style_opts = dict(
        line_width=line_width,
        marker_size=marker_size,
        vary_markers=vary_markers,
        vary_line_styles=vary_line_styles,
        font_size=font_size,
        title_size=title_size,
        legend_size=legend_size,
        legend_loc=legend_loc,
        legend_alpha=legend_alpha,
        show_grid=show_grid,
        grid_alpha=grid_alpha,
        grid_style=grid_style,
        std_alpha=std_alpha,
        show_trend_line=show_trend_line,
        colors=colors,
        x_label=x_label,
        y_label_suffix=y_label_suffix,
    )

    n_metrics = len(plot_metrics)
    ncols = min(n_metrics, 2)
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(plot_metrics):
        _plot_metric_lines(axes_flat[idx], all_data, metric, show_std, show_individual_runs, style_opts)

    # Hide unused axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Title
    if not no_title:
        _add_figure_titles(fig, all_data, metadata, experiment_ids, title_size, font_size, custom_title)
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))

    # Save
    out_dir = get_output_dir()
    ext = output_format
    if output_path:
        save_path = Path(output_path)
    elif len(all_data) == 1:
        save_path = out_dir / f"{experiment_ids[0]}_temporal_stability.{ext}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = out_dir / f"comparison_{timestamp}_temporal.{ext}"

    plt.savefig(save_path, format=output_format, dpi=dpi, bbox_inches="tight")
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
    style_opts: Dict[str, Any],
) -> None:
    """Plot lines for a single metric across all experiments."""
    metadata = all_data[0]["metadata"]
    granularity = metadata["granularity"]
    use_temporal = granularity != "unknown"

    colors = style_opts["colors"]
    lw = style_opts["line_width"]
    ms = style_opts["marker_size"]
    vary_markers = style_opts["vary_markers"]
    vary_ls = style_opts["vary_line_styles"]
    std_alpha = style_opts["std_alpha"]
    show_trend_line = style_opts["show_trend_line"]
    font_size = style_opts["font_size"]

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

        color = colors[exp_idx % len(colors)]
        marker = MARKER_STYLES[exp_idx % len(MARKER_STYLES)] if vary_markers else "o"
        linestyle = LINE_STYLES[exp_idx % len(LINE_STYLES)] if vary_ls else "-"
        label = data["metadata"]["model"] if len(all_data) > 1 else "Mean"

        ax.plot(
            x_vals, means,
            color=color, linewidth=lw, linestyle=linestyle,
            marker=marker, markersize=ms, label=label,
        )

        if show_std and any(s > 0 for s in stds):
            means_arr, stds_arr = np.array(means), np.array(stds)
            ax.fill_between(x_vals, means_arr - stds_arr, means_arr + stds_arr, alpha=std_alpha, color=color)

        if show_trend_line and len(x_vals) >= 2:
            coeffs = np.polyfit(x_vals, means, 1)
            trend_y = np.polyval(coeffs, x_vals)
            ax.plot(
                x_vals, trend_y,
                color=color, linewidth=1.0, linestyle="--", alpha=0.5,
            )

        if show_individual_runs:
            run_marker = marker if vary_markers else "o"
            for i, w in enumerate(window_nums):
                vals = windows[w]["values"][metric]
                ax.scatter([x_vals[i]] * len(vals), vals, alpha=0.3, s=ms * 3.3, color=color, marker=run_marker)

    # Formatting
    gran_label = granularity.capitalize() if use_temporal else "Window"
    x_lbl = style_opts["x_label"] or f"Time ({gran_label}s)"
    y_lbl = f"{metric.upper()}{style_opts['y_label_suffix'] or ''}"
    ax.set_xlabel(x_lbl, fontsize=font_size, fontweight="bold")
    ax.set_ylabel(y_lbl, fontsize=font_size, fontweight="bold")
    ax.set_title(f"{metric.upper()} Over Time", fontsize=font_size + 1, fontweight="bold")
    if style_opts["show_grid"]:
        ax.grid(True, alpha=style_opts["grid_alpha"], linestyle=style_opts["grid_style"])
    else:
        ax.grid(False)
    legend = ax.legend(loc=style_opts["legend_loc"], fontsize=style_opts["legend_size"])
    if legend:
        legend.get_frame().set_alpha(style_opts["legend_alpha"])


def _add_figure_titles(
    fig: Figure,
    all_data: List[Dict],
    metadata: Dict,
    exp_ids: List[str],
    title_size: float = 13.0,
    font_size: float = 11.0,
    custom_title: Optional[str] = None,
) -> None:
    """Add title and subtitle to figure."""
    n_win, runs = metadata["total_windows"], metadata["runs_per_window"]

    if custom_title:
        title = custom_title
        subtitle = ""
    elif len(all_data) == 1:
        title = (
            f"Temporal Stability: {metadata['model']} on {metadata['dataset']} ({n_win} windows, {runs} runs/window)"
        )
        subtitle = metadata.get("description") or metadata["experiment_id"]
    else:
        models = ", ".join(d["metadata"]["model"] for d in all_data)
        title = f"Temporal Stability Comparison on {metadata['dataset']} ({n_win} windows, {runs} runs/window)"
        subtitle = f"Models: {models}"

    subtitle_size = max(font_size - 2, 7)
    fig.suptitle(title, fontsize=title_size, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", fontsize=subtitle_size, style="italic", color="#555")

    config = f"Window: {metadata['window_size']} {metadata['granularity']}s (stride: {metadata['window_stride']})"
    fig.text(0.5, 0.02, config, ha="center", fontsize=subtitle_size, color="#666")


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
        show_trend_line=args.show_trend,
        figsize=tuple(args.figsize),
        line_width=args.line_width,
        marker_size=args.marker_size,
        vary_markers=args.vary_markers,
        vary_line_styles=args.vary_line_styles,
        font_size=args.font_size,
        title_size=args.title_size,
        legend_size=args.legend_size,
        legend_loc=args.legend_loc,
        legend_alpha=args.legend_alpha,
        show_grid=not args.no_grid,
        grid_alpha=args.grid_alpha,
        grid_style=args.grid_style,
        std_alpha=args.std_alpha,
        dpi=args.dpi,
        output_format=args.format,
        color_palette=args.color_palette,
        custom_title=args.title,
        no_title=args.no_title,
        dark_mode=args.dark_mode,
        x_label=args.x_label,
        y_label_suffix=args.y_label_suffix,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize temporal stability of recommendation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --experiment-id exp1 --vary-markers --line-width 3
  %(prog)s --experiment-id exp1 exp2 --vary-markers --vary-line-styles --color-palette colorblind
  %(prog)s --experiment-id exp1 --dark-mode --format png --dpi 150
  %(prog)s --experiment-id exp1 --title "My Plot" --no-grid --font-size 14
""",
    )

    # --- Data selection ---
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--experiment-id", nargs="+", required=True, help="Experiment ID(s) to visualize")
    data_group.add_argument("--jsonl-path", default="output/results/experiments.jsonl", help="Path to JSONL results file")
    data_group.add_argument("--metrics", nargs="+", help="Metrics to plot (default: all available)")

    # --- Line & marker styling ---
    style_group = parser.add_argument_group("line & marker style")
    style_group.add_argument("--line-width", "--lw", type=float, default=2.5, help="Line width (default: 2.5)")
    style_group.add_argument("--marker-size", "--ms", type=float, default=6.0, help="Marker size (default: 6.0)")
    style_group.add_argument(
        "--vary-markers", action="store_true",
        help="Use different marker shapes (o, s, ^, D, v, P, X, *) per experiment",
    )
    style_group.add_argument(
        "--vary-line-styles", action="store_true",
        help="Use different line styles (solid, dashed, dash-dot, dotted) per experiment",
    )

    # --- Std deviation & individual runs ---
    band_group = parser.add_argument_group("bands & scatter")
    band_group.add_argument("--no-std", action="store_true", help="Hide standard deviation bands")
    band_group.add_argument("--std-alpha", type=float, default=0.15, help="Alpha for std deviation bands (default: 0.15)")
    band_group.add_argument("--show-runs", action="store_true", help="Show individual run data points")
    band_group.add_argument("--show-trend", action="store_true", help="Show linear trend line per experiment")

    # --- Text & font sizing ---
    text_group = parser.add_argument_group("text")
    text_group.add_argument("--font-size", type=float, default=11.0, help="Axis label font size (default: 11)")
    text_group.add_argument("--title-size", type=float, default=13.0, help="Figure title font size (default: 13)")
    text_group.add_argument("--title", type=str, default=None, help="Custom figure title")
    text_group.add_argument("--no-title", action="store_true", help="Hide figure title entirely")
    text_group.add_argument("--x-label", type=str, default=None, help="Custom x-axis label")
    text_group.add_argument("--y-label-suffix", type=str, default=None, help="Suffix appended to y-axis metric name")

    # --- Legend ---
    legend_group = parser.add_argument_group("legend")
    legend_group.add_argument("--legend-size", type=float, default=9.0, help="Legend font size (default: 9)")
    legend_group.add_argument("--legend-alpha", type=float, default=0.8, help="Legend background opacity (default: 0.8)")
    legend_group.add_argument(
        "--legend-loc", default="best",
        choices=["best", "upper right", "upper left", "lower left", "lower right", "center left", "center right", "lower center", "upper center", "center"],
        help="Legend location (default: best)",
    )

    # --- Grid ---
    grid_group = parser.add_argument_group("grid")
    grid_group.add_argument("--no-grid", action="store_true", help="Disable grid lines")
    grid_group.add_argument("--grid-alpha", type=float, default=0.3, help="Grid line alpha (default: 0.3)")
    grid_group.add_argument(
        "--grid-style", default="--",
        choices=["-", "--", "-.", ":"],
        help="Grid line style (default: --)")

    # --- Colors & theme ---
    theme_group = parser.add_argument_group("colors & theme")
    theme_group.add_argument(
        "--color-palette", default="default",
        choices=list(COLOR_PALETTES.keys()),
        help="Color palette (default: default)",
    )
    theme_group.add_argument("--dark-mode", action="store_true", help="Use dark background theme")

    # --- Output ---
    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output", "-o", help="Output file path")
    output_group.add_argument("--figsize", nargs=2, type=float, default=[14, 8], metavar=("W", "H"), help="Figure size in inches (default: 14 8)")
    output_group.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI (default: 300)")
    output_group.add_argument(
        "--format", default="pdf",
        choices=["pdf", "png", "svg", "eps"],
        help="Output file format (default: pdf)",
    )

    run_cli(_run, parser)


if __name__ == "__main__":
    main()
