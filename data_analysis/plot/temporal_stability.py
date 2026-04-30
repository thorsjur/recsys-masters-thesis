import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.plot.common import (
    AXIS_LABEL_SIZE,
    AXIS_NUMBER_SIZE,
    COLORS,
    LEGEND_FONT_SIZE,
    PLOT_TITLE_SIZE,
    dataset_label,
    get_output_dir,
    save_figure,
)
from util.experiment_data import extract_temporal_metrics, load_experiment_results

MARKER_STYLES = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINE_STYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

COLOR_PALETTES = {
    "default": COLORS,
    "muted": ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB", "#E8A667", "#92C6FF"],
    "vibrant": ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#264653", "#A8DADC", "#1D3557"],
    "pastel": ["#A8D8EA", "#AA96DA", "#FCBAD3", "#FFFFD2", "#B5EAD7", "#C7CEEA", "#FFB7B2", "#E2F0CB"],
    "colorblind": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"],
}


def plot_temporal_stability(
    all_data: List[Dict[str, Any]],
    show_std: bool = True,
    show_individual_runs: bool = False,
    figsize: Optional[tuple[float, float]] = None,
    line_width: float = 2.5,
    marker_size: float = 6.0,
    vary_markers: bool = False,
    vary_line_styles: bool = False,
    font_size: float = AXIS_LABEL_SIZE,
    tick_label_size: float = AXIS_NUMBER_SIZE,
    title_size: float = PLOT_TITLE_SIZE,
    legend_size: float = LEGEND_FONT_SIZE,
    legend_loc: str = "best",
    legend_alpha: float = 0.8,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    grid_style: str = "--",
    std_alpha: float = 0.15,
    show_trend_line: bool = False,
    color_palette: str = "default",
    dark_mode: bool = False,
    x_label: Optional[str] = None,
    y_label_suffix: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    if not all_data:
        raise ValueError("all_data must not be empty")

    if dark_mode:
        plt.style.use("dark_background")

    plot_metrics = all_data[0]["metrics"]
    if not plot_metrics:
        raise ValueError("No metrics found in experiment data")

    colors = COLOR_PALETTES.get(color_palette, COLOR_PALETTES["default"])
    style_opts = {
        "line_width": line_width,
        "marker_size": marker_size,
        "vary_markers": vary_markers,
        "vary_line_styles": vary_line_styles,
        "font_size": font_size,
        "tick_label_size": tick_label_size,
        "title_size": title_size,
        "legend_size": legend_size,
        "legend_loc": legend_loc,
        "legend_alpha": legend_alpha,
        "show_grid": show_grid,
        "grid_alpha": grid_alpha,
        "grid_style": grid_style,
        "std_alpha": std_alpha,
        "show_trend_line": show_trend_line,
        "colors": colors,
        "x_label": x_label,
        "y_label_suffix": y_label_suffix,
    }

    n_metrics = len(plot_metrics)
    if figsize is None:
        figsize = (10, 5) if n_metrics == 1 else (14, 8)

    ncols = min(n_metrics, 2)
    nrows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(plot_metrics):
        _plot_metric_lines(axes_flat[idx], all_data, metric, show_std, show_individual_runs, style_opts)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=title_size, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig


def _plot_metric_lines(
    ax: Axes,
    all_data: List[Dict[str, Any]],
    metric: str,
    show_std: bool,
    show_individual_runs: bool,
    style_opts: Dict[str, Any],
) -> None:
    metadata = all_data[0]["metadata"]
    dataset_name = dataset_label(metadata["dataset"])
    granularity = metadata["granularity"]
    use_temporal = granularity != "unknown"

    colors = style_opts["colors"]
    for exp_idx, data in enumerate(all_data):
        windows = data["windows"]
        window_numbers = sorted(windows.keys())

        if use_temporal:
            x_vals = [windows[w]["info"].get("end_unit", 0) for w in window_numbers]
        else:
            x_vals = list(window_numbers)

        means = [windows[w]["mean"][metric] for w in window_numbers]
        stds = [windows[w]["std"][metric] for w in window_numbers]

        color = colors[exp_idx % len(colors)]
        marker = MARKER_STYLES[exp_idx % len(MARKER_STYLES)] if style_opts["vary_markers"] else "o"
        linestyle = LINE_STYLES[exp_idx % len(LINE_STYLES)] if style_opts["vary_line_styles"] else "-"
        label = data["metadata"]["model"] if len(all_data) > 1 else "Mean"

        ax.plot(
            x_vals,
            means,
            color=color,
            linewidth=style_opts["line_width"],
            linestyle=linestyle,
            marker=marker,
            markersize=style_opts["marker_size"],
            label=label,
        )

        if show_std and any(std > 0 for std in stds):
            means_arr = np.asarray(means, dtype=float)
            stds_arr = np.asarray(stds, dtype=float)
            ax.fill_between(
                x_vals, means_arr - stds_arr, means_arr + stds_arr, alpha=style_opts["std_alpha"], color=color
            )

        if style_opts["show_trend_line"] and len(x_vals) >= 2:
            coeffs = np.polyfit(x_vals, means, 1)
            trend = np.polyval(coeffs, x_vals)
            ax.plot(x_vals, trend, color=color, linewidth=1.0, linestyle="--", alpha=0.5)

        if show_individual_runs:
            for x_idx, window_num in enumerate(window_numbers):
                vals = windows[window_num]["values"][metric]
                ax.scatter(
                    [x_vals[x_idx]] * len(vals),
                    vals,
                    alpha=0.3,
                    s=style_opts["marker_size"] * 3.3,
                    color=color,
                    marker=marker,
                )

    gran_label = granularity.capitalize() if use_temporal else "Window"
    x_axis_label = style_opts["x_label"] or f"Time ({gran_label}s)"
    y_axis_label = f"{metric.upper()}{style_opts['y_label_suffix'] or ''}"

    ax.set_xlabel(x_axis_label, fontsize=style_opts["font_size"])
    ax.set_ylabel(y_axis_label, fontsize=style_opts["font_size"])
    ax.set_title(
        f"{dataset_name} {metric.upper()} Temporal Stability",
        fontsize=style_opts["title_size"],
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="both", which="major", labelsize=style_opts["tick_label_size"])

    if style_opts["show_grid"]:
        ax.grid(True, alpha=style_opts["grid_alpha"], linestyle=style_opts["grid_style"])
    else:
        ax.grid(False)

    legend = ax.legend(loc=style_opts["legend_loc"], fontsize=style_opts["legend_size"])
    if legend:
        legend.get_frame().set_alpha(style_opts["legend_alpha"])


def _default_output_path(experiment_ids: list[str]) -> Path:
    out_dir = get_output_dir()
    joined_ids = "_vs_".join(experiment_ids)
    safe_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in joined_ids)
    return out_dir / f"{safe_name}_temporal_stability.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot temporal stability from experiment results")
    parser.add_argument("--experiment-id", nargs="+", required=True, help="One or more experiment ids")
    parser.add_argument("--metrics", nargs="+", help="Metrics to plot")
    parser.add_argument("--title")

    args = parser.parse_args()

    all_data = [
        extract_temporal_metrics(
            load_experiment_results("output/results/experiments.jsonl", experiment_id), metrics=args.metrics
        )
        for experiment_id in args.experiment_id
    ]
    legend_size = LEGEND_FONT_SIZE if len(all_data[0]["metrics"]) == 1 else 7.0

    fig = plot_temporal_stability(
        all_data=all_data,
        show_std=False,
        line_width=0.0,
        marker_size=6,
        vary_markers=True,
        legend_size=legend_size,
        legend_alpha=0.5,
        show_trend_line=True,
        title=args.title,
    )

    save_figure(fig, _default_output_path(args.experiment_id))


if __name__ == "__main__":
    main()
