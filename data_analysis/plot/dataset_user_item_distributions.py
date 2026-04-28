import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from data_analysis.plot.common import ANNOTATION_FONT_SIZE, style_axis, dataset_plot_title


def plot_user_item_distributions(
    df: pd.DataFrame,
    axes: tuple[Axes, Axes],
    dataset_name: str,
    log_scale: bool,
    user_color: str,
    item_color: str,
) -> None:
    ax_user, ax_item = axes

    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    suffix = " (Log-Log)" if log_scale else ""
    _plot_distribution(
        ax_user,
        user_counts,
        "User",
        user_color,
        log_scale,
        dataset_plot_title(dataset_name, f"User Activity Distribution{suffix}"),
    )
    _plot_distribution(
        ax_item,
        item_counts,
        "Item",
        item_color,
        log_scale,
        dataset_plot_title(dataset_name, f"Item Popularity Distribution{suffix}"),
    )


def _plot_distribution(ax: Axes, counts: pd.Series, label: str, color: str, log_scale: bool, title: str) -> None:
    if log_scale:
        bins = np.logspace(np.log10(max(1, counts.min())), np.log10(counts.max()), 50)
        ax.hist(counts, bins=bins, color=color, alpha=0.7, edgecolor="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.hist(counts, bins=50, color=color, alpha=0.7, edgecolor="black")

    style_axis(ax, f"Interactions per {label}", f"Number of {label}s", title)
    ax.grid(True, alpha=0.3, linestyle="--")

    stats_text = f"Mean: {counts.mean():.1f}\nMedian: {counts.median():.1f}\nStd: {counts.std():.1f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
