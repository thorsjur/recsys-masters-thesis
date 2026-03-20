"""Plotting utilities for window validation and statistics."""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from data_analysis.plot.common import AXIS_LABEL_SIZE, LEGEND_FONT_SIZE, PLOT_TITLE_SIZE

COLORS = {
    "train": "#2E86AB",
    "test": "#A23B72",
    "train_items": "#F18F01",
    "test_items": "#C73E1D",
    "users": "#6A4C93",
    "items": "#1982C4",
}


def _setup_axis(ax: Axes, xlabel: str, ylabel: str, title: str):
    """Apply common axis formatting."""
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")


def plot_window_data_distribution(
    stats_df: pd.DataFrame, ax: Optional[Axes] = None, title: str = "Data Distribution Across Windows"
) -> Axes:
    """Plot interactions and items counts across windows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    windows = stats_df["window_number"].values
    x = np.arange(len(windows))
    width = 0.35

    # Interactions as bars
    ax.bar(
        x - width / 2,
        stats_df["train_interactions"],
        width,
        label="Train Int.",
        color=COLORS["train"],
        alpha=0.8,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        stats_df["test_interactions"],
        width,
        label="Test Int.",
        color=COLORS["test"],
        alpha=0.8,
        edgecolor="black",
    )

    # Items as lines on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        x,
        stats_df["train_items"],
        "o-",
        label="Train Items",
        color=COLORS["train_items"],
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax2.plot(
        x,
        stats_df["test_items"],
        "s-",
        label="Test Items",
        color=COLORS["test_items"],
        linewidth=2.5,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    _setup_axis(ax, "Window Number", "Number of Interactions", title)
    ax2.set_ylabel("Number of Items", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(w)) for w in windows])

    # Combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=LEGEND_FONT_SIZE)

    return ax


def plot_cold_start_ratios(
    stats_df: pd.DataFrame, ax: Optional[Axes] = None, title: str = "Cold-Start Ratios Across Windows"
) -> Axes:
    """Plot cold-start ratios (new users/items) across windows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    windows = stats_df["window_number"].values
    x = np.arange(len(windows))
    width = 0.35

    user_pct = stats_df["cold_start_user_ratio"] * 100
    item_pct = stats_df["cold_start_item_ratio"] * 100

    ax.bar(x - width / 2, user_pct, width, label="New Users %", color=COLORS["users"], alpha=0.8, edgecolor="black")
    ax.bar(x + width / 2, item_pct, width, label="New Items %", color=COLORS["items"], alpha=0.8, edgecolor="black")

    # Mean reference lines
    ax.axhline(
        user_pct.mean(),
        color=COLORS["users"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Mean Users: {user_pct.mean():.1f}%",
    )
    ax.axhline(
        item_pct.mean(),
        color=COLORS["items"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Mean Items: {item_pct.mean():.1f}%",
    )

    _setup_axis(ax, "Window Number", "Cold-Start Ratio (%)", title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(w)) for w in windows])
    ax.set_ylim(0, max(user_pct.max(), item_pct.max()) * 1.15)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    return ax

