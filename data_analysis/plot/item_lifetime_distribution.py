from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes

from data_analysis.plot.common import PLOT_TITLE_SIZE, style_axis
from data_analysis.plot.lifetime_common import (
    DEFAULT_QUANTILES,
    add_quantile_lines,
    build_bucket_histogram,
    set_hour_axis_ticks,
    trim_visible_range,
)


def plot_item_lifetime_distribution(
    df: pd.DataFrame,
    ax: Axes,
    dataset_name: str,
    bucket_hours: float = 1.0,
    color: str = "#F18F01",
    normalize: bool = False,
    max_percentile: float | None = 99.0,
    ignore_single_interaction_items: bool = False,
    show_quantiles: bool = True,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> None:
    lifetime_hours = _compute_item_lifetime_hours(df, ignore_single_interaction_items=ignore_single_interaction_items)
    visible_hours, cutoff = trim_visible_range(lifetime_hours, max_percentile)
    x_values, y_values = build_bucket_histogram(visible_hours, bucket_hours)

    ylabel = "Number of Items"
    if normalize and y_values.sum() > 0:
        y_values = (y_values / y_values.sum()) * 100.0
        ylabel = "Percentage of Items (%)"

    ax.bar(x_values, y_values, width=bucket_hours, align="edge", color=color, edgecolor="black", alpha=0.85)
    style_axis(ax, "Item Lifetime (hours)", ylabel, "Distribution of Item Lifetimes")
    ax.grid(axis="x", which="major", alpha=0.15)
    if cutoff is not None:
        ax.set_xlim(0, cutoff)
    set_hour_axis_ticks(ax, bucket_hours, x_max=cutoff)

    if show_quantiles:
        add_quantile_lines(ax, lifetime_hours, quantiles=quantiles, x_max=cutoff)

    ax.figure.suptitle(f"{dataset_name} - Item Lifetime Distribution", fontsize=PLOT_TITLE_SIZE, fontweight="bold")
    ax.figure.tight_layout()


def _compute_item_lifetime_hours(df: pd.DataFrame, ignore_single_interaction_items: bool = False) -> pd.Series:
    grouped = df.groupby("item_id")["timestamp"].agg(["min", "max", "size"])
    if ignore_single_interaction_items:
        grouped = grouped[grouped["size"] > 1]
    if grouped.empty:
        return pd.Series(dtype=float)

    return (grouped["max"] - grouped["min"]) / 3600.0
