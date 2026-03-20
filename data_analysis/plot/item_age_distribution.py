from __future__ import annotations

import numpy as np
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


def plot_item_age_distribution(
    df: pd.DataFrame,
    ax: Axes,
    dataset_name: str,
    bucket_hours: float = 1.0,
    color: str = "#2E86AB",
    normalize: bool = True,
    max_percentile: float | None = 99.0,
    ignore_single_interaction_items: bool = False,
    show_quantiles: bool = True,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> None:
    filtered_df, age_hours = _compute_item_age_hours(df, ignore_single_interaction_items=ignore_single_interaction_items)
    x_values, y_values, quantile_hours, cutoff = _compute_average_item_views_per_bucket(
        filtered_df,
        age_hours,
        bucket_hours,
        normalize=normalize,
        max_percentile=max_percentile,
    )

    ylabel = "Average Share of Interactions per Item (%)" if normalize else "Average Number of Interactions per Item"

    ax.bar(x_values, y_values, width=bucket_hours, align="edge", color=color, edgecolor="black", alpha=0.85)
    title = "Average Interaction Share per Item Over Item Age" if normalize else "Average Interaction Count per Item Over Item Age"
    style_axis(ax, "Item Age (hours)", ylabel, title)
    ax.grid(axis="x", which="major", alpha=0.15)
    if cutoff is not None:
        ax.set_xlim(0, cutoff)
    set_hour_axis_ticks(ax, bucket_hours, x_max=cutoff)

    if show_quantiles:
        add_quantile_lines(ax, quantile_hours, quantiles=quantiles, x_max=cutoff)

    ax.figure.suptitle(f"{dataset_name} - Item Age Distribution", fontsize=PLOT_TITLE_SIZE, fontweight="bold")
    ax.figure.tight_layout()


def _compute_item_age_hours(df: pd.DataFrame, ignore_single_interaction_items: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    if ignore_single_interaction_items:
        item_counts = df.groupby("item_id")["timestamp"].transform("size")
        df = df[item_counts > 1]
    if df.empty:
        return df, pd.Series(dtype=float)

    first_timestamps = df.groupby("item_id")["timestamp"].transform("min")
    age_hours = (pd.to_numeric(df["timestamp"], errors="coerce") - first_timestamps) / 3600.0
    return df, age_hours


def _compute_average_item_views_per_bucket(
    df: pd.DataFrame,
    age_hours: pd.Series,
    bucket_hours: float,
    normalize: bool,
    max_percentile: float | None,
) -> tuple[np.ndarray, np.ndarray, pd.Series, float | None]:
    if df.empty or age_hours.empty:
        return np.array([], dtype=float), np.array([], dtype=float), pd.Series(dtype=float), None

    quantile_hours = age_hours
    _, cutoff = trim_visible_range(quantile_hours, max_percentile)
    bucket_ids = np.floor(age_hours.to_numpy(dtype=float) / bucket_hours).astype(int)
    bucketed = pd.DataFrame({"item_id": df["item_id"].to_numpy(), "bucket_id": bucket_ids})
    per_item_bucket_counts = bucketed.groupby(["item_id", "bucket_id"]).size().rename("count").reset_index()

    if normalize:
        item_totals = per_item_bucket_counts.groupby("item_id")["count"].transform("sum")
        per_item_bucket_counts["value"] = (per_item_bucket_counts["count"] / item_totals) * 100.0
    else:
        per_item_bucket_counts["value"] = per_item_bucket_counts["count"].astype(float)

    if cutoff is not None:
        max_bucket_id = int(np.floor(cutoff / bucket_hours))
        per_item_bucket_counts = per_item_bucket_counts[per_item_bucket_counts["bucket_id"] <= max_bucket_id]

    item_count = df["item_id"].nunique()
    mean_values = per_item_bucket_counts.groupby("bucket_id")["value"].sum() / float(item_count)
    bucket_positions = mean_values.index.to_numpy(dtype=float) * bucket_hours
    return bucket_positions, mean_values.to_numpy(dtype=float), quantile_hours, cutoff
