from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from data_analysis.plot.common import LEGEND_FONT_SIZE


DEFAULT_QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75)


def build_bucket_histogram(values_hours: pd.Series, bucket_hours: float) -> tuple[np.ndarray, np.ndarray]:
    if bucket_hours <= 0:
        raise ValueError("bucket_hours must be positive")

    buckets = np.floor(values_hours.to_numpy(dtype=float) / bucket_hours).astype(int)
    counts = pd.Series(buckets).value_counts().sort_index()
    x_values = counts.index.to_numpy(dtype=float) * bucket_hours
    y_values = counts.to_numpy(dtype=float)
    return x_values, y_values


def trim_visible_range(values_hours: pd.Series, max_percentile: float | None) -> tuple[pd.Series, float | None]:
    if values_hours.empty:
        return values_hours, None
    if max_percentile is None:
        return values_hours, None
    if not 0 < max_percentile <= 100:
        raise ValueError("max_percentile must be in the range (0, 100]")

    cutoff = float(np.percentile(values_hours.to_numpy(dtype=float), max_percentile))
    return values_hours[values_hours <= cutoff], cutoff


def add_quantile_lines(
    ax: Axes,
    values_hours: pd.Series,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    x_max: float | None = None,
) -> None:
    if values_hours.empty:
        return

    ymax = ax.get_ylim()[1]
    for quantile in quantiles:
        value = float(np.quantile(values_hours.to_numpy(dtype=float), quantile))
        if x_max is not None and value > x_max:
            continue

        label = f"Q{int(round(quantile * 100))}"
        ax.axvline(value, linestyle="--", linewidth=1.2, alpha=0.6, color="#000000")
        ax.text(
            value,
            ymax * 0.98,
            label,
            ha="right",
            va="top",
            fontsize=LEGEND_FONT_SIZE - 1,
            alpha=0.7,
        )


def set_hour_axis_ticks(ax: Axes, bucket_hours: float, x_max: float | None = None) -> None:
    if bucket_hours <= 0:
        raise ValueError("bucket_hours must be positive")

    if x_max is None:
        x_max = float(ax.get_xlim()[1])
    if not np.isfinite(x_max) or x_max <= 0:
        return

    major_step = max(bucket_hours, _choose_tick_step(x_max / 12.0))
    minor_step = max(bucket_hours / 2.0, major_step / 2.0)
    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.grid(axis="x", which="minor", alpha=0.08)


def _choose_tick_step(target_step: float) -> float:
    if target_step <= 0:
        return 1.0

    magnitude = 10 ** np.floor(np.log10(target_step))
    scaled = target_step / magnitude

    if scaled <= 1:
        step = 1.0
    elif scaled <= 2:
        step = 2.0
    elif scaled <= 2.5:
        step = 2.5
    elif scaled <= 5:
        step = 5.0
    else:
        step = 10.0

    return float(step * magnitude)
