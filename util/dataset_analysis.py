"""Dataset temporal analysis and visualization utilities."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes

from util.statistics import basic_stats

COLORS = {"primary": "#2E86AB", "secondary": "#A23B72", "accent": "#F18F01"}


def _setup_axis(ax: Axes, xlabel: str, ylabel: str, title: str):
    """Apply common axis formatting."""
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")


def _filter_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only positive interactions (label == 1).

    Non-clicked impressions (label == 0) are excluded from analysis.
    """
    if "label" in df.columns:
        return df[df["label"] == 1].copy()
    return df


def load_temporal_interaction_data(
    dataset_path: str, dataset_name: str, granularity: str, time_units: range, positive_only: bool = True
) -> pd.DataFrame:
    """Load and combine interaction data from temporal split files.

    Args:
        dataset_path: Path to datasets directory.
        dataset_name: Name of the dataset.
        granularity: Time granularity ('hour' or 'day').
        time_units: Range of time units to load.
        positive_only: If True (default), only include positive interactions (label==1).
            Set to False to include non-clicked impressions.
    """
    dataset_dir = Path(dataset_path) / dataset_name
    dfs = []

    for unit in time_units:
        suffix = f"{granularity}_{unit}"
        file_path = dataset_dir / f"{dataset_name}.{suffix}.inter"
        if file_path.exists():
            df = pd.read_csv(
                file_path, sep="\t", skiprows=1, names=["user_id", "item_id", "label", "timestamp", "impression_id"]
            )
            df["time_unit"] = unit
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No data files found for {dataset_name} in range {time_units}")

    result = pd.concat(dfs, ignore_index=True)
    return _filter_positive(result) if positive_only else result


def compute_temporal_statistics(
    df: pd.DataFrame, granularity: str, start_timestamp: Optional[float] = None
) -> Dict[str, Any]:
    """Compute temporal statistics from interaction data.

    Expects df to be pre-filtered for positive interactions if that's desired.
    """
    start_ts = start_timestamp or float(df["timestamp"].min())

    stats = {
        "total_interactions": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_items": df["item_id"].nunique(),
        "time_span_seconds": float(df["timestamp"].max() - df["timestamp"].min()),
        "first_timestamp": float(df["timestamp"].min()),
        "last_timestamp": float(df["timestamp"].max()),
        "start_datetime": datetime.fromtimestamp(start_ts).isoformat(),
        "granularity": granularity,
    }

    # Per time-unit stats
    per_unit = df.groupby("time_unit").size()
    stats["interactions_per_unit"] = basic_stats(per_unit.to_numpy())

    # User activity stats
    user_counts = df.groupby("user_id").size()
    stats["user_activity"] = basic_stats(user_counts.to_numpy())

    # Item popularity stats
    item_counts = df.groupby("item_id").size()
    stats["item_popularity"] = basic_stats(item_counts.to_numpy())

    return stats


def plot_interactions_over_time(
    df: pd.DataFrame,
    granularity: str,
    start_timestamp: float,
    window_info: Optional[List[Dict]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Plot interactions over time with optional day/night shading."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6))

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    start_dt = pd.to_datetime(start_timestamp, unit="s")

    # Aggregate by time unit
    time_col = "hour_mark" if granularity == "hour" else "day_mark"
    unit = "h" if granularity == "hour" else "D"
    df[time_col] = start_dt + pd.to_timedelta(df["time_unit"], unit=unit)
    agg = df.groupby(time_col).size().reset_index(name="count")

    # Day/night shading for hourly data
    if granularity == "hour":
        _add_day_night_shading(ax, agg[time_col].min(), agg[time_col].max())

    ax.plot(
        agg[time_col],
        agg["count"],
        linewidth=2.5,
        color=COLORS["primary"],
        marker="o",
        markersize=5,
        zorder=3,
        markerfacecolor=COLORS["primary"],
        markeredgecolor="white",
        markeredgewidth=1,
    )

    xlabel = f"Time ({granularity.title()}ly)"
    _setup_axis(ax, xlabel, "Number of Interactions", "Interaction Volume Over Time")

    # Format x-axis
    fmt = "%m-%d %H:%M" if granularity == "hour" else "%Y-%m-%d"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    return ax


def _add_day_night_shading(ax: Axes, time_min, time_max):
    """Add day/night gradient background to plot."""
    current = time_min
    while current <= time_max:
        day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)

        # Night hours (0-6)
        for i in range(6):
            h_start = day_start + timedelta(hours=i)
            h_end = day_start + timedelta(hours=i + 1)
            if h_start < time_max and h_end > time_min:
                alpha = 0.25 - (i * 0.035)
                ax.axvspan(max(h_start, time_min), min(h_end, time_max), alpha=alpha, color="#1a2332", zorder=0)

        # Day hours (6-18)
        for i in range(12):
            h_start = day_start + timedelta(hours=6 + i)
            h_end = day_start + timedelta(hours=7 + i)
            if h_start < time_max and h_end > time_min:
                color = "#87CEEB" if i < 3 else ("#FFD700" if i < 9 else "#FFA500")
                alpha = 0.15 if 3 <= i < 9 else 0.08 + (min(i, 2) * 0.02)
                ax.axvspan(max(h_start, time_min), min(h_end, time_max), alpha=alpha, color=color, zorder=0)

        # Evening hours (18-24)
        for i in range(6):
            h_start = day_start + timedelta(hours=18 + i)
            h_end = day_start + timedelta(hours=19 + i)
            if h_start < time_max and h_end > time_min:
                alpha = 0.10 + (i * 0.025)
                color = "#1a2332" if i > 2 else "#4a5a6a"
                ax.axvspan(max(h_start, time_min), min(h_end, time_max), alpha=alpha, color=color, zorder=0)

        current += timedelta(days=1)


def plot_time_of_day_pattern(df: pd.DataFrame, start_timestamp: float, ax: Optional[Axes] = None) -> Axes:
    """Plot interaction patterns by hour of day."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_of_day"] = df["datetime"].dt.hour  # type: ignore[union-attr]

    hourly = df.groupby("hour_of_day").size()
    hours = range(24)
    counts = [hourly.get(h, 0) for h in hours]
    colors = [COLORS["primary"] if 6 <= h < 22 else COLORS["secondary"] for h in hours]

    ax.bar(hours, counts, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvspan(-0.5, 6, alpha=0.1, color="gray", label="Night")
    ax.axvspan(22, 24, alpha=0.1, color="gray")
    ax.axvspan(6, 22, alpha=0.05, color="yellow", label="Day")

    _setup_axis(ax, "Hour of Day", "Number of Interactions", "Interaction Pattern by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right", fontsize=9)

    return ax


def plot_user_item_distributions(
    df: pd.DataFrame, ax: Optional[Tuple[Axes, Axes]] = None, log_scale: bool = False
) -> Tuple[Axes, Axes]:
    """Plot user activity and item popularity distributions."""
    if ax is None:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax1, ax2 = ax

    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    for axis, counts, label, color in [
        (ax1, user_counts, "User", COLORS["primary"]),
        (ax2, item_counts, "Item", COLORS["accent"]),
    ]:
        if log_scale:
            bins = np.logspace(np.log10(counts.min()), np.log10(counts.max()), 50)
            axis.hist(counts, bins=bins, color=color, alpha=0.7, edgecolor="black")
            axis.set_xscale("log")
            axis.set_yscale("log")
            suffix = " (Log-Log)"
        else:
            axis.hist(counts, bins=50, color=color, alpha=0.7, edgecolor="black")
            suffix = ""

        _setup_axis(
            axis,
            f"Interactions per {label}",
            f"Number of {label}s",
            f'{label} {"Activity" if label == "User" else "Popularity"} Distribution{suffix}',
        )

        stats_text = f"Mean: {counts.mean():.1f}\nMedian: {counts.median():.1f}\nStd: {counts.std():.1f}"
        axis.text(
            0.95,
            0.95,
            stats_text,
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    return ax1, ax2
