"""Dataset temporal analysis and visualization utilities."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib

from util.constants import SEPARATOR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes

from util.statistics import basic_stats
from recbole.data.dataset import Dataset

COLORS = {"primary": "#2E86AB", "secondary": "#A23B72", "accent": "#F18F01"}


def collect_recbole_dataset_stats(dataset: Dataset, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect statistics from a RecBole dataset."""
    
    inter_feat = dataset.inter_feat
    if inter_feat is None:
        raise ValueError("Dataset has no interaction features")
    
    label_field = config.get("RATING_FIELD", "label") if config else "label"

    # Basic counts from RecBole dataset object
    num_users = int(dataset.user_num)
    num_items = int(dataset.item_num)
    total_interactions = int(dataset.inter_num)

    # Get label tensor/array for positive/negative breakdown
    if label_field in inter_feat:
        labels = inter_feat[label_field].to_numpy()
        num_positive = int(np.sum(labels == 1))
        num_negative = int(np.sum(labels == 0))
    else:
        # If no label field, assume all interactions are positive
        num_positive = total_interactions
        num_negative = 0

    # Sparsity based on positive interactions only (actual clicks)
    sparsity_positive = float(1 - num_positive / (num_users * num_items)) if num_users * num_items > 0 else 1.0
    sparsity_total = float(1 - total_interactions / (num_users * num_items)) if num_users * num_items > 0 else 1.0

    stats: Dict[str, Any] = {
        # Basic counts
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions_total": total_interactions,
        "num_positive_interactions": num_positive,
        "num_negative_interactions": num_negative,
        # Ratios
        "positive_ratio": float(num_positive / total_interactions) if total_interactions > 0 else 0.0,
        "negative_ratio": float(num_negative / total_interactions) if total_interactions > 0 else 0.0,
        # Sparsity
        "sparsity_positive_only": sparsity_positive,
        "sparsity_all_interactions": sparsity_total,
        # Averages based on positive interactions
        "avg_positive_per_user": float(num_positive / num_users) if num_users > 0 else 0.0,
        "avg_positive_per_item": float(num_positive / num_items) if num_items > 0 else 0.0,
        # Averages based on negative interactions
        "avg_negative_per_user": float(num_negative / num_users) if num_users > 0 else 0.0,
        "avg_negative_per_item": float(num_negative / num_items) if num_items > 0 else 0.0,
        # Averages based on all interactions
        "avg_interactions_per_user": float(total_interactions / num_users) if num_users > 0 else 0.0,
        "avg_interactions_per_item": float(total_interactions / num_items) if num_items > 0 else 0.0,
        # Feature availability
        "has_item_features": dataset.item_feat is not None,
        "has_user_features": dataset.user_feat is not None,
    }

    # Compute per-user and per-item statistics using interaction features
    user_field = dataset.uid_field
    item_field = dataset.iid_field

    if user_field in inter_feat and item_field in inter_feat:
        user_ids = inter_feat[user_field].to_numpy()
        item_ids = inter_feat[item_field].to_numpy()

        if label_field in inter_feat:
            labels = inter_feat[label_field].to_numpy()
            positive_mask = labels == 1
            negative_mask = labels == 0

            # Per-user positive interaction distribution
            user_positive_counts = pd.Series(user_ids[positive_mask]).value_counts()
            stats["user_positive_activity"] = basic_stats(user_positive_counts.to_numpy())

            # Per-user negative interaction distribution
            if num_negative > 0:
                user_negative_counts = pd.Series(user_ids[negative_mask]).value_counts()
                stats["user_negative_activity"] = basic_stats(user_negative_counts.to_numpy())

            # Per-item positive interaction distribution (popularity)
            item_positive_counts = pd.Series(item_ids[positive_mask]).value_counts()
            stats["item_positive_popularity"] = basic_stats(item_positive_counts.to_numpy())

            # Per-item negative interaction distribution
            if num_negative > 0:
                item_negative_counts = pd.Series(item_ids[negative_mask]).value_counts()
                stats["item_negative_popularity"] = basic_stats(item_negative_counts.to_numpy())
        else:
            # No labels, compute total activity
            user_counts = pd.Series(user_ids).value_counts()
            stats["user_activity"] = basic_stats(user_counts.to_numpy())

            item_counts = pd.Series(item_ids).value_counts()
            stats["item_popularity"] = basic_stats(item_counts.to_numpy())

    # Impression-level statistics (if impression_id exists)
    if "impression_id" in inter_feat:
        impression_ids = inter_feat["impression_id"].to_numpy()
        unique_impressions = len(np.unique(impression_ids))
        stats["num_impressions"] = unique_impressions
        stats["avg_items_per_impression"] = float(total_interactions / unique_impressions) if unique_impressions > 0 else 0.0

        if label_field in inter_feat:
            labels = inter_feat[label_field].to_numpy()
            # Create DataFrame for impression-level analysis
            impression_df = pd.DataFrame({
                "impression_id": impression_ids,
                "label": labels
            })
            impression_stats = impression_df.groupby("impression_id").agg(
                total_items=("label", "count"),
                positive_items=("label", "sum")
            )
            impression_stats["negative_items"] = impression_stats["total_items"] - impression_stats["positive_items"]

            stats["avg_positive_per_impression"] = float(impression_stats["positive_items"].mean())
            stats["avg_negative_per_impression"] = float(impression_stats["negative_items"].mean())
            stats["impression_size_stats"] = basic_stats(impression_stats["total_items"].to_numpy())

    # Timestamp statistics (if available)
    time_field = config.get("TIME_FIELD", "timestamp") if config else "timestamp"
    if time_field in inter_feat:
        timestamps = inter_feat[time_field].to_numpy()
        stats["timestamp_min"] = float(np.min(timestamps))
        stats["timestamp_max"] = float(np.max(timestamps))
        stats["time_span_seconds"] = float(np.max(timestamps) - np.min(timestamps))
        stats["time_span_hours"] = stats["time_span_seconds"] / 3600
        stats["time_span_days"] = stats["time_span_seconds"] / 86400

    return stats


def format_dataset_stats_summary(stats: Dict[str, Any]) -> str:
    lines = [
        SEPARATOR,
        "Dataset Statistics Summary",
        SEPARATOR,
        f"Users: {stats['num_users']:,}",
        f"Items: {stats['num_items']:,}",
        f"Total Interactions: {stats['num_interactions_total']:,}",
        f"  - Positive (clicks): {stats['num_positive_interactions']:,} ({stats['positive_ratio']:.1%})",
        f"  - Negative (non-clicks): {stats['num_negative_interactions']:,} ({stats['negative_ratio']:.1%})",
        "",
        f"Sparsity (positive only): {stats['sparsity_positive_only']:.6f}",
        f"Sparsity (all interactions): {stats['sparsity_all_interactions']:.6f}",
        "",
        f"Avg positive interactions per user: {stats['avg_positive_per_user']:.2f}",
        f"Avg positive interactions per item: {stats['avg_positive_per_item']:.2f}",
        f"Avg negative interactions per user: {stats['avg_negative_per_user']:.2f}",
        f"Avg negative interactions per item: {stats['avg_negative_per_item']:.2f}",
    ]

    if "num_impressions" in stats:
        lines.extend([
            "",
            f"Impressions: {stats['num_impressions']:,}",
            f"Avg items per impression: {stats['avg_items_per_impression']:.2f}",
            f"Avg clicks per impression: {stats.get('avg_positive_per_impression', 0):.2f}",
        ])

    if "time_span_days" in stats:
        lines.extend([
            "",
            f"Time span: {stats['time_span_days']:.1f} days ({stats['time_span_hours']:.1f} hours)",
        ])

    lines.extend([
        "",
        f"Has item features: {stats['has_item_features']}",
        f"Has user features: {stats['has_user_features']}",
        SEPARATOR,
    ])

    return "\n".join(lines)


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
            df = pd.read_csv(file_path, sep="\t")
            df.columns = [c.split(":")[0] for c in df.columns]
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
