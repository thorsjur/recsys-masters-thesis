from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes


def plot_interactions_timeline(
    df: pd.DataFrame,
    ax: Axes,
    dataset_name: str,
    bucket_hours: int,
    color: str,
) -> None:
    if bucket_hours <= 0:
        raise ValueError("bucket size must be a positive integer number of hours")

    data = df.copy()
    data["datetime"] = pd.to_datetime(data["timestamp"], unit="s")

    start = data["datetime"].min().floor("h")
    end = data["datetime"].max().ceil("h")
    bucket = pd.Timedelta(hours=bucket_hours)
    bins = pd.date_range(start=start, end=end + bucket, freq=bucket)

    ax.hist(data["datetime"], bins=bins, color=color, alpha=0.85, edgecolor="black", zorder=3)

    day = start.normalize()
    while day < end:
        ax.axvspan(day, day + pd.Timedelta(hours=6), color="gray", alpha=0.15, zorder=0)
        ax.axvspan(day + pd.Timedelta(hours=22), day + pd.Timedelta(days=1), color="gray", alpha=0.15, zorder=0)
        day += pd.Timedelta(days=1)

    ax.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Interactions", fontsize=11, fontweight="bold")
    ax.set_title(f"Interaction Volume Over Time ({bucket_hours}h buckets)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.tick_params(axis="x", rotation=45)

    fig = ax.figure
    fig.suptitle(f"{dataset_name} - Interactions Over Time", fontsize=12, fontweight="bold")
    fig.tight_layout()
