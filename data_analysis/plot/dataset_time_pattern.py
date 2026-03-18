from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes


def plot_time_pattern(
    df: pd.DataFrame,
    ax: Axes,
    dataset_name: str,
    granularity: str,
    bucket_hours: int,
    primary_color: str,
) -> None:
    if granularity == "hour":
        _plot_hourly_pattern(df, ax, primary_color)
        ax.figure.suptitle(f"{dataset_name} - Hourly Interaction Pattern", fontsize=12, fontweight="bold")
        ax.figure.tight_layout()
        return

    _plot_weekly_histogram(df, ax, bucket_hours, primary_color)
    ax.figure.suptitle(f"{dataset_name} - Weekly Interaction Histogram", fontsize=12, fontweight="bold")
    ax.figure.tight_layout()


def _plot_hourly_pattern(df: pd.DataFrame, ax: Axes, primary_color: str) -> None:
    data = df.copy()
    data["datetime"] = pd.to_datetime(data["timestamp"], unit="s")
    hourly = data.groupby(data["datetime"].dt.hour).size()

    hours = range(24)
    counts = [int(hourly.get(h, 0)) for h in hours]

    ax.bar(hours, counts, color=primary_color, alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.axvspan(-0.5, 6, alpha=0.1, color="gray", label="Night")
    ax.axvspan(22, 24, alpha=0.1, color="gray")
    ax.axvspan(6, 22, alpha=0.05, color="yellow", label="Day")

    ax.set_xlabel("Hour of Day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Interactions", fontsize=11, fontweight="bold")
    ax.set_title("Interaction Pattern by Hour", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)


def _plot_weekly_histogram(df: pd.DataFrame, ax: Axes, bucket_hours: int, primary_color: str) -> None:
    if bucket_hours <= 0:
        raise ValueError("weekly bucket size must be a positive integer number of hours")
    if 168 % bucket_hours != 0:
        raise ValueError("weekly bucket size must divide 168 evenly (hours in a week)")

    data = df.copy()
    data["datetime"] = pd.to_datetime(data["timestamp"], unit="s")
    data["hour_of_week"] = (
        data["datetime"].dt.dayofweek * 24
        + data["datetime"].dt.hour
        + data["datetime"].dt.minute / 60.0
        + data["datetime"].dt.second / 3600.0
    )

    bins = list(range(0, 168 + bucket_hours, bucket_hours))
    ax.hist(data["hour_of_week"], bins=bins, color=primary_color, alpha=0.8, edgecolor="black")

    tick_positions = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    tick_labels = [f"{bins[i]}-{bins[i + 1]}h" for i in range(len(bins) - 1)]

    ax.set_xlabel("Hour of Week Bucket", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Interactions", fontsize=11, fontweight="bold")
    ax.set_title(f"Interaction Histogram by Week ({bucket_hours}h buckets)", fontsize=12, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
