
import pandas as pd
from matplotlib.axes import Axes

from data_analysis.plot.common import LEGEND_FONT_SIZE, PLOT_TITLE_SIZE, style_axis, DATASET_NAMING


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
        ax.figure.suptitle(f"{DATASET_NAMING.get(dataset_name, dataset_name)} Hourly Interaction Pattern", fontsize=PLOT_TITLE_SIZE, fontweight="bold")
        ax.figure.tight_layout()
        return

    _plot_weekly_histogram(df, ax, bucket_hours, primary_color)
    ax.figure.suptitle(f"{DATASET_NAMING.get(dataset_name, dataset_name)} Weekly Interaction Histogram", fontsize=PLOT_TITLE_SIZE, fontweight="bold")
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

    style_axis(ax, "Hour of Day", "Number of Interactions", "Interaction Pattern by Hour")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)


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

    style_axis(ax, "Hour of Week Bucket", "Number of Interactions", f"Interaction Histogram by Week ({bucket_hours}h buckets)")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
