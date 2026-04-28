import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from data_analysis.plot.common import LEGEND_FONT_SIZE, SEMANTIC_COLORS, style_axis, dataset_plot_title


def plot_field_length_distribution(
    lengths: pd.Series,
    ax: Axes,
    field: str,
    dataset_name: str,
    max_quantile: float | None = None,
    primary_color: str = SEMANTIC_COLORS["item_property"],
) -> float | None:
    max_len = int(lengths.max())
    bins = np.arange(-0.5, max_len + 1.5, 1)
    quantile_value = None

    visible_lengths = lengths
    if max_quantile is not None:
        quantile_value = float(np.percentile(lengths, max_quantile))
        visible_lengths = lengths[lengths <= quantile_value]

    ax.hist(visible_lengths, bins=bins, color=primary_color, edgecolor="black", alpha=0.85) # type: ignore

    mean_value = float(lengths.mean())
    median_value = float(lengths.median())
    ax.axvline(mean_value, color="#D81B60", linestyle="--", linewidth=2.0, label=f"Mean={mean_value:.2f}")
    ax.axvline(median_value, color="#00429D", linestyle="-.", linewidth=2.0, label=f"Median={median_value:.2f}")
    if quantile_value is not None:
        ax.set_xlim(-0.5, quantile_value + 0.5)

    style_axis(
        ax,
        xlabel="Length (words)",
        ylabel="Count",
        title=dataset_plot_title(dataset_name, f"{field.title()} Length Distribution"),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    return quantile_value
