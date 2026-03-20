from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_analysis.plot.common import AXIS_LABEL_SIZE, LEGEND_FONT_SIZE, PLOT_TITLE_SIZE, SEMANTIC_COLORS


def save_field_length_distribution(
    lengths: pd.Series,
    field: str,
    output_path: Path,
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(visible_lengths, bins=bins, color=primary_color, edgecolor="black", alpha=0.85)

    mean_value = float(lengths.mean())
    median_value = float(lengths.median())
    ax.axvline(mean_value, color="#D81B60", linestyle="--", linewidth=2.0, label=f"Mean={mean_value:.2f}")
    ax.axvline(median_value, color="#00429D", linestyle="-.", linewidth=2.0, label=f"Median={median_value:.2f}")
    if quantile_value is not None:
        ax.set_xlim(-0.5, quantile_value + 0.5)

    ax.set_xlabel("Length (words)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(f"{field.title()} Length Distribution", fontsize=PLOT_TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=LEGEND_FONT_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return quantile_value
