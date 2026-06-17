from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from data_analysis.plot.common import (
    AXIS_LABEL_SIZE,
    AXIS_NUMBER_SIZE,
    COLORS,
    PLOT_TITLE_SIZE,
    get_output_dir,
    save_figure,
)
from data_analysis.plot.residual_window_effects_common import (
    DEFAULT_LEVELS_PATH,
    DEFAULT_OBSERVATIONS_PATH,
    load_residual_window_data,
)

DATASET_COLORS = {"ebnerd": COLORS[0], "mind": COLORS[2]}


def plot_residual_window_effect_distributions(
    residual_draws: pd.DataFrame,
    window_effects: pd.DataFrame,
    bins: int = 55,
) -> plt.Figure:
    datasets = list(window_effects["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.4), sharey=True, squeeze=False)

    for ax, dataset in zip(axes.ravel(), datasets):
        draw_group = residual_draws[residual_draws["dataset"].eq(dataset)]
        effect_group = window_effects[window_effects["dataset"].eq(dataset)]
        color = DATASET_COLORS.get(dataset, "#555555")

        ax.hist(
            draw_group["metric_delta"],
            bins=bins,
            density=True,
            color=color,
            alpha=0.60,
            edgecolor="#ffffff",
            linewidth=0.25,
        )
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
        for value in effect_group["metric_delta_mean"]:
            ax.axvline(value, color="#111111", alpha=0.18, linewidth=0.7)
        ax.set_title(
            f"{effect_group['dataset_label'].iloc[0]} ({effect_group['window_number'].nunique()} windows)",
            fontsize=PLOT_TITLE_SIZE - 4,
        )
        ax.set_xlabel("Residual NDCG@5 delta", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=AXIS_NUMBER_SIZE)
        ax.grid(axis="y", alpha=0.25)

    axes.ravel()[0].set_ylabel("Pooled posterior density", fontsize=AXIS_LABEL_SIZE)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pooled posterior residual a_DW effect distributions.")
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--levels-path", type=Path, default=DEFAULT_LEVELS_PATH)
    parser.add_argument("--observations-path", type=Path, default=DEFAULT_OBSERVATIONS_PATH)
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--output-name", default="residual_window_effect_distributions.pdf")
    parser.add_argument("--bins", type=int, default=55)
    args = parser.parse_args()

    window_effects, residual_draws = load_residual_window_data(
        fit_dir=args.fit_dir,
        levels_path=args.levels_path,
        observations_path=args.observations_path,
    )
    fig = plot_residual_window_effect_distributions(residual_draws, window_effects, bins=args.bins)
    save_figure(fig, get_output_dir(args.output_dir) / args.output_name)


if __name__ == "__main__":
    main()
