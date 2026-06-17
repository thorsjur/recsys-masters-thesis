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


def plot_residual_window_effect_trajectories(window_effects: pd.DataFrame) -> plt.Figure:
    datasets = list(window_effects["dataset"].drop_duplicates())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3.4 * len(datasets)), sharey=True, squeeze=False)

    for ax, dataset in zip(axes.ravel(), datasets):
        group = window_effects[window_effects["dataset"].eq(dataset)].sort_values("window_number")
        x = group["window_number"].to_numpy(dtype=float)
        color = DATASET_COLORS.get(dataset, "#555555")

        ax.fill_between(
            x,
            group["metric_delta_q05"].to_numpy(dtype=float),
            group["metric_delta_q95"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            x,
            group["metric_delta_mean"].to_numpy(dtype=float),
            color=color,
            marker="o",
            linewidth=1.8,
            markersize=4,
        )
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(f"{group['dataset_label'].iloc[0]} residual window trajectory", fontsize=PLOT_TITLE_SIZE - 4)
        ax.set_ylabel("Residual NDCG@5 delta", fontsize=AXIS_LABEL_SIZE)
        ax.set_xticks(group["window_number"].astype(int).tolist())
        ax.tick_params(axis="both", labelsize=AXIS_NUMBER_SIZE)
        ax.grid(alpha=0.25)

    axes.ravel()[-1].set_xlabel("Evaluation window", fontsize=AXIS_LABEL_SIZE)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot posterior residual dataset-window effects from a_DW.")
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--levels-path", type=Path, default=DEFAULT_LEVELS_PATH)
    parser.add_argument("--observations-path", type=Path, default=DEFAULT_OBSERVATIONS_PATH)
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--output-name", default="residual_window_effect_trajectories.pdf")
    args = parser.parse_args()

    window_effects, _ = load_residual_window_data(
        fit_dir=args.fit_dir,
        levels_path=args.levels_path,
        observations_path=args.observations_path,
    )
    fig = plot_residual_window_effect_trajectories(window_effects)
    save_figure(fig, get_output_dir(args.output_dir) / args.output_name)


if __name__ == "__main__":
    main()
