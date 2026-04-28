import argparse
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd

from data_analysis.plot.window_validation_plots import plot_cold_start_ratios, plot_window_data_distribution


def plot_window_validation(
    stats_df: pd.DataFrame,
    dataset_name: str,
    figsize: tuple[float, float] = (10, 5),
) -> dict[str, Any]:
    figures: dict[str, Any] = {}

    fig_data, ax_data = plt.subplots(figsize=figsize)
    plot_window_data_distribution(stats_df, ax=ax_data, dataset_name=dataset_name)
    fig_data.tight_layout()
    figures["data_distribution"] = fig_data

    fig_cold, ax_cold = plt.subplots(figsize=figsize)
    plot_cold_start_ratios(stats_df, ax=ax_cold, dataset_name=dataset_name)
    fig_cold.tight_layout()
    figures["cold_start"] = fig_cold

    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window validation plotting from experiment results")
    parser.add_argument("--experiment-id", required=True, help="Temporal experiment ID")
    parser.add_argument("--jsonl-path", default="output/results/experiments.jsonl")
    parser.add_argument("--dataset-path", default="data/atomic_files")
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    from data_analysis.plot_workflows import validate_sliding_windows

    validate_sliding_windows(
        experiment_id=args.experiment_id,
        jsonl_path=args.jsonl_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
