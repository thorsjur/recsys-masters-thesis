import argparse
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd

from data_analysis.plot.common import SEMANTIC_COLORS
from data_analysis.plot.dataset_interactions_timeline import plot_interactions_timeline
from data_analysis.plot.dataset_time_pattern import plot_time_pattern
from data_analysis.plot.dataset_user_item_distributions import plot_user_item_distributions
from data_analysis.plot.item_age_distribution import plot_item_age_distribution
from data_analysis.plot.item_lifetime_distribution import plot_item_lifetime_distribution


def plot_dataset_analysis(
    df: pd.DataFrame,
    dataset_name: str,
    granularity: str,
    figsize: tuple[float, float] = (12, 6),
    log_scale: bool = False,
    primary_color: str = SEMANTIC_COLORS["user_interaction"],
    bucket_hours: int = 24,
    lifetime_bucket_hours: float = 1.0,
    lifetime_max_percentile: float | None = 99.0,
    ignore_single_interaction_items: bool = False,
    item_age_absolute_values: bool = False,
) -> dict[str, Any]:
    user_interaction_color = primary_color
    item_interaction_color = SEMANTIC_COLORS["item_interaction"]
    item_property_color = SEMANTIC_COLORS["item_property"]

    figures: dict[str, Any] = {}

    fig1, ax1 = plt.subplots(figsize=figsize)
    plot_interactions_timeline(df, ax1, dataset_name, bucket_hours, user_interaction_color)
    figures["interactions_timeline"] = fig1

    fig2, ax2 = plt.subplots(figsize=figsize)
    plot_time_pattern(df, ax2, dataset_name, granularity, bucket_hours, user_interaction_color)
    figures["time_pattern"] = fig2

    fig3, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
    plot_user_item_distributions(
        df,
        (axes[0], axes[1]),
        dataset_name,
        log_scale,
        user_color=user_interaction_color,
        item_color=item_interaction_color,
    )
    fig3.tight_layout()
    figures["distributions"] = fig3

    fig4, ax4 = plt.subplots(figsize=figsize)
    plot_item_age_distribution(
        df,
        ax4,
        dataset_name,
        bucket_hours=lifetime_bucket_hours,
        color=item_property_color,
        normalize=not item_age_absolute_values,
        max_percentile=lifetime_max_percentile,
        ignore_single_interaction_items=ignore_single_interaction_items,
    )
    figures["item_age_distribution"] = fig4

    fig5, ax5 = plt.subplots(figsize=figsize)
    plot_item_lifetime_distribution(
        df,
        ax5,
        dataset_name,
        bucket_hours=lifetime_bucket_hours,
        color=item_property_color,
        max_percentile=lifetime_max_percentile,
        ignore_single_interaction_items=ignore_single_interaction_items,
    )
    figures["item_lifetime_distribution"] = fig5

    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset analysis plotting from experiment results")
    parser.add_argument("--experiment-id", required=True, help="Temporal experiment ID")
    parser.add_argument("--jsonl-path", default="output/results/experiments.jsonl")
    parser.add_argument("--dataset-path", default="data/atomic_files")
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--bucket-hours", type=int, default=24)
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--ignore-single-interaction-items", action="store_true")
    parser.add_argument("--lifetime-max-percentile", type=float, default=99.0)

    args = parser.parse_args()

    from data_analysis.plot_workflows import analyze_dataset_from_experiment

    analyze_dataset_from_experiment(
        experiment_id=args.experiment_id,
        jsonl_path=args.jsonl_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        bucket_hours=args.bucket_hours,
        log_scale=args.log_scale,
        ignore_single_interaction_items=args.ignore_single_interaction_items,
        lifetime_max_percentile=args.lifetime_max_percentile,
    )


if __name__ == "__main__":
    main()
