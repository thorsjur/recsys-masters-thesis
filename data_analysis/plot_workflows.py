from typing import Any, Optional
import matplotlib.pyplot as plt

from data_analysis.dataset_analysis import compute_temporal_statistics, load_temporal_interaction_data
from data_analysis.plot.common import SEMANTIC_COLORS, get_output_dir
from data_analysis.plot.dataset_analysis import plot_dataset_analysis
from data_analysis.plot.window_validation import plot_window_validation
from data_analysis.window_validation import compute_all_window_statistics
from util.experiment_data import load_experiment_results


def _collect_windows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_number: dict[int, dict[str, Any]] = {}
    for result in results:
        window_info = result.get("window_info", {})
        window_number = window_info.get("window_number")
        if window_number is not None and window_number not in by_number:
            by_number[window_number] = window_info
    return [by_number[k] for k in sorted(by_number.keys())]


def _load_temporal_experiment(experiment_id: str, jsonl_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = load_experiment_results(jsonl_path, experiment_id)
    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")

    first = results[0]
    if not first.get("window_info"):
        raise ValueError(f"Experiment {experiment_id} has no window_info - not a temporal experiment")

    return results, first


def _save_and_close_figure(fig, path) -> None:
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset_from_experiment(
    experiment_id: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    dataset_path: str = "data/atomic_files",
    output_dir: Optional[str] = None,
    start_timestamp: Optional[float] = None,
    figsize: tuple[float, float] = (12, 6),
    log_scale: bool = False,
    primary_color: str = SEMANTIC_COLORS["user_interaction"],
    bucket_hours: int = 24,
    lifetime_bucket_hours: float = 1.0,
    lifetime_max_percentile: float | None = 99.0,
    ignore_single_interaction_items: bool = False,
    item_age_absolute_values: bool = False,
) -> dict[str, Any]:
    results, first = _load_temporal_experiment(experiment_id, jsonl_path)

    dataset_name = first.get("run_info", {}).get("dataset", "unknown")
    granularity = first["window_info"].get("granularity", "hour")
    windows = _collect_windows(results)
    min_unit = min(window.get("start_unit", 0) for window in windows)
    max_unit = max(window.get("end_unit", 0) for window in windows)

    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1),
    )
    if df.empty:
        raise ValueError(
            f"No interactions found for dataset '{dataset_name}' ({granularity} {min_unit}-{max_unit}). "
            "Check that atomic files exist and dataset name is correct."
        )

    if start_timestamp is None:
        start_timestamp = float(df["timestamp"].min())

    stats = compute_temporal_statistics(df, granularity, start_timestamp)
    figures = plot_dataset_analysis(
        df=df,
        dataset_name=dataset_name,
        granularity=granularity,
        figsize=figsize,
        log_scale=log_scale,
        primary_color=primary_color,
        bucket_hours=bucket_hours,
        lifetime_bucket_hours=lifetime_bucket_hours,
        lifetime_max_percentile=lifetime_max_percentile,
        ignore_single_interaction_items=ignore_single_interaction_items,
        item_age_absolute_values=item_age_absolute_values,
    )

    out_dir = get_output_dir(output_dir)
    file_map = {
        "interactions_timeline": f"{experiment_id}_interactions_timeline.pdf",
        "time_pattern": f"{experiment_id}_time_pattern.pdf",
        "distributions": f"{experiment_id}_distributions.pdf",
        "item_age_distribution": f"{experiment_id}_item_age_distribution.pdf",
        "item_lifetime_distribution": f"{experiment_id}_item_lifetime_distribution.pdf",
    }

    output_paths: list[str] = []
    for key, fig in figures.items():
        path = out_dir / file_map[key]
        _save_and_close_figure(fig, path)
        output_paths.append(str(path))

    return {
        "statistics": stats,
        "dataframe": df,
        "output_paths": output_paths,
    }


def validate_sliding_windows(
    experiment_id: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    dataset_path: str = "data/atomic_files",
    output_dir: Optional[str] = None,
    generate_plots: bool = True,
) -> dict[str, Any]:
    results, first = _load_temporal_experiment(experiment_id, jsonl_path)

    dataset_name = first.get("run_info", {}).get("dataset", "unknown")
    granularity = first["window_info"].get("granularity", "hour")
    windows = _collect_windows(results)

    min_unit = min(window.get("start_unit", 0) for window in windows)
    max_unit = max(window.get("end_unit", 0) for window in windows)

    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1),
    )
    stats_df = compute_all_window_statistics(df, windows, granularity)

    output_paths: dict[str, str] = {}

    if generate_plots:
        out_dir = get_output_dir(output_dir)
        figures = plot_window_validation(stats_df=stats_df, dataset_name=dataset_name, figsize=(10, 5))
        for key, fig in figures.items():
            path = out_dir / f"{experiment_id}_{key}.pdf"
            _save_and_close_figure(fig, path)
            output_paths[f"{key}_plot"] = str(path)

    return {
        "statistics": stats_df,
        "output_paths": output_paths,
        "experiment_id": experiment_id,
        "dataset_name": dataset_name,
        "num_windows": len(windows),
    }
