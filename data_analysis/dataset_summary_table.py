from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_analysis.atomic_file import find_interaction_files, find_item_file, load_interaction_dataframe, load_item_dataframe


def summarize_interaction_dataframe(
    dataset: str,
    interactions: pd.DataFrame,
    item_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    total_rows = int(len(interactions))
    clicked_df = interactions
    has_label_column = "label" in interactions.columns

    clicked_rows = int(len(clicked_df))
    user_counts = clicked_df.groupby("user_id").size() if "user_id" in clicked_df.columns else pd.Series(dtype="int64")
    item_counts = clicked_df.groupby("item_id").size() if "item_id" in clicked_df.columns else pd.Series(dtype="int64")

    catalog_items = int(len(item_df)) if item_df is not None else int(clicked_df["item_id"].nunique())
    unique_users = int(interactions["user_id"].nunique()) if "user_id" in interactions.columns else 0
    unique_items = int(clicked_df["item_id"].nunique()) if "item_id" in clicked_df.columns else 0
    denom = unique_users * catalog_items

    row: dict[str, Any] = {
        "dataset": dataset,
        "rows_total": total_rows,
        "clicked_rows": clicked_rows,
        "unique_users": unique_users,
        "unique_items": unique_items,
        "catalog_items": catalog_items,
        "item_coverage": _safe_ratio(unique_items, catalog_items),
        "mean_clicks_per_user": _safe_stat(user_counts, "mean"),
        "median_clicks_per_user": _safe_stat(user_counts, "median"),
        "mean_clicks_per_item": _safe_stat(item_counts, "mean"),
        "median_clicks_per_item": _safe_stat(item_counts, "median"),
        "user_activity_gini": _gini(user_counts.to_numpy()),
        "item_popularity_gini": _gini(item_counts.to_numpy()),
        "sparsity_clicked": 1.0 - _safe_ratio(clicked_rows, denom),
    }

    if "impression_id" in interactions.columns:
        impression_sizes = clicked_df.groupby("impression_id").size()
        row["num_impressions"] = int(impression_sizes.size)
        row["mean_clicks_per_impression"] = _safe_stat(impression_sizes, "mean")
        row["median_clicks_per_impression"] = _safe_stat(impression_sizes, "median")

    if "neg_item_id_list" in interactions.columns:
        negative_counts = _sequence_lengths(interactions["neg_item_id_list"])
        row["mean_negative_candidates"] = _safe_stat(negative_counts, "mean")
        row["median_negative_candidates"] = _safe_stat(negative_counts, "median")
        row["mean_candidate_set_size"] = float((negative_counts + 1).mean()) if not negative_counts.empty else 0.0

    if "history_item_id_list" in interactions.columns:
        history_lengths = _sequence_lengths(interactions["history_item_id_list"])
        row["mean_history_length"] = _safe_stat(history_lengths, "mean")
        row["median_history_length"] = _safe_stat(history_lengths, "median")

    if "timestamp" in interactions.columns:
        timestamps = pd.to_numeric(interactions["timestamp"], errors="coerce").dropna()
        if not timestamps.empty:
            start_ts = float(timestamps.min())
            end_ts = float(timestamps.max())
            row["start_time"] = pd.to_datetime(start_ts, unit="s").isoformat()
            row["end_time"] = pd.to_datetime(end_ts, unit="s").isoformat()
            row["time_span_days"] = (end_ts - start_ts) / 86400.0

    if item_df is not None:
        if "category" in item_df.columns:
            row["num_categories"] = int(item_df["category"].replace("", pd.NA).dropna().nunique())
        if "sub_category" in item_df.columns:
            row["num_sub_categories"] = int(item_df["sub_category"].replace("", pd.NA).dropna().nunique())

    if has_label_column:
        non_click_rows = total_rows - clicked_rows
        row["non_click_rows"] = non_click_rows
        row["click_ratio"] = _safe_ratio(clicked_rows, total_rows)

    return row


def build_dataset_summary_table(datasets: list[str], base_path: str) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        interaction_df = load_interaction_dataframe(find_interaction_files(dataset, base_path))
        item_df = _maybe_load_item_dataframe(dataset, base_path)
        rows.append(summarize_interaction_dataframe(dataset, interaction_df, item_df))

    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def save_dataset_summary_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def run(datasets: list[str], base_path: str, output: str | None) -> Path:
    table = build_dataset_summary_table(datasets, base_path)
    output_path = Path(output) if output else Path("data_analysis/output/dataset_summary_table.csv")
    saved_path = save_dataset_summary_table(table, output_path)
    print(f"Saved: {saved_path}")
    print(table.to_string(index=False))
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis-style dataset summary statistics to CSV")
    parser.add_argument("--dataset", nargs="+", required=True, help="One or more dataset folder names under data/atomic_files")
    parser.add_argument("--base-path", default="data/atomic_files", help="Base directory containing dataset folders")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()
    run(args.dataset, args.base_path, args.output)


def _maybe_load_item_dataframe(dataset: str, base_path: str) -> pd.DataFrame | None:
    try:
        return load_item_dataframe(find_item_file(dataset, base_path))
    except FileNotFoundError:
        return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _safe_stat(values: pd.Series, op: str) -> float:
    if values.empty:
        return 0.0
    if op == "mean":
        return float(values.mean())
    if op == "median":
        return float(values.median())
    raise ValueError(f"Unsupported op: {op}")


def _sequence_lengths(values: pd.Series) -> pd.Series:
    string_values = values.fillna("").astype(str).str.strip()
    return string_values.apply(lambda value: 0 if value == "" else len(value.split())).astype(int)


def _gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0

    arr = np.sort(values.astype(float))
    total = float(arr.sum())
    if total == 0.0:
        return 0.0

    n = arr.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * arr)) / (n * total) - (n + 1) / n)


if __name__ == "__main__":
    main()
