from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_analysis.atomic_file import find_interaction_files, load_interaction_dataframe


def summarize_interaction_dataframe(
    dataset: str,
    interactions: pd.DataFrame,
) -> dict[str, Any]:
    user_counts = interactions.groupby("user_id").size()
    item_counts = interactions.groupby("item_id").size()
    impression_counts = interactions.groupby("impression_id").size()

    row: dict[str, Any] = {
        "dataset": dataset,
        "num_impressions": int(impression_counts.size),
        "num_interactions": int(len(interactions)),
        "num_unique_users": int(user_counts.size),
        "num_unique_items": int(item_counts.size),
        "duration_hours": _duration_hours(interactions),
        "history_length_at_interaction": (_describe_series(_sequence_lengths(interactions["history_item_id_list"]))),
        "clicks_per_impression": _describe_series(impression_counts),
        "negative_candidates_per_impression": (_describe_series(_negative_counts_per_impression(interactions))),
        "positive_interactions_per_user": _describe_series(user_counts),
        "positive_interactions_per_item": _describe_series(item_counts),
    }
    return row


def build_dataset_summary_table(datasets: list[str], base_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        interaction_df = load_interaction_dataframe(find_interaction_files(dataset, base_path))
        rows.append(summarize_interaction_dataframe(dataset, interaction_df))

    return sorted(rows, key=lambda row: str(row["dataset"]))


def save_dataset_summary_table(table: dict[str, Any] | list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return path


def run(datasets: list[str], base_path: str, output: str | None) -> Path:
    table = build_dataset_summary_table(datasets, base_path)
    output_path = Path(output) if output else Path("data_analysis/output/dataset_summary_table.json")
    saved_path = save_dataset_summary_table(table, output_path)
    print(f"Saved: {saved_path}")
    print(json.dumps(table, indent=2, sort_keys=True))
    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dataset summary statistics to JSON")
    parser.add_argument(
        "--dataset", nargs="+", required=True, help="One or more dataset folder names under data/atomic_files"
    )
    parser.add_argument("--base-path", default="data/atomic_files", help="Base directory containing dataset folders")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()
    run(args.dataset, args.base_path, args.output)


def _sequence_lengths(values: pd.Series) -> pd.Series:
    string_values = values.fillna("").astype(str).str.strip()
    return string_values.apply(lambda value: 0 if value == "" else len(value.split())).astype(int)


def _negative_counts_per_impression(interactions: pd.DataFrame) -> pd.Series:
    per_impression = interactions.groupby("impression_id", sort=False)["neg_item_id_list"].first()
    return _sequence_lengths(per_impression)


def _duration_hours(interactions: pd.DataFrame) -> float:
    if "timestamp" not in interactions.columns:
        return 0.0

    timestamps = pd.to_numeric(interactions["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return 0.0
    return float((timestamps.max() - timestamps.min()) / 3600.0)


def _describe_series(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return _empty_stats()
    return {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _empty_stats() -> dict[str, float | int]:
    return {
        "count": 0,
        "min": 0,
        "max": 0,
        "mean": 0.0,
        "median": 0.0,
        "std": 0.0,
    }


if __name__ == "__main__":
    main()
