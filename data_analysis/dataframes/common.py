import json
from pathlib import Path

import pandas as pd

BASIC_COLUMNS = [
    "experiment_id",
    "description",
    "model",
    "dataset",
    "seed",
    "window_number",
    "y",
]

CONTEXT_COLUMNS = [
    "start_unit",
    "end_unit",
    "num_users",
    "num_impressions",
    "avg_interactions_per_user",
    "avg_interactions_per_item",
    "user_activity_mean",
    "user_activity_std",
    "active_items",
    "item_popularity_mean",
    "item_popularity_std",
    "timestamp_min",
    "timestamp_max",
]


def read_experiments(results_path: Path) -> list[dict]:
    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def make_metric_dataframe(results_path: Path, metric: str, columns: str = "basic") -> pd.DataFrame:
    rows = []

    for row in read_experiments(results_path):
        window_info = row["window_info"]
        dataset_info = row.get("dataset_info", row.get("dataset_stats", {}))
        user_activity = dataset_info.get("user_activity", {})
        item_popularity = dataset_info.get("item_popularity", {})

        output_row = {
            "experiment_id": row["experiment_id"],
            "description": row["description"],
            "model": row["run_info"]["model"],
            "dataset": row["run_info"]["dataset"],
            "seed": row["run_info"]["seed"],
            "window_number": window_info["window_number"],
            "y": row["test_results"][metric],
        }

        if columns == "context":
            output_row.update(
                {
                    "start_unit": window_info.get("start_unit"),
                    "end_unit": window_info.get("end_unit"),
                    "train_units": window_info.get("train_units"),
                    "test_units": window_info.get("test_units"),
                    "num_users": dataset_info.get("num_users"),
                    "num_interactions_total": dataset_info.get("num_interactions_total"),
                    "num_impressions": dataset_info.get("num_impressions"),
                    "avg_interactions_per_user": dataset_info.get("avg_interactions_per_user"),
                    "avg_interactions_per_item": dataset_info.get("avg_interactions_per_item"),
                    "avg_items_per_impression": dataset_info.get("avg_items_per_impression"),
                    "sparsity_all_interactions": dataset_info.get("sparsity_all_interactions"),
                    "user_activity_mean": user_activity.get("mean"),
                    "user_activity_std": user_activity.get("std"),
                    "user_activity_cv": user_activity.get("cv"),
                    "user_activity_max": user_activity.get("max"),
                    "active_items": item_popularity.get("n"),
                    "item_popularity_mean": item_popularity.get("mean"),
                    "item_popularity_std": item_popularity.get("std"),
                    "item_popularity_cv": item_popularity.get("cv"),
                    "item_popularity_max": item_popularity.get("max"),
                    "timestamp_min": dataset_info.get("timestamp_min"),
                    "timestamp_max": dataset_info.get("timestamp_max"),
                }
            )

        rows.append(output_row)

    df = pd.DataFrame(rows).sort_values(["experiment_id", "window_number", "seed"]).reset_index(drop=True)

    if columns == "context":
        return df[BASIC_COLUMNS + CONTEXT_COLUMNS]

    return df[BASIC_COLUMNS]
