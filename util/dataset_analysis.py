from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from recbole.data.dataset import Dataset

from util.constants import SEPARATOR
from util.statistics import basic_stats


def collect_recbole_dataset_stats(dataset: Dataset, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    inter_feat = dataset.inter_feat
    if inter_feat is None:
        raise ValueError("Dataset has no interaction features")

    label_field = config.get("RATING_FIELD", "label") if config else "label"

    num_users = int(dataset.user_num)
    num_items = int(dataset.item_num)
    total_interactions = int(dataset.inter_num)

    if label_field in inter_feat:
        labels = inter_feat[label_field].to_numpy()
        num_positive = int(np.sum(labels == 1))
        num_negative = int(np.sum(labels == 0))
    else:
        num_positive = total_interactions
        num_negative = 0

    denom = num_users * num_items
    sparsity_positive = float(1 - num_positive / denom) if denom else 1.0
    sparsity_total = float(1 - total_interactions / denom) if denom else 1.0

    stats: Dict[str, Any] = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions_total": total_interactions,
        "num_positive_interactions": num_positive,
        "num_negative_interactions": num_negative,
        "positive_ratio": float(num_positive / total_interactions) if total_interactions else 0.0,
        "negative_ratio": float(num_negative / total_interactions) if total_interactions else 0.0,
        "sparsity_positive_only": sparsity_positive,
        "sparsity_all_interactions": sparsity_total,
        "avg_positive_per_user": float(num_positive / num_users) if num_users else 0.0,
        "avg_positive_per_item": float(num_positive / num_items) if num_items else 0.0,
        "avg_negative_per_user": float(num_negative / num_users) if num_users else 0.0,
        "avg_negative_per_item": float(num_negative / num_items) if num_items else 0.0,
        "avg_interactions_per_user": float(total_interactions / num_users) if num_users else 0.0,
        "avg_interactions_per_item": float(total_interactions / num_items) if num_items else 0.0,
        "has_item_features": dataset.item_feat is not None,
        "has_user_features": dataset.user_feat is not None,
    }

    user_field = dataset.uid_field
    item_field = dataset.iid_field
    if user_field in inter_feat and item_field in inter_feat:
        user_ids = inter_feat[user_field].to_numpy()
        item_ids = inter_feat[item_field].to_numpy()

        if label_field in inter_feat:
            labels = inter_feat[label_field].to_numpy()
            positive_mask = labels == 1
            negative_mask = labels == 0

            user_positive_counts = pd.Series(user_ids[positive_mask]).value_counts()
            stats["user_positive_activity"] = basic_stats(user_positive_counts.to_numpy())

            item_positive_counts = pd.Series(item_ids[positive_mask]).value_counts()
            stats["item_positive_popularity"] = basic_stats(item_positive_counts.to_numpy())

            if num_negative > 0:
                user_negative_counts = pd.Series(user_ids[negative_mask]).value_counts()
                item_negative_counts = pd.Series(item_ids[negative_mask]).value_counts()
                stats["user_negative_activity"] = basic_stats(user_negative_counts.to_numpy())
                stats["item_negative_popularity"] = basic_stats(item_negative_counts.to_numpy())
        else:
            stats["user_activity"] = basic_stats(pd.Series(user_ids).value_counts().to_numpy())
            stats["item_popularity"] = basic_stats(pd.Series(item_ids).value_counts().to_numpy())

    if "impression_id" in inter_feat:
        impression_ids = inter_feat["impression_id"].to_numpy()
        unique_impressions = len(np.unique(impression_ids))
        stats["num_impressions"] = unique_impressions
        stats["avg_items_per_impression"] = float(total_interactions / unique_impressions) if unique_impressions else 0.0

        if label_field in inter_feat:
            labels = inter_feat[label_field].to_numpy()
            impression_df = pd.DataFrame({"impression_id": impression_ids, "label": labels})
            impression_stats = impression_df.groupby("impression_id").agg(
                total_items=("label", "count"),
                positive_items=("label", "sum"),
            )
            impression_stats["negative_items"] = impression_stats["total_items"] - impression_stats["positive_items"]
            stats["avg_positive_per_impression"] = float(impression_stats["positive_items"].mean())
            stats["avg_negative_per_impression"] = float(impression_stats["negative_items"].mean())
            stats["impression_size_stats"] = basic_stats(impression_stats["total_items"].to_numpy())

    time_field = config.get("TIME_FIELD", "timestamp") if config else "timestamp"
    if time_field in inter_feat:
        timestamps = inter_feat[time_field].to_numpy()
        ts_min = float(np.min(timestamps))
        ts_max = float(np.max(timestamps))
        span = ts_max - ts_min
        stats["timestamp_min"] = ts_min
        stats["timestamp_max"] = ts_max
        stats["time_span_seconds"] = span
        stats["time_span_hours"] = span / 3600
        stats["time_span_days"] = span / 86400

    return stats


def format_dataset_stats_summary(stats: Dict[str, Any]) -> str:
    lines = [
        SEPARATOR,
        "Dataset Statistics Summary",
        SEPARATOR,
        f"Users: {stats['num_users']:,}",
        f"Items: {stats['num_items']:,}",
        f"Total Interactions: {stats['num_interactions_total']:,}",
        f"  - Positive (clicks): {stats['num_positive_interactions']:,} ({stats['positive_ratio']:.1%})",
        f"  - Negative (non-clicks): {stats['num_negative_interactions']:,} ({stats['negative_ratio']:.1%})",
        "",
        f"Sparsity (positive only): {stats['sparsity_positive_only']:.6f}",
        f"Sparsity (all interactions): {stats['sparsity_all_interactions']:.6f}",
        "",
        f"Avg positive interactions per user: {stats['avg_positive_per_user']:.2f}",
        f"Avg positive interactions per item: {stats['avg_positive_per_item']:.2f}",
        f"Avg negative interactions per user: {stats['avg_negative_per_user']:.2f}",
        f"Avg negative interactions per item: {stats['avg_negative_per_item']:.2f}",
    ]

    if "num_impressions" in stats:
        lines.extend(
            [
                "",
                f"Impressions: {stats['num_impressions']:,}",
                f"Avg items per impression: {stats['avg_items_per_impression']:.2f}",
                f"Avg clicks per impression: {stats.get('avg_positive_per_impression', 0):.2f}",
            ]
        )

    if "time_span_days" in stats:
        lines.extend(["", f"Time span: {stats['time_span_days']:.1f} days ({stats['time_span_hours']:.1f} hours)"])

    lines.extend(["", f"Has item features: {stats['has_item_features']}", f"Has user features: {stats['has_user_features']}", SEPARATOR])
    return "\n".join(lines)


def load_temporal_interaction_data(
    dataset_path: str,
    dataset_name: str,
    granularity: str,
    time_units: range,
    positive_only: bool = True,
) -> pd.DataFrame:
    dataset_dir = Path(dataset_path) / dataset_name
    dfs = []

    for unit in time_units:
        file_path = dataset_dir / f"{dataset_name}.{granularity}_{unit}.inter"
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path, sep="\t")
        df.columns = [c.split(":")[0] for c in df.columns]
        df["time_unit"] = unit
        dfs.append(df)

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        if positive_only and "label" in result.columns:
            return result[result["label"] == 1].copy()
        return result

    full_inter_path = dataset_dir / f"{dataset_name}.inter"
    if not full_inter_path.exists():
        raise ValueError(f"No data files found for {dataset_name} in range {time_units}")

    result = pd.read_csv(full_inter_path, sep="\t")
    result.columns = [c.split(":")[0] for c in result.columns]

    if "timestamp" not in result.columns:
        raise ValueError(f"Cannot derive time units for {dataset_name}: missing timestamp column in {full_inter_path}")

    seconds_per_unit = _seconds_per_unit(granularity)
    timestamps = pd.to_numeric(result["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Cannot derive time units for {dataset_name}: timestamp column contains non-numeric values")

    start_timestamp = float(timestamps.min())
    result["time_unit"] = ((timestamps - start_timestamp) / seconds_per_unit).astype(int) + 1
    result = result[result["time_unit"].isin(time_units)].copy()

    if result.empty:
        raise ValueError(f"No data rows found for {dataset_name} in range {time_units}")

    if positive_only and "label" in result.columns:
        return result[result["label"] == 1].copy()
    return result


def _seconds_per_unit(granularity: str) -> int:
    if granularity == "hour":
        return 3600
    if granularity == "day":
        return 86400
    raise ValueError(f"Unsupported granularity: {granularity}")


def compute_temporal_statistics(
    df: pd.DataFrame,
    granularity: str,
    start_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    start_ts = float(start_timestamp) if start_timestamp is not None else float(df["timestamp"].min())

    per_unit = df.groupby("time_unit").size()
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    return {
        "total_interactions": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_items": df["item_id"].nunique(),
        "time_span_seconds": float(df["timestamp"].max() - df["timestamp"].min()),
        "first_timestamp": float(df["timestamp"].min()),
        "last_timestamp": float(df["timestamp"].max()),
        "start_datetime": pd.to_datetime(start_ts, unit="s").isoformat(),
        "granularity": granularity,
        "interactions_per_unit": basic_stats(per_unit.to_numpy()),
        "user_activity": basic_stats(user_counts.to_numpy()),
        "item_popularity": basic_stats(item_counts.to_numpy()),
    }
