"""Sliding window validation and statistics computation."""

from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

from util.constants import SEPARATOR, SUB_SEPARATOR


def _split_by_time_unit(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Filter dataframe to rows within time_unit range [start, end]."""
    return df[df["time_unit"].between(start, end)]


def _filter_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only positive interactions (label == 1).

    Non-clicked impressions (label == 0) are excluded from analysis.
    """
    if "label" in df.columns:
        return df[df["label"] == 1]
    return df


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Compute ratio safely, returning 0 if denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


def compute_window_statistics(
    df: pd.DataFrame, window_info: Dict[str, Any], granularity: str = "hour"
) -> Dict[str, Any]:
    """
    Compute statistics for a single temporal window.

    Only positive interactions (label == 1) are counted. Non-clicked impressions
    (label == 0) are excluded from all statistics.
    """
    train_start = window_info.get("train_start_unit") or window_info.get("start_unit") or 0
    train_units = window_info.get("train_units", 0)
    train_end = train_start + train_units - 1
    test_start = window_info.get("test_start_unit") or (train_end + 1)
    test_end = window_info.get("end_unit") or test_start

    train_df = _filter_positive(_split_by_time_unit(df, train_start, train_end))
    test_df = _filter_positive(_split_by_time_unit(df, test_start, test_end))

    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())
    train_items = set(train_df["item_id"].unique())
    test_items = set(test_df["item_id"].unique())

    # Cold-start entities in test but not in train
    new_users = test_users - train_users
    new_items = test_items - train_items

    # Overlap
    user_overlap = len(train_users & test_users)
    item_overlap = len(train_items & test_items)

    return {
        "window_number": window_info.get("window_number"),
        "train_start_unit": train_start,
        "train_end_unit": train_end,
        "test_start_unit": test_start,
        "test_end_unit": test_end,
        "train_interactions": len(train_df),
        "test_interactions": len(test_df),
        "train_users": len(train_users),
        "test_users": len(test_users),
        "train_items": len(train_items),
        "test_items": len(test_items),
        "new_users_in_test": len(new_users),
        "new_items_in_test": len(new_items),
        "cold_start_user_ratio": _safe_ratio(len(new_users), len(test_users)),
        "cold_start_item_ratio": _safe_ratio(len(new_items), len(test_items)),
        "user_overlap": user_overlap,
        "item_overlap": item_overlap,
        "user_overlap_ratio": _safe_ratio(user_overlap, len(train_users)),
        "item_overlap_ratio": _safe_ratio(item_overlap, len(train_items)),
    }


def compute_all_window_statistics(
    df: pd.DataFrame, windows: List[Dict[str, Any]], granularity: str = "hour"
) -> pd.DataFrame:
    """Compute statistics for all windows, returning a DataFrame with one row per window."""
    # Deduplicate windows by window_number (filter out None keys)
    unique: Dict[int, Dict[str, Any]] = {
        int(w["window_number"]): w for w in windows if w.get("window_number") is not None
    }

    stats = [compute_window_statistics(df, unique[n], granularity) for n in sorted(unique.keys())]

    return pd.DataFrame(stats).sort_values("window_number").reset_index(drop=True)


def generate_validation_report(
    stats_df: pd.DataFrame, experiment_id: str, dataset_name: str, granularity: str = "hour"
) -> str:
    """Generate a text report validating the sliding window methodology."""
    lines = [
        SEPARATOR,
        f"Sliding Window Validation Report",
        f"Experiment: {experiment_id}",
        f"Dataset: {dataset_name}",
        SEPARATOR,
        "",
        f"Total Windows: {len(stats_df)}",
        f"Granularity: {granularity}",
        "",
        SUB_SEPARATOR,
        "DATA DISTRIBUTION ACROSS WINDOWS",
        SUB_SEPARATOR,
        "",
        f"{'Win':>3} | {'Train':>8} | {'Test':>8} | {'Train':>7} | {'Test':>7} | {'Train':>7} | {'Test':>7}",
        f"{'#':>3} | {'Interact':>8} | {'Interact':>8} | {'Users':>7} | {'Users':>7} | {'Items':>7} | {'Items':>7}",
        SUB_SEPARATOR,
    ]

    for _, row in stats_df.iterrows():
        lines.append(
            f"{int(row['window_number']):3d} | "
            f"{int(row['train_interactions']):8,d} | {int(row['test_interactions']):8,d} | "
            f"{int(row['train_users']):7,d} | {int(row['test_users']):7,d} | "
            f"{int(row['train_items']):7,d} | {int(row['test_items']):7,d}"
        )

    # Summary stats helper
    def summarize(col):
        return (
            f"Mean: {stats_df[col].mean():,.0f}, Std: {stats_df[col].std():,.0f}, "
            f"Range: {stats_df[col].min():,.0f}-{stats_df[col].max():,.0f}"
        )

    lines.extend(
        [
            SUB_SEPARATOR,
            "",
            "SUMMARY STATISTICS:",
            f"  Train Interactions - {summarize('train_interactions')}",
            f"  Test Interactions  - {summarize('test_interactions')}",
            f"  Train Items        - {summarize('train_items')}",
            f"  Test Items         - {summarize('test_items')}",
            "",
            SUB_SEPARATOR,
            "COLD-START ANALYSIS",
            SUB_SEPARATOR,
            "",
            f"{'Win':>3} | {'New Users':>10} | {'New Items':>10} | {'User CS%':>9} | {'Item CS%':>9}",
            SUB_SEPARATOR,
        ]
    )

    for _, row in stats_df.iterrows():
        lines.append(
            f"{int(row['window_number']):3d} | "
            f"{int(row['new_users_in_test']):10,d} | {int(row['new_items_in_test']):10,d} | "
            f"{row['cold_start_user_ratio']*100:8.2f}% | {row['cold_start_item_ratio']*100:8.2f}%"
        )

    lines.extend(
        [
            SUB_SEPARATOR,
            "",
            "COLD-START SUMMARY:",
            f"  New Users (mean): {stats_df['cold_start_user_ratio'].mean()*100:.2f}%",
            f"  New Items (mean): {stats_df['cold_start_item_ratio'].mean()*100:.2f}%",
            "",
        ]
    )

    return "\n".join(lines)


def export_statistics_table(stats_df: pd.DataFrame, output_path: str, format: str = "latex") -> str:
    """Export window statistics table in publication-ready format (latex, csv, or markdown)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = stats_df[
        [
            "window_number",
            "train_interactions",
            "test_interactions",
            "train_items",
            "test_items",
            "new_items_in_test",
            "cold_start_item_ratio",
        ]
    ].copy()

    table.columns = ["Window", "Train Int.", "Test Int.", "Train Items", "Test Items", "New Items", "Cold-Start %"]
    table["Cold-Start %"] = (table["Cold-Start %"] * 100).round(2)

    if format == "latex":
        content = table.to_latex(
            index=False,
            float_format="%.2f",
            caption="Data distribution across temporal windows.",
            label="tab:window_statistics",
        )
    elif format == "csv":
        content = table.to_csv(index=False)
    elif format == "markdown":
        content = table.to_markdown(index=False, floatfmt=".2f")
    else:
        raise ValueError(f"Unknown format: {format}")

    with open(path, "w") as f:
        f.write(content)
    return str(path)
