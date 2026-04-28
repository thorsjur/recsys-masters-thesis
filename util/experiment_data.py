import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from util.constants import DEFAULT_METRICS
from util.statistics import coefficient_of_variation, range_statistic, mean, std

logger = logging.getLogger(__name__)


def load_experiment_results(jsonl_path: str, experiment_id: str) -> List[Dict[str, Any]]:
    """Load all results for a specific experiment from JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    results = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("experiment_id") == experiment_id:
                    results.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")
    return results


def extract_temporal_metrics(results: List[Dict[str, Any]], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract and organize metrics across temporal windows.

    Returns dict with 'windows' (per-window stats), 'metadata', and 'metrics' keys.
    """
    metrics = metrics or DEFAULT_METRICS

    # Sort by window number, then run number
    def sort_key(x):
        info = x.get("window_info") or {}
        return (info.get("window_number", 0), info.get("run_number", 0))

    sorted_results = sorted(results, key=sort_key)

    # Group runs by window
    windows = {}
    for result in sorted_results:
        win_info = result.get("window_info") or {}
        win_num = win_info.get("window_number", 0)

        if win_num not in windows:
            windows[win_num] = {"runs": [], "window_info": win_info, "dataset_info": result.get("dataset_info", {})}

        test_results = result.get("test_results", {})
        if any(m not in test_results for m in metrics):
            missing = [m for m in metrics if m not in test_results]
            raise ValueError(
                f"Metrics {missing} not found in test_results for window {win_num}, available: {list(test_results.keys())}"
            )

        windows[win_num]["runs"].append({m: test_results.get(m) for m in metrics})

    # Calculate per-window statistics
    window_data = {}
    for win_num, data in sorted(windows.items()):
        stats = {
            "mean": {},
            "std": {},
            "min": {},
            "max": {},
            "values": {},
            "info": data["window_info"],
            "dataset": data["dataset_info"],
        }

        for metric in metrics:
            values = [r[metric] for r in data["runs"] if r[metric] is not None]
            if values:
                stats["mean"][metric] = np.mean(values)
                stats["std"][metric] = np.std(values)
                stats["min"][metric] = np.min(values)
                stats["max"][metric] = np.max(values)
                stats["values"][metric] = values

        window_data[win_num] = stats

    # Extract metadata from first result
    first = sorted_results[0]
    win_info = first.get("window_info") or {}
    run_info = first.get("run_info") or {}

    return {
        "windows": window_data,
        "metadata": {
            "experiment_id": first.get("experiment_id"),
            "description": first.get("description"),
            "model": run_info.get("model"),
            "dataset": run_info.get("dataset"),
            "granularity": win_info.get("granularity", "unknown"),
            "window_size": win_info.get("window_size"),
            "window_stride": win_info.get("window_stride"),
            "total_windows": len(windows),
            "runs_per_window": len(data["runs"]),
        },
        "metrics": metrics,
    }


def compute_temporal_stability_stats(window_data: Dict[int, Dict], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute stability statistics across the window means (mean across seeds per window)."""
    stats = {}
    for metric in metrics:
        window_means = [
            window_data[w]["mean"][metric] for w in sorted(window_data.keys()) if metric in window_data[w]["mean"]
        ]
        if window_means:
            stats[metric] = {
                "mean": mean(window_means),
                "std": std(window_means),
                "cv": coefficient_of_variation(window_means),
                "min": float(np.min(window_means)),
                "max": float(np.max(window_means)),
                "range": range_statistic(window_means),
            }
    return stats
