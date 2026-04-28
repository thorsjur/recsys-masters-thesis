import numpy as np
from typing import Dict, List, Optional

from util.constants import SEPARATOR, SUB_SEPARATOR
from util.statistics import coefficient_of_variation, range_statistic, mean, std


def calculate_stability_metrics(values: List[float]) -> Dict[str, float]:
    """Calculate stability metrics for a set of values across runs."""
    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean": mean(arr),
        "std": std(arr, ddof=1) if len(arr) > 1 else 0.0,
        "cv": coefficient_of_variation(arr),
        "max_drop": range_statistic(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_runs": len(arr),
    }


def aggregate_runs_stability(
    results: List[Dict[str, float]], metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple runs and compute stability per metric.

    Args:
        results: List of {metric_name: value} dicts from multiple runs
        metrics: Metrics to analyze (default: all keys from first result)
    """
    if not results:
        return {}

    if metrics is None:
        metrics = list(results[0].keys())

    return {metric: calculate_stability_metrics([r[metric] for r in results if metric in r]) for metric in metrics}