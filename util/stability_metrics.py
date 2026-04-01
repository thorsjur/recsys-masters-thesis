import numpy as np
from typing import Dict, List, Optional

from util.constants import SEPARATOR, SUB_SEPARATOR
from util.statistics import coefficient_of_variation, range_statistic, mean, std


def calculate_stability_metrics(values: List[float]) -> Dict[str, float]:
    """Calculate stability metrics (mean, std, cv, range) for a set of values across runs."""
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


def format_stability_report(stats: Dict[str, Dict[str, float]], sort_by: str = "cv") -> str:
    """Format stability statistics as a readable report, sorted by specified key."""
    if not stats:
        return "No stability statistics available."

    sorted_items = sorted(stats.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)

    lines = [
        SEPARATOR,
        "STABILITY ANALYSIS",
        SEPARATOR,
        f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'CV (%)':<10} {'Max Drop':<10} {'Range':<15}",
        SUB_SEPARATOR,
    ]

    for metric, s in sorted_items:
        range_str = f"[{s['min']:.4f}, {s['max']:.4f}]"
        lines.append(
            f"{metric:<20} {s['mean']:<10.4f} {s['std']:<10.4f} "
            f"{s['cv']:<10.2f} {s['max_drop']:<10.4f} {range_str:<15}"
        )

    n_runs = list(stats.values())[0].get("n_runs", "N/A")
    lines.extend(
        [
            SEPARATOR,
            f"Total runs analyzed: {n_runs}",
            SEPARATOR,
        ]
    )

    return "\n".join(lines)
