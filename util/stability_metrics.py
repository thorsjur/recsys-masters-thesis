import numpy as np
from typing import Dict, List

from util.statistics import coefficient_of_variation, range_statistic, mean, std


def calculate_stability_metrics(metric_values: List[float]) -> Dict[str, float]:
    """
    Calculate stability metrics for a set of metric values across multiple runs.
    
    Args:
        metric_values: List of metric values from different runs
        
    Returns:
        Dictionary containing:
            - mean: Average value
            - std: Standard deviation (sample std with ddof=1)
            - cv: Coefficient of Variation (%)
            - max_drop: Maximum drop (best - worst)
            - min: Minimum value
            - max: Maximum value
    """
    if not metric_values or len(metric_values) == 0:
        return {}
    
    values = np.array(metric_values)
    
    std_val = std(values, ddof=1) if len(values) > 1 else 0.0
    
    return {
        'mean': mean(values),
        'std': std_val,
        'cv': coefficient_of_variation(values, percent=True),
        'max_drop': range_statistic(values),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'n_runs': len(values)
    }


def aggregate_runs_stability(results: List[Dict[str, float]], metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Aggregate multiple run results and compute stability metrics.
    
    Args:
        results: List of result dictionaries from multiple runs
        metrics: List of metric names to analyze (if None, use all common metrics)
        
    Returns:
        Dictionary mapping metric names to their stability statistics
    """
    if not results:
        return {}
    
    # Determine which metrics to analyze
    if metrics is None:
        metrics = list(results[0].keys())
    
    stability_stats = {}
    
    for metric in metrics:
        # Collect values for this metric across all runs
        values = [run[metric] for run in results if metric in run]
        
        if values:
            stability_stats[metric] = calculate_stability_metrics(values)
    
    return stability_stats


def format_stability_report(stability_stats: Dict[str, Dict[str, float]], sort_by: str = 'cv') -> str:
    """
    Format stability statistics as a readable report.
    
    Args:
        stability_stats: Output from aggregate_runs_stability
        sort_by: Metric to sort by ('cv', 'max_drop', 'std')
        
    Returns:
        Formatted string report
    """
    if not stability_stats:
        return "No stability statistics available."
    
    # Sort metrics by the specified criterion
    sorted_metrics = sorted(
        stability_stats.items(),
        key=lambda x: x[1].get(sort_by, 0),
        reverse=True
    )
    
    lines = []
    lines.append("=" * 80)
    lines.append("STABILITY ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'CV (%)':<10} {'Max Drop':<10} {'Range':<15}")
    lines.append("-" * 80)
    
    for metric, stats in sorted_metrics:
        range_str = f"[{stats['min']:.4f}, {stats['max']:.4f}]"
        lines.append(
            f"{metric:<20} "
            f"{stats['mean']:<10.4f} "
            f"{stats['std']:<10.4f} "
            f"{stats['cv']:<10.2f} "
            f"{stats['max_drop']:<10.4f} "
            f"{range_str:<15}"
        )
    
    lines.append("=" * 80)
    lines.append(f"Total runs analyzed: {list(stability_stats.values())[0]['n_runs']}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - Lower CV (Coefficient of Variation) indicates higher stability")
    lines.append("  - Lower Max Drop indicates less risk from bad initializations")
    lines.append("=" * 80)
    
    return "\n".join(lines)
