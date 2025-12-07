"""
Experiment data loading and processing utilities.

This module provides functions for loading and organizing experimental results
from JSONL logs, extracting metrics, and computing statistics across runs.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from util.statistics import coefficient_of_variation, range_statistic, mean, std


def load_experiment_results(jsonl_path: str, experiment_id: str) -> List[Dict[str, Any]]:
    """
    Load all results for a specific experiment from JSONL file.
    
    Args:
        jsonl_path: Path to experiments.jsonl file
        experiment_id: Experiment ID to filter by
        
    Returns:
        List of result dictionaries for the specified experiment
        
    Raises:
        FileNotFoundError: If jsonl_path does not exist
        ValueError: If no results found for experiment_id
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")
    
    results = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get('experiment_id') == experiment_id:
                    results.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")
    
    return results


def extract_temporal_metrics(results: List[Dict[str, Any]], 
                            metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract and organize metrics across temporal windows.
    
    Computes mean, std, min, max for each metric within each window,
    and organizes the data for plotting and analysis.
    
    Args:
        results: List of experiment results
        metrics: List of metrics to extract (e.g., ['ndcg@10', 'recall@20'])
                If None, uses news-optimized metrics: ndcg@5, ndcg@10, mrr@5, hit@5
        
    Returns:
        Dictionary with structure:
            - windows: Dict[int, Dict] - Per-window statistics and values
            - metadata: Dict - Experiment metadata
            - metrics: List[str] - Metrics being tracked
    """
    if metrics is None:
        metrics = ['ndcg@5', 'ndcg@10', 'mrr@5', 'hit@5']
    
    # Sort results by window number and run number
    # Handle both new format (with window_info) and old format (without)
    def sort_key(x):
        window_info = x.get('window_info') or {}
        return (window_info.get('window_number', 0), window_info.get('run_number', 0))
    
    sorted_results = sorted(results, key=sort_key)
    
    # Group by window
    windows = {}
    for result in sorted_results:
        window_info = result.get('window_info') or {}
        window_num = window_info.get('window_number', 0)
        
        if window_num not in windows:
            windows[window_num] = {
                'runs': [],
                'window_info': window_info,
                'dataset_info': result.get('dataset_info', {})
            }
        
        # Extract test results for this run
        test_results = result.get('test_results', {})
        run_metrics = {metric: test_results.get(metric) for metric in metrics}
        windows[window_num]['runs'].append(run_metrics)
    
    # Calculate statistics per window
    window_data = {}
    for window_num, data in sorted(windows.items()):
        window_data[window_num] = {
            'mean': {},
            'std': {},
            'min': {},
            'max': {},
            'values': {},
            'info': data['window_info'],
            'dataset': data['dataset_info']
        }
        
        for metric in metrics:
            values = [run[metric] for run in data['runs'] if run[metric] is not None]
            if values:
                window_data[window_num]['mean'][metric] = np.mean(values)
                window_data[window_num]['std'][metric] = np.std(values)
                window_data[window_num]['min'][metric] = np.min(values)
                window_data[window_num]['max'][metric] = np.max(values)
                window_data[window_num]['values'][metric] = values
    
    # Extract metadata
    first_result = sorted_results[0]
    window_info = first_result.get('window_info') or {}
    run_info = first_result.get('run_info') or {}
    
    metadata = {
        'experiment_id': first_result.get('experiment_id'),
        'description': first_result.get('description'),
        'model': run_info.get('model'),
        'dataset': run_info.get('dataset'),
        'granularity': window_info.get('granularity', 'unknown'),
        'window_size': window_info.get('window_size'),
        'window_stride': window_info.get('window_stride'),
        'total_windows': len(windows),
        'runs_per_window': len(data['runs'])
    }
    
    return {
        'windows': window_data,
        'metadata': metadata,
        'metrics': metrics
    }


def compute_temporal_stability_stats(window_data: Dict[int, Dict], 
                                     metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute temporal stability statistics across all windows.
    
    Calculates coefficient of variation (CV), range, and other stability
    metrics by looking at how mean performance varies across windows.
    
    Args:
        window_data: Per-window statistics (output from extract_temporal_metrics)
        metrics: List of metric names to analyze
        
    Returns:
        Dictionary mapping metric names to stability statistics:
            - mean: Mean across all window means
            - std: Std across all window means
            - cv: Coefficient of variation (%)
            - min: Minimum window mean
            - max: Maximum window mean
            - range: max - min
    """
    stability_stats = {}
    
    for metric in metrics:
        # Get mean value from each window (temporal variance)
        window_means = [window_data[w]['mean'][metric] 
                       for w in sorted(window_data.keys())
                       if metric in window_data[w]['mean']]
        
        if window_means:
            stability_stats[metric] = {
                'mean': mean(window_means),
                'std': std(window_means, ddof=0),
                'cv': coefficient_of_variation(window_means, percent=True),
                'min': float(np.min(window_means)),
                'max': float(np.max(window_means)),
                'range': range_statistic(window_means)
            }
    
    return stability_stats
