"""
Temporal stability visualization for recommendation models.

This module provides functions to visualize how model performance changes
across temporal windows, helping identify temporal drift and stability patterns.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime


def load_experiment_results(jsonl_path: str, experiment_id: str) -> List[Dict[str, Any]]:
    """
    Load all results for a specific experiment from JSONL file.
    
    Args:
        jsonl_path: Path to experiments.jsonl file
        experiment_id: Experiment ID to filter by
        
    Returns:
        List of result dictionaries for the specified experiment
    """
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
    
    Args:
        results: List of experiment results
        metrics: List of metrics to extract (e.g., ['ndcg@10', 'recall@20'])
                If None, uses news-optimized metrics: ndcg@10, recall@10, mrr@1, hit@5
        
    Returns:
        Dictionary with organized metrics data and metadata
    """
    if metrics is None:
        metrics = ['ndcg@5', 'ndcg@10', 'mrr@5', 'hit@5']
    
    # Sort results by window number and run number
    sorted_results = sorted(results, 
                          key=lambda x: (x.get('window_info', {}).get('window_number', 0),
                                       x.get('window_info', {}).get('run_number', 0)))
    
    # Group by window
    windows = {}
    for result in sorted_results:
        window_info = result.get('window_info', {})
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
    metadata = {
        'experiment_id': first_result.get('experiment_id'),
        'description': first_result.get('description'),
        'model': first_result.get('run_info', {}).get('model'),
        'dataset': first_result.get('run_info', {}).get('dataset'),
        'granularity': first_result.get('window_info', {}).get('granularity', 'unknown'),
        'window_size': first_result.get('window_info', {}).get('window_size'),
        'window_stride': first_result.get('window_info', {}).get('window_stride'),
        'total_windows': len(windows),
        'runs_per_window': len(data['runs'])
    }
    
    return {
        'windows': window_data,
        'metadata': metadata,
        'metrics': metrics
    }


def plot_temporal_stability(experiment_id: str,
                           jsonl_path: str = 'output/results/experiments.jsonl',
                           metrics: Optional[List[str]] = None,
                           output_path: Optional[str] = None,
                           show_std: bool = True,
                           show_individual_runs: bool = False,
                           figsize: tuple = (14, 8)):
    """
    Plot temporal stability of a model across sliding windows.
    
    Args:
        experiment_id: Experiment ID to visualize
        jsonl_path: Path to experiments.jsonl file
        metrics: List of metrics to plot (default: ndcg@10, recall@10, mrr@1, hit@5)
        output_path: Path to save PDF (default: plot/output/{experiment_id}_temporal_stability.pdf)
        show_std: Whether to show standard deviation bands
        show_individual_runs: Whether to show individual run points
        figsize: Figure size (width, height)
    """
    # Load and process data
    results = load_experiment_results(jsonl_path, experiment_id)
    data = extract_temporal_metrics(results, metrics)
    
    windows = data['windows']
    metadata = data['metadata']
    metrics = data['metrics']
    
    # Set up plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        window_numbers = sorted(windows.keys())
        means = [windows[w]['mean'][metric] for w in window_numbers]
        stds = [windows[w]['std'][metric] for w in window_numbers]
        
        # Plot mean line
        ax.plot(window_numbers, means, 
               color=colors[idx], linewidth=2.5, 
               marker='o', markersize=8, label='Mean')
        
        # Plot standard deviation band
        if show_std and any(s > 0 for s in stds):
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(window_numbers, 
                           means_arr - stds_arr, 
                           means_arr + stds_arr,
                           alpha=0.2, color=colors[idx], label='±1 std')
        
        # Plot individual runs
        if show_individual_runs:
            for w in window_numbers:
                values = windows[w]['values'][metric]
                ax.scatter([w] * len(values), values, 
                         alpha=0.4, s=30, color=colors[idx])
        
        # Formatting
        granularity_label = metadata['granularity'].capitalize() if metadata['granularity'] != 'unknown' else 'Time Unit'
        ax.set_xlabel(f'Time ({granularity_label}s)', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        
        # Set x-axis to show temporal units instead of window numbers
        if metadata['granularity'] != 'unknown':
            # Get the end time of each test window as the x-axis value
            x_values = []
            x_labels = []
            for w in window_numbers:
                info = windows[w]['info']
                test_end = info.get('end_unit', 0)
                x_values.append(test_end)
                
                # Create label showing the test period
                test_range = info.get('test_range', '')
                x_labels.append(test_range)
            
            # Re-plot with temporal x-axis
            ax.clear()
            means = [windows[w]['mean'][metric] for w in window_numbers]
            stds = [windows[w]['std'][metric] for w in window_numbers]
            
            # Plot mean line with temporal x-axis
            ax.plot(x_values, means, 
                   color=colors[idx], linewidth=2.5, 
                   marker='o', markersize=8, label='Mean')
            
            # Plot standard deviation band
            if show_std and any(s > 0 for s in stds):
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.fill_between(x_values, 
                               means_arr - stds_arr, 
                               means_arr + stds_arr,
                               alpha=0.2, color=colors[idx], label='±1 std')
            
            # Plot individual runs
            if show_individual_runs:
                for i, w in enumerate(window_numbers):
                    values = windows[w]['values'][metric]
                    ax.scatter([x_values[i]] * len(values), values, 
                             alpha=0.4, s=30, color=colors[idx])
            
            # Re-apply formatting
            ax.set_xlabel(f'Time ({granularity_label}s)', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.upper()} Over Time', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)
            
            # Set custom x-tick labels showing test ranges
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            
            # Only show every other label if too many
            if len(window_numbers) > 10:
                for i, label in enumerate(ax.get_xticklabels()):
                    if i % 2 != 0:
                        label.set_visible(False)
    
    # Add overall title with metadata (single line to save space)
    main_title = f"Temporal Stability: {metadata['model']} on {metadata['dataset']} ({metadata['total_windows']} windows, {metadata['runs_per_window']} runs/window)"
    fig.suptitle(main_title, fontsize=13, fontweight='bold', y=0.98)
    
    # Add experiment info as subtitle
    subtitle = f"{metadata['experiment_id']}: {metadata['description']}" if metadata['description'] else metadata['experiment_id']
    fig.text(0.5, 0.92, subtitle, ha='center', fontsize=9, style='italic', color='#555')
    
    # Add window configuration info
    config_text = (
        f"Window: {metadata['window_size']} {metadata['granularity']}s "
        f"(stride: {metadata['window_stride']} {metadata['granularity']}s)"
    )
    fig.text(0.5, 0.02, config_text, ha='center', fontsize=9, color='#666')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    # Save to PDF
    if output_path is None:
        output_dir = Path('plot/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{experiment_id}_temporal_stability.pdf"
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    # Calculate and print stability metrics
    print(f"\n{'='*70}")
    print(f"Temporal Stability Analysis - {metadata['experiment_id']}")
    print(f"{'='*70}")
    print(f"Model: {metadata['model']} | Dataset: {metadata['dataset']}")
    print(f"Windows: {metadata['total_windows']} | Runs per window: {metadata['runs_per_window']}")
    print(f"{'-'*70}")
    
    for metric in metrics:
        all_means = [windows[w]['mean'][metric] for w in sorted(windows.keys())]
        cv = (np.std(all_means) / np.mean(all_means)) * 100 if np.mean(all_means) > 0 else 0
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean across windows: {np.mean(all_means):.4f}")
        print(f"  Std across windows:  {np.std(all_means):.4f}")
        print(f"  Coefficient of Variation: {cv:.2f}%")
        print(f"  Min: {np.min(all_means):.4f} | Max: {np.max(all_means):.4f}")
        print(f"  Range: {np.max(all_means) - np.min(all_means):.4f}")
    
    print(f"{'='*70}\n")
    
    return fig, data


def main():
    parser = argparse.ArgumentParser(
        description='Visualize temporal stability of recommendation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot with default metrics
  python -m plot.temporal_stability --experiment-id exp_tfidf_temporal_36h
  
  # Custom metrics and output path
  python -m plot.temporal_stability --experiment-id exp001 \\
      --metrics ndcg@10 recall@20 \\
      --output my_plot.pdf
  
  # Show individual run points with std bands
  python -m plot.temporal_stability --experiment-id exp001 \\
      --show-runs --show-std
        """
    )
    
    parser.add_argument('--experiment-id', required=True,
                       help='Experiment ID to visualize')
    parser.add_argument('--jsonl-path', default='output/results/experiments.jsonl',
                       help='Path to experiments.jsonl file')
    parser.add_argument('--metrics', nargs='+',
                       help='Metrics to plot (default: ndcg@10 recall@10 mrr@1 hit@5)')
    parser.add_argument('--output', '-o',
                       help='Output PDF path (default: plot/output/{experiment_id}_temporal_stability.pdf)')
    parser.add_argument('--show-runs', action='store_true',
                       help='Show individual run points')
    parser.add_argument('--show-std', action='store_true', default=True,
                       help='Show standard deviation bands (default: True)')
    parser.add_argument('--no-std', action='store_true',
                       help='Hide standard deviation bands')
    parser.add_argument('--figsize', nargs=2, type=float, default=[14, 8],
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    try:
        plot_temporal_stability(
            experiment_id=args.experiment_id,
            jsonl_path=args.jsonl_path,
            metrics=args.metrics,
            output_path=args.output,
            show_std=not args.no_std,
            show_individual_runs=args.show_runs,
            figsize=tuple(args.figsize)
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure {args.jsonl_path} exists")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == '__main__':
    main()
