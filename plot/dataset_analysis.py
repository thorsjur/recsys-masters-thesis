"""
Dataset temporal analysis visualization.

This script generates comprehensive visualizations of dataset temporal properties
based on experimental configurations stored in experiments.jsonl.

Features:
- Interaction volume over time with window overlays
- Time-of-day/week patterns
- User activity and item popularity distributions
- Window-wise statistics comparison
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.experiment_data import load_experiment_results
from util.dataset_analysis import (
    load_temporal_interaction_data,
    compute_temporal_statistics,
    plot_interactions_over_time,
    plot_time_of_day_pattern,
    plot_user_item_distributions,
    plot_window_statistics,
    plot_interaction_heatmap
)


def analyze_dataset_from_experiment(experiment_id: str,
                                   jsonl_path: str = 'output/results/experiments.jsonl',
                                   dataset_path: str = 'datasets/atomic_files',
                                   output_dir: Optional[str] = None,
                                   start_timestamp: Optional[float] = None,
                                   figsize: tuple = (12, 6)) -> Dict[str, Any]:
    """
    Generate comprehensive dataset analysis from experiment configuration.
    
    Args:
        experiment_id: Experiment ID to analyze
        jsonl_path: Path to experiments.jsonl file
        dataset_path: Path to dataset directory
        output_dir: Directory to save PDFs (default: plot/output/)
        start_timestamp: Optional start timestamp (uses first interaction if None)
        figsize: Figure size (width, height) for each plot
        
    Returns:
        Dictionary with analysis results and statistics
    """
    # Load experiment results
    results = load_experiment_results(jsonl_path, experiment_id)
    
    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")
    
    # Extract configuration from first result
    first_result = results[0]
    window_info = first_result.get('window_info')
    
    if not window_info:
        raise ValueError(f"Experiment {experiment_id} has no window_info - not a temporal experiment")
    
    dataset_name = first_result.get('run_info', {}).get('dataset', 'unknown')
    granularity = window_info.get('granularity', 'hour')
    
    # Collect all windows (deduplicate by window_number since there can be multiple runs per window)
    windows_by_number = {}
    for result in results:
        w_info = result.get('window_info', {})
        if w_info:
            win_num = w_info.get('window_number')
            if win_num is not None and win_num not in windows_by_number:
                windows_by_number[win_num] = w_info
    
    # Sort by window number
    all_windows = [windows_by_number[k] for k in sorted(windows_by_number.keys())]
    
    # Determine time range to load
    min_unit = min(w.get('start_unit', 0) for w in all_windows)
    max_unit = max(w.get('end_unit', 0) for w in all_windows)
    
    print(f"Loading {dataset_name} data from {granularity} {min_unit} to {max_unit}...")
    
    # Load interaction data
    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1)
    )
    
    print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
    
    # Use provided start_timestamp or compute from data
    if start_timestamp is None:
        start_timestamp = df['timestamp'].min()
    
    # Compute statistics
    temporal_stats = compute_temporal_statistics(df, granularity, start_timestamp)
    
    # Compute per-window statistics
    window_stats = []
    for w_info in all_windows:
        train_start = w_info.get('train_start_unit', w_info.get('start_unit'))
        train_end = train_start + w_info.get('train_units', 0)
        test_start = w_info.get('test_start_unit', train_end)
        test_end = w_info.get('end_unit')
        
        train_df = df[df['time_unit'].between(train_start, train_end - 1)]
        test_df = df[df['time_unit'].between(test_start, test_end)]
        
        window_stats.append({
            'window_number': w_info.get('window_number'),
            'train_interactions': len(train_df),
            'test_interactions': len(test_df),
            'train_users': train_df['user_id'].nunique(),
            'test_users': test_df['user_id'].nunique(),
            'train_items': train_df['item_id'].nunique(),
            'test_items': test_df['item_id'].nunique(),
        })
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path('plot/output')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = []
    
    # 1. Interactions over time
    # Don't pass ax parameter so the function can create its own layout with window tracks
    plot_interactions_over_time(df, granularity, start_timestamp, all_windows)
    start_dt = datetime.fromtimestamp(start_timestamp)
    plt.suptitle(f"{dataset_name} - Interactions Over Time", fontsize=12, fontweight='bold')
    output_path1 = output_dir / f"{experiment_id}_interactions_timeline.pdf"
    plt.savefig(output_path1, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    output_paths.append(output_path1)
    print(f"✓ Saved: {output_path1}")
    
    # 2. Time of day/week pattern
    fig2, ax2 = plt.subplots(figsize=figsize)
    if granularity == 'hour':
        plot_time_of_day_pattern(df, start_timestamp, ax=ax2)
        fig2.suptitle(f"{dataset_name} - Hourly Interaction Pattern", fontsize=12, fontweight='bold')
    else:
        # For daily granularity, show day of week pattern
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['day_of_week'] = df['datetime'].dt.dayofweek
        dow_counts = df.groupby('day_of_week').size()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(range(7), [dow_counts.get(i, 0) for i in range(7)], 
               color='#2E86AB', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Interactions', fontsize=11, fontweight='bold')
        ax2.set_title('Interaction Pattern by Day of Week', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        fig2.suptitle(f"{dataset_name} - Day of Week Pattern", fontsize=12, fontweight='bold')
    output_path2 = output_dir / f"{experiment_id}_time_pattern.pdf"
    plt.savefig(output_path2, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    output_paths.append(output_path2)
    print(f"✓ Saved: {output_path2}")
    
    # 3. User/Item distributions
    fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
    plot_user_item_distributions(df, ax=(ax3_left, ax3_right))
    fig3.suptitle(f"{dataset_name} - User Activity & Item Popularity", fontsize=12, fontweight='bold')
    output_path3 = output_dir / f"{experiment_id}_distributions.pdf"
    plt.savefig(output_path3, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    output_paths.append(output_path3)
    print(f"✓ Saved: {output_path3}")
    
    # Print statistics summary
    print(f"\n{'='*70}")
    print(f"Dataset Analysis Summary - {experiment_id}")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Granularity: {granularity}")
    print(f"Time range: {min_unit} to {max_unit} ({max_unit - min_unit + 1} {granularity}s)")
    print(f"Start datetime: {temporal_stats['start_datetime']}")
    print(f"\nInteractions:")
    print(f"  Total: {temporal_stats['total_interactions']:,}")
    print(f"  Positive: {temporal_stats['positive_interactions']:,}")
    print(f"  Negative: {temporal_stats['negative_interactions']:,}")
    print(f"\nUsers: {temporal_stats['unique_users']:,}")
    print(f"  Avg interactions/user: {temporal_stats['user_activity']['mean']:.1f}")
    print(f"  Median interactions/user: {temporal_stats['user_activity']['median']:.1f}")
    print(f"  Range: {temporal_stats['user_activity']['min']} - {temporal_stats['user_activity']['max']}")
    print(f"\nItems: {temporal_stats['unique_items']:,}")
    print(f"  Avg interactions/item: {temporal_stats['item_popularity']['mean']:.1f}")
    print(f"  Median interactions/item: {temporal_stats['item_popularity']['median']:.1f}")
    print(f"  Range: {temporal_stats['item_popularity']['min']} - {temporal_stats['item_popularity']['max']}")
    print(f"\nInteractions per {granularity}:")
    print(f"  Mean: {temporal_stats['interactions_per_unit']['mean']:.1f}")
    print(f"  Std: {temporal_stats['interactions_per_unit']['std']:.1f}")
    print(f"  Range: {temporal_stats['interactions_per_unit']['min']} - {temporal_stats['interactions_per_unit']['max']}")
    print(f"\nWindows analyzed: {len(all_windows)}")
    print(f"{'='*70}\n")
    
    print(f"\n✓ Generated {len(output_paths)} analysis plots in: {output_dir}\n")
    
    return {
        'statistics': temporal_stats,
        'window_stats': window_stats,
        'dataframe': df,
        'output_paths': [str(p) for p in output_paths]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize dataset temporal properties from experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset from temporal experiment
  python -m plot.dataset_analysis --experiment-id exp_tfidf_temporal_36h
  
  # Specify custom start timestamp (Unix timestamp)
  python -m plot.dataset_analysis --experiment-id exp001 \\
      --start-timestamp 1573463158
  
  # Custom output directory
  python -m plot.dataset_analysis --experiment-id exp001 \\
      --output-dir my_plots/
        """
    )
    
    parser.add_argument('--experiment-id', required=True,
                       help='Experiment ID to analyze (must be temporal experiment)')
    parser.add_argument('--jsonl-path', default='output/results/experiments.jsonl',
                       help='Path to experiments.jsonl file')
    parser.add_argument('--dataset-path', default='datasets/atomic_files',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory for PDFs (default: plot/output/)')
    parser.add_argument('--start-timestamp', type=float,
                       help='Start timestamp (Unix timestamp). If not provided, uses first interaction.')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 6],
                       help='Figure size (width height) for each plot')
    
    args = parser.parse_args()
    
    try:
        analyze_dataset_from_experiment(
            experiment_id=args.experiment_id,
            jsonl_path=args.jsonl_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            start_timestamp=args.start_timestamp,
            figsize=tuple(args.figsize)
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
