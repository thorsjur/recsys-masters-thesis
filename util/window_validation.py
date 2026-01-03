"""
Sliding window validation and statistics.

This module provides functions to validate that the sliding window methodology
functioned as intended and to compute statistics about data distribution,
cold-start ratios, and temporal drift across windows.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


def compute_window_statistics(df: pd.DataFrame, 
                              window_info: Dict[str, Any],
                              granularity: str = 'hour') -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a single temporal window.
    
    Args:
        df: DataFrame with columns ['user_id', 'item_id', 'timestamp', 'time_unit', 'label']
        window_info: Dictionary with window configuration (start_unit, end_unit, etc.)
        granularity: Time granularity ('hour' or 'day')
        
    Returns:
        Dictionary with window statistics including:
        - train/test interaction counts
        - train/test user/item counts
        - cold-start ratios (new users/items in test)
        - overlap statistics
    """
    # Extract window boundaries
    train_start = window_info.get('train_start_unit', window_info.get('start_unit'))
    train_units = window_info.get('train_units', 0)
    train_end = train_start + train_units
    test_start = window_info.get('test_start_unit', train_end)
    test_end = window_info.get('end_unit')
    
    # Split into train and test
    train_df = df[df['time_unit'].between(train_start, train_end - 1)]
    test_df = df[df['time_unit'].between(test_start, test_end)]
    
    # Get unique users and items
    train_users = set(train_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    train_items = set(train_df['item_id'].unique())
    test_items = set(test_df['item_id'].unique())
    
    # Compute cold-start ratios
    new_users_in_test = test_users - train_users
    new_items_in_test = test_items - train_items
    
    cold_start_user_ratio = len(new_users_in_test) / len(test_users) if len(test_users) > 0 else 0
    cold_start_item_ratio = len(new_items_in_test) / len(test_items) if len(test_items) > 0 else 0
    
    # Compute overlap
    user_overlap = len(train_users & test_users)
    item_overlap = len(train_items & test_items)
    
    # Compute interaction statistics
    train_positive = len(train_df[train_df['label'] == 1]) if 'label' in train_df.columns else len(train_df)
    test_positive = len(test_df[test_df['label'] == 1]) if 'label' in test_df.columns else len(test_df)
    
    return {
        'window_number': window_info.get('window_number'),
        'train_start_unit': train_start,
        'train_end_unit': train_end - 1,
        'test_start_unit': test_start,
        'test_end_unit': test_end,
        'train_interactions': len(train_df),
        'test_interactions': len(test_df),
        'train_positive': train_positive,
        'test_positive': test_positive,
        'train_users': len(train_users),
        'test_users': len(test_users),
        'train_items': len(train_items),
        'test_items': len(test_items),
        'new_users_in_test': len(new_users_in_test),
        'new_items_in_test': len(new_items_in_test),
        'cold_start_user_ratio': cold_start_user_ratio,
        'cold_start_item_ratio': cold_start_item_ratio,
        'user_overlap': user_overlap,
        'item_overlap': item_overlap,
        'user_overlap_ratio': user_overlap / len(train_users) if len(train_users) > 0 else 0,
        'item_overlap_ratio': item_overlap / len(train_items) if len(train_items) > 0 else 0,
    }


def compute_all_window_statistics(df: pd.DataFrame,
                                  windows: List[Dict[str, Any]],
                                  granularity: str = 'hour') -> pd.DataFrame:
    """
    Compute statistics for all windows in a temporal experiment.
    
    Args:
        df: DataFrame with temporal interaction data
        windows: List of window configuration dictionaries (should be deduplicated by window_number)
        granularity: Time granularity ('hour' or 'day')
        
    Returns:
        DataFrame with statistics for each window (one row per window)
        
    Note:
        If multiple runs exist for the same window, pass only unique window configurations
        (deduplicated by window_number) to avoid computing statistics multiple times.
    """
    # Deduplicate windows by window_number if not already done
    unique_windows = {}
    for window_info in windows:
        win_num = window_info.get('window_number')
        if win_num is not None and win_num not in unique_windows:
            unique_windows[win_num] = window_info
    
    all_stats = []
    
    # Process in sorted order
    for win_num in sorted(unique_windows.keys()):
        window_info = unique_windows[win_num]
        stats = compute_window_statistics(df, window_info, granularity)
        all_stats.append(stats)
    
    # Convert to DataFrame for easy analysis
    stats_df = pd.DataFrame(all_stats)
    
    # Sort by window number (should already be sorted, but ensure it)
    if 'window_number' in stats_df.columns:
        stats_df = stats_df.sort_values('window_number').reset_index(drop=True)
    
    return stats_df


def generate_validation_report(stats_df: pd.DataFrame,
                               experiment_id: str,
                               dataset_name: str,
                               granularity: str = 'hour') -> str:
    """
    Generate a text report validating the sliding window methodology.
    
    Args:
        stats_df: DataFrame with window statistics (from compute_all_window_statistics)
        experiment_id: Experiment identifier
        dataset_name: Dataset name
        granularity: Time granularity
        
    Returns:
        Formatted text report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"Sliding Window Validation Report")
    report_lines.append(f"Experiment: {experiment_id}")
    report_lines.append(f"Dataset: {dataset_name}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overview
    num_windows = len(stats_df)
    report_lines.append(f"Total Windows: {num_windows}")
    report_lines.append(f"Granularity: {granularity}")
    report_lines.append("")
    
    # Data distribution summary
    report_lines.append("-" * 80)
    report_lines.append("DATA DISTRIBUTION ACROSS WINDOWS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Table header
    header = f"{'Win':>3} | {'Train':>8} | {'Test':>8} | {'Train':>7} | {'Test':>7} | {'Train':>7} | {'Test':>7}"
    subheader = f"{'#':>3} | {'Interact':>8} | {'Interact':>8} | {'Users':>7} | {'Users':>7} | {'Items':>7} | {'Items':>7}"
    report_lines.append(header)
    report_lines.append(subheader)
    report_lines.append("-" * 80)
    
    for _, row in stats_df.iterrows():
        line = (f"{int(row['window_number']):3d} | "
                f"{int(row['train_interactions']):8,d} | "
                f"{int(row['test_interactions']):8,d} | "
                f"{int(row['train_users']):7,d} | "
                f"{int(row['test_users']):7,d} | "
                f"{int(row['train_items']):7,d} | "
                f"{int(row['test_items']):7,d}")
        report_lines.append(line)
    
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS:")
    report_lines.append(f"  Train Interactions - Mean: {stats_df['train_interactions'].mean():,.0f}, "
                       f"Std: {stats_df['train_interactions'].std():,.0f}, "
                       f"Range: {stats_df['train_interactions'].min():,.0f}-{stats_df['train_interactions'].max():,.0f}")
    report_lines.append(f"  Test Interactions  - Mean: {stats_df['test_interactions'].mean():,.0f}, "
                       f"Std: {stats_df['test_interactions'].std():,.0f}, "
                       f"Range: {stats_df['test_interactions'].min():,.0f}-{stats_df['test_interactions'].max():,.0f}")
    report_lines.append(f"  Train Items        - Mean: {stats_df['train_items'].mean():,.0f}, "
                       f"Std: {stats_df['train_items'].std():,.0f}, "
                       f"Range: {stats_df['train_items'].min():,.0f}-{stats_df['train_items'].max():,.0f}")
    report_lines.append(f"  Test Items         - Mean: {stats_df['test_items'].mean():,.0f}, "
                       f"Std: {stats_df['test_items'].std():,.0f}, "
                       f"Range: {stats_df['test_items'].min():,.0f}-{stats_df['test_items'].max():,.0f}")
    report_lines.append("")
    
    # Cold-start analysis
    report_lines.append("-" * 80)
    report_lines.append("COLD-START ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append(f"{'Win':>3} | {'New Users':>10} | {'New Items':>10} | {'User CS%':>9} | {'Item CS%':>9}")
    report_lines.append(f"{'#':>3} | {'in Test':>10} | {'in Test':>10} | {'Ratio':>9} | {'Ratio':>9}")
    report_lines.append("-" * 80)
    
    for _, row in stats_df.iterrows():
        line = (f"{int(row['window_number']):3d} | "
                f"{int(row['new_users_in_test']):10,d} | "
                f"{int(row['new_items_in_test']):10,d} | "
                f"{row['cold_start_user_ratio']*100:8.2f}% | "
                f"{row['cold_start_item_ratio']*100:8.2f}%")
        report_lines.append(line)
    
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Cold-start summary
    report_lines.append("COLD-START SUMMARY:")
    report_lines.append(f"  New Users (mean):   {stats_df['cold_start_user_ratio'].mean()*100:.2f}% "
                       f"(std: {stats_df['cold_start_user_ratio'].std()*100:.2f}%)")
    report_lines.append(f"  New Items (mean):   {stats_df['cold_start_item_ratio'].mean()*100:.2f}% "
                       f"(std: {stats_df['cold_start_item_ratio'].std()*100:.2f}%)")
    report_lines.append(f"  Max New Items:      {stats_df['cold_start_item_ratio'].max()*100:.2f}% "
                       f"(Window {int(stats_df.loc[stats_df['cold_start_item_ratio'].idxmax(), 'window_number'])})")
    report_lines.append(f"  Min New Items:      {stats_df['cold_start_item_ratio'].min()*100:.2f}% "
                       f"(Window {int(stats_df.loc[stats_df['cold_start_item_ratio'].idxmin(), 'window_number'])})")
    report_lines.append("")
    
    # Interpretation
    report_lines.append("-" * 80)
    report_lines.append("INTERPRETATION:")
    report_lines.append("-" * 80)
    
    avg_cold_start = stats_df['cold_start_item_ratio'].mean()
    if avg_cold_start > 0.3:
        report_lines.append(f"⚠ HIGH cold-start ratio ({avg_cold_start*100:.1f}% new items on average).")
        report_lines.append("  This is expected in news recommendation where items are constantly refreshed.")
        report_lines.append("  Models relying on item history may struggle in these conditions.")
    elif avg_cold_start > 0.1:
        report_lines.append(f"MODERATE cold-start ratio ({avg_cold_start*100:.1f}% new items on average).")
        report_lines.append("  Content-based models should have an advantage over collaborative methods.")
    else:
        report_lines.append(f"LOW cold-start ratio ({avg_cold_start*100:.1f}% new items on average).")
        report_lines.append("  Both collaborative and content-based models should perform well.")
    
    report_lines.append("")
    
    # Stability assessment
    train_cv = stats_df['train_interactions'].std() / stats_df['train_interactions'].mean()
    test_cv = stats_df['test_interactions'].std() / stats_df['test_interactions'].mean()
    
    report_lines.append(f"Data volume stability (coefficient of variation):")
    report_lines.append(f"  Train: {train_cv:.3f}")
    report_lines.append(f"  Test:  {test_cv:.3f}")
    
    if train_cv < 0.1:
        report_lines.append("  Training data volume is STABLE across windows.")
    elif train_cv < 0.3:
        report_lines.append("  ~ Training data volume has MODERATE variation across windows.")
    else:
        report_lines.append("  ⚠ Training data volume has HIGH variation across windows.")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def export_statistics_table(stats_df: pd.DataFrame,
                           output_path: str,
                           format: str = 'latex') -> str:
    """
    Export window statistics table in publication-ready format.
    
    Args:
        stats_df: DataFrame with window statistics
        output_path: Path to save the table
        format: Output format ('latex', 'csv', or 'markdown')
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select and rename columns for publication
    table_df = stats_df[[
        'window_number',
        'train_interactions',
        'test_interactions', 
        'train_items',
        'test_items',
        'new_items_in_test',
        'cold_start_item_ratio'
    ]].copy()
    
    table_df.columns = [
        'Window',
        'Train Interactions',
        'Test Interactions',
        'Train Items',
        'Test Items',
        'New Items',
        'Cold-Start %'
    ]
    
    # Convert percentages
    table_df['Cold-Start %'] = (table_df['Cold-Start %'] * 100).round(2)
    
    if format == 'latex':
        latex_str = table_df.to_latex(
            index=False,
            float_format='%.2f',
            caption='Data distribution and cold-start statistics across temporal windows.',
            label='tab:window_statistics',
            column_format='c' + 'r' * (len(table_df.columns) - 1)
        )
        
        with open(output_path, 'w') as f:
            f.write(latex_str)
            
    elif format == 'csv':
        table_df.to_csv(output_path, index=False)
        
    elif format == 'markdown':
        markdown_str = table_df.to_markdown(index=False, floatfmt='.2f')
        
        with open(output_path, 'w') as f:
            f.write(markdown_str)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'latex', 'csv', or 'markdown'.")
    
    return str(output_path)
