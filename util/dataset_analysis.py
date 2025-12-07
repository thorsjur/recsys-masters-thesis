"""
Dataset temporal analysis and visualization.

This module provides functions to analyze and visualize temporal properties
of datasets used in experiments, including interaction patterns over time,
user/item distributions, and data characteristics across windows.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from util.experiment_data import load_experiment_results
from util.statistics import basic_stats


def load_temporal_interaction_data(dataset_path: str, 
                                   dataset_name: str,
                                   granularity: str,
                                   time_units: range) -> pd.DataFrame:
    """
    Load interaction data from temporal split files.
    
    Args:
        dataset_path: Path to dataset directory
        dataset_name: Name of dataset
        granularity: 'hour' or 'day'
        time_units: Range of time units to load
        
    Returns:
        DataFrame with columns: user_id, item_id, label, timestamp, time_unit
    """
    dataset_dir = Path(dataset_path) / dataset_name
    all_data = []
    
    for unit in time_units:
        if granularity == 'hour':
            file_path = dataset_dir / f"{dataset_name}.hour_{unit}.inter"
        else:
            file_path = dataset_dir / f"{dataset_name}.day_{unit}.inter"
        
        if not file_path.exists():
            continue
        
        # Read the file, skipping header
        df = pd.read_csv(file_path, sep='\t', skiprows=1, 
                        names=['user_id', 'item_id', 'label', 'timestamp', 'impression_id'])
        df['time_unit'] = unit
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No data files found for {dataset_name} in range {time_units}")
    
    return pd.concat(all_data, ignore_index=True)


def compute_temporal_statistics(df: pd.DataFrame, 
                                granularity: str,
                                start_timestamp: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute temporal statistics from interaction data.
    
    Args:
        df: DataFrame with interaction data
        granularity: 'hour' or 'day'
        start_timestamp: Optional start timestamp for absolute time calculation
        
    Returns:
        Dictionary with temporal statistics
    """
    if start_timestamp is None:
        start_timestamp = df['timestamp'].min()
    
    stats = {
        'total_interactions': len(df),
        'positive_interactions': int((df['label'] == 1).sum()),
        'negative_interactions': int((df['label'] == 0).sum()),
        'unique_users': df['user_id'].nunique(),
        'unique_items': df['item_id'].nunique(),
        'time_span_seconds': float(df['timestamp'].max() - df['timestamp'].min()),
        'first_timestamp': float(df['timestamp'].min()),
        'last_timestamp': float(df['timestamp'].max()),
        'start_datetime': datetime.fromtimestamp(start_timestamp).isoformat(),
        'granularity': granularity,
    }
    
    # Interactions per time unit
    interactions_per_unit = df.groupby('time_unit').size()
    stats['interactions_per_unit'] = {
        'mean': float(interactions_per_unit.mean()),
        'std': float(interactions_per_unit.std()),
        'min': int(interactions_per_unit.min()),
        'max': int(interactions_per_unit.max()),
    }
    
    # User activity
    user_interactions = df.groupby('user_id').size()
    stats['user_activity'] = {
        'mean': float(user_interactions.mean()),
        'std': float(user_interactions.std()),
        'median': float(user_interactions.median()),
        'min': int(user_interactions.min()),
        'max': int(user_interactions.max()),
    }
    
    # Item popularity
    item_interactions = df.groupby('item_id').size()
    stats['item_popularity'] = {
        'mean': float(item_interactions.mean()),
        'std': float(item_interactions.std()),
        'median': float(item_interactions.median()),
        'min': int(item_interactions.min()),
        'max': int(item_interactions.max()),
    }
    
    return stats


def plot_interactions_over_time(df: pd.DataFrame,
                                granularity: str,
                                start_timestamp: float,
                                window_info: Optional[List[Dict]] = None,
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot interactions over time with time-of-day/week patterns.
    
    Args:
        df: DataFrame with interaction data
        granularity: 'hour' or 'day'
        start_timestamp: Timestamp of first interaction
        window_info: Optional list of window configurations to overlay
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert timestamps to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    start_datetime = pd.to_datetime(start_timestamp, unit='s')
    
    # Aggregate by time unit
    if granularity == 'hour':
        # Hourly aggregation
        df['hour_mark'] = start_datetime + pd.to_timedelta(df['time_unit'], unit='h')
        agg_data = df.groupby('hour_mark').size().reset_index(name='count')
        xlabel = 'Time (Hourly)'
    else:
        # Daily aggregation
        df['day_mark'] = start_datetime + pd.to_timedelta(df['time_unit'], unit='D')
        agg_data = df.groupby('day_mark').size().reset_index(name='count')
        xlabel = 'Time (Daily)'
    
    time_col = 'hour_mark' if granularity == 'hour' else 'day_mark'
    
    # Plot interactions
    ax.plot(agg_data[time_col], agg_data['count'], 
           linewidth=2, color='#2E86AB', marker='o', markersize=4)
    
    # Overlay window boundaries if provided
    if window_info:
        colors = ['#F18F01', '#A23B72', '#06A77D', '#9B59B6']
        for i, window in enumerate(window_info):
            train_start = window.get('train_start_unit', window.get('start_unit'))
            train_end = window.get('train_end_unit', train_start + window.get('train_units', 0))
            test_start = window.get('test_start_unit', train_end)
            test_end = window.get('end_unit')
            
            if granularity == 'hour':
                train_start_dt = start_datetime + timedelta(hours=train_start)
                train_end_dt = start_datetime + timedelta(hours=train_end)
                test_start_dt = start_datetime + timedelta(hours=test_start)
                test_end_dt = start_datetime + timedelta(hours=test_end)
            else:
                train_start_dt = start_datetime + timedelta(days=train_start)
                train_end_dt = start_datetime + timedelta(days=train_end)
                test_start_dt = start_datetime + timedelta(days=test_start)
                test_end_dt = start_datetime + timedelta(days=test_end)
            
            color = colors[i % len(colors)]
            
            # Train region
            ax.axvspan(train_start_dt, train_end_dt, alpha=0.15, color=color, 
                      label=f'W{window["window_number"]} Train' if i < 3 else None)
            # Test region
            ax.axvspan(test_start_dt, test_end_dt, alpha=0.3, color=color,
                      label=f'W{window["window_number"]} Test' if i < 3 else None)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Interactions', fontsize=12, fontweight='bold')
    ax.set_title('Interaction Volume Over Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis dates
    if granularity == 'hour':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    if window_info and len(window_info) <= 5:
        ax.legend(loc='best', fontsize=9)
    
    return ax


def plot_time_of_day_pattern(df: pd.DataFrame,
                             start_timestamp: float,
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot interaction patterns by hour of day.
    
    Args:
        df: DataFrame with interaction data
        start_timestamp: Timestamp of first interaction
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to datetime and extract hour of day
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour_of_day'] = df['datetime'].dt.hour
    
    # Aggregate by hour
    hourly_counts = df.groupby('hour_of_day').size()
    
    # Create bar plot
    hours = range(24)
    counts = [hourly_counts.get(h, 0) for h in hours]
    
    colors = ['#2E86AB' if 6 <= h < 22 else '#A23B72' for h in hours]
    ax.bar(hours, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Interactions', fontsize=12, fontweight='bold')
    ax.set_title('Interaction Pattern by Hour of Day', fontsize=13, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add day/night labels
    ax.axvspan(-0.5, 6, alpha=0.1, color='gray', label='Night')
    ax.axvspan(22, 24, alpha=0.1, color='gray')
    ax.axvspan(6, 22, alpha=0.05, color='yellow', label='Day')
    ax.legend(loc='upper right', fontsize=9)
    
    return ax


def plot_user_item_distributions(df: pd.DataFrame,
                                 ax: Optional[Tuple[plt.Axes, plt.Axes]] = None) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot user activity and item popularity distributions.
    
    Args:
        df: DataFrame with interaction data
        ax: Optional tuple of (user_ax, item_ax)
        
    Returns:
        Tuple of matplotlib axes objects
    """
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        ax1, ax2 = ax
    
    # User activity distribution
    user_interactions = df.groupby('user_id').size()
    ax1.hist(user_interactions, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Interactions per User', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
    ax1.set_title('User Activity Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add statistics text
    stats_text = f"Mean: {user_interactions.mean():.1f}\nMedian: {user_interactions.median():.1f}\nStd: {user_interactions.std():.1f}"
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    # Item popularity distribution
    item_interactions = df.groupby('item_id').size()
    ax2.hist(item_interactions, bins=50, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Interactions per Item', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Items', fontsize=11, fontweight='bold')
    ax2.set_title('Item Popularity Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add statistics text
    stats_text = f"Mean: {item_interactions.mean():.1f}\nMedian: {item_interactions.median():.1f}\nStd: {item_interactions.std():.1f}"
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)
    
    return ax1, ax2


def plot_window_statistics(window_stats: List[Dict[str, Any]],
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot statistics across temporal windows.
    
    Args:
        window_stats: List of statistics for each window
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    windows = [w['window_number'] for w in window_stats]
    train_interactions = [w['train_interactions'] for w in window_stats]
    test_interactions = [w['test_interactions'] for w in window_stats]
    
    x = np.arange(len(windows))
    width = 0.35
    
    ax.bar(x - width/2, train_interactions, width, label='Train', color='#2E86AB', alpha=0.7)
    ax.bar(x + width/2, test_interactions, width, label='Test', color='#F18F01', alpha=0.7)
    
    ax.set_xlabel('Window Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Interactions', fontsize=11, fontweight='bold')
    ax.set_title('Train/Test Interactions per Window', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'W{w}' for w in windows])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    return ax


def plot_interaction_heatmap(df: pd.DataFrame,
                             granularity: str,
                             start_timestamp: float,
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot heatmap of interactions across time units and hours/days.
    
    Args:
        df: DataFrame with interaction data
        granularity: 'hour' or 'day'
        start_timestamp: Timestamp of first interaction
        ax: Optional matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    import matplotlib.colors as mcolors
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    if granularity == 'hour':
        # Hour of day vs time unit
        df['hour_of_day'] = df['datetime'].dt.hour
        pivot = df.groupby(['time_unit', 'hour_of_day']).size().unstack(fill_value=0)
        
        # Normalize by row for better visualization
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100
        
        im = ax.imshow(pivot_norm.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Time Unit (Hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_title('Interaction Distribution: Hour of Day vs Time Unit (%)', fontsize=12, fontweight='bold')
        
        # Show every Nth hour on x-axis
        step = max(1, len(pivot) // 20)
        ax.set_xticks(range(0, len(pivot), step))
        ax.set_xticklabels([str(i) for i in pivot.index[::step]], fontsize=8)
        ax.set_yticks(range(24))
        ax.set_yticklabels(range(24), fontsize=8)
        
    else:
        # Day of week vs time unit
        df['day_of_week'] = df['datetime'].dt.dayofweek
        pivot = df.groupby(['time_unit', 'day_of_week']).size().unstack(fill_value=0)
        
        # Normalize by row
        pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100
        
        im = ax.imshow(pivot_norm.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Time Unit (Days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Day of Week', fontsize=11, fontweight='bold')
        ax.set_title('Interaction Distribution: Day of Week vs Time Unit (%)', fontsize=12, fontweight='bold')
        
        step = max(1, len(pivot) // 15)
        ax.set_xticks(range(0, len(pivot), step))
        ax.set_xticklabels([str(i) for i in pivot.index[::step]], fontsize=8)
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Percentage of Interactions', rotation=270, labelpad=20, fontsize=10)
    
    return ax
