"""
Plotting functions for window validation and statistics.

This module provides visualization functions for sliding window methodology
validation, including data distribution plots and cold-start analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def plot_window_data_distribution(stats_df: pd.DataFrame,
                                  ax: Optional[plt.Axes] = None,
                                  title: str = 'Data Distribution Across Windows') -> plt.Axes:
    """
    Plot data distribution (interactions and items) across windows.
    
    Args:
        stats_df: DataFrame with window statistics (from compute_all_window_statistics)
        ax: Optional matplotlib axes. If None, creates new figure
        title: Plot title
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    windows = stats_df['window_number'].values
    x = np.arange(len(windows))
    width = 0.35
    
    # Plot interactions
    ax2 = ax.twinx()
    
    train_bars = ax.bar(x - width/2, stats_df['train_interactions'], width,
                       label='Train Interactions', color='#2E86AB', alpha=0.8, edgecolor='black')
    test_bars = ax.bar(x + width/2, stats_df['test_interactions'], width,
                      label='Test Interactions', color='#A23B72', alpha=0.8, edgecolor='black')
    
    # Plot items on secondary axis
    train_items_line = ax2.plot(x, stats_df['train_items'], 'o-', 
                               label='Train Items', color='#F18F01', linewidth=2.5, 
                               markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    test_items_line = ax2.plot(x, stats_df['test_items'], 's-', 
                              label='Test Items', color='#C73E1D', linewidth=2.5,
                              markersize=8, markeredgecolor='white', markeredgewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Interactions', fontsize=12, fontweight='bold', color="#000000")
    ax2.set_ylabel('Number of Items', fontsize=12, fontweight='bold', color="#000000")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels([int(w) for w in windows])
    ax.tick_params(axis='y', labelcolor="#000000")
    ax2.tick_params(axis='y', labelcolor="#000000")
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Combined legend
    bars_legend = [train_bars, test_bars]
    lines_legend = [train_items_line[0], test_items_line[0]]
    labels = ['Train Interactions', 'Test Interactions', 'Train Items', 'Test Items']
    
    ax.legend(bars_legend + lines_legend, labels, 
             loc='upper left', framealpha=0.95, fontsize=10)
    
    return ax


def plot_cold_start_ratios(stats_df: pd.DataFrame,
                          ax: Optional[plt.Axes] = None,
                          title: str = 'Cold-Start Ratios Across Windows') -> plt.Axes:
    """
    Plot cold-start ratios (new users/items) across windows.
    
    Args:
        stats_df: DataFrame with window statistics
        ax: Optional matplotlib axes. If None, creates new figure
        title: Plot title
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    windows = stats_df['window_number'].values
    x = np.arange(len(windows))
    width = 0.35
    
    # Convert to percentages
    user_cs_pct = stats_df['cold_start_user_ratio'] * 100
    item_cs_pct = stats_df['cold_start_item_ratio'] * 100
    
    # Plot bars
    user_bars = ax.bar(x - width/2, user_cs_pct, width,
                      label='New Users %', color='#6A4C93', alpha=0.8, edgecolor='black')
    item_bars = ax.bar(x + width/2, item_cs_pct, width,
                      label='New Items %', color='#1982C4', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bars in [user_bars, item_bars]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:  # Only label if > 1%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    # Add mean lines
    mean_user = user_cs_pct.mean()
    mean_item = item_cs_pct.mean()
    ax.axhline(mean_user, color='#6A4C93', linestyle='--', linewidth=1.5, alpha=0.6,
              label=f'Mean New Users: {mean_user:.1f}%')
    ax.axhline(mean_item, color='#1982C4', linestyle='--', linewidth=1.5, alpha=0.6,
              label=f'Mean New Items: {mean_item:.1f}%')
    
    # Formatting
    ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cold-Start Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([int(w) for w in windows])
    ax.set_ylim(0, max(max(user_cs_pct), max(item_cs_pct)) * 1.15)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    return ax


def plot_window_overlap_analysis(stats_df: pd.DataFrame,
                                 ax: Optional[plt.Axes] = None,
                                 title: str = 'Train-Test Overlap Analysis') -> plt.Axes:
    """
    Plot overlap ratios between train and test sets.
    
    Args:
        stats_df: DataFrame with window statistics
        ax: Optional matplotlib axes. If None, creates new figure
        title: Plot title
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    windows = stats_df['window_number'].values
    x = np.arange(len(windows))
    
    # Calculate overlap percentages (how many test entities were seen in train)
    test_user_seen_pct = (1 - stats_df['cold_start_user_ratio']) * 100
    test_item_seen_pct = (1 - stats_df['cold_start_item_ratio']) * 100
    
    # Plot lines
    ax.plot(x, test_user_seen_pct, 'o-', label='Users (Seen in Train)', 
           color='#2E86AB', linewidth=2.5, markersize=8, 
           markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(x, test_item_seen_pct, 's-', label='Items (Seen in Train)', 
           color='#F18F01', linewidth=2.5, markersize=8,
           markeredgecolor='white', markeredgewidth=1.5)
    
    # Fill between to show the gap
    ax.fill_between(x, test_user_seen_pct, test_item_seen_pct, 
                    alpha=0.2, color='gray', label='User-Item Gap')
    
    # Add mean lines
    mean_user = test_user_seen_pct.mean()
    mean_item = test_item_seen_pct.mean()
    ax.axhline(mean_user, color='#2E86AB', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.axhline(mean_item, color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.4)
    
    # Formatting
    ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Seen in Training (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([int(w) for w in windows])
    ax.set_ylim(0, 105)
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    
    # Add annotation
    if mean_item < 80:
        ax.text(0.02, 0.98, 
               f"⚠ High cold-start:\n{100-mean_item:.1f}% new items on avg",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return ax


def plot_comprehensive_window_validation(stats_df: pd.DataFrame,
                                        experiment_id: str,
                                        dataset_name: str,
                                        figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create a comprehensive figure with all window validation plots.
    
    Args:
        stats_df: DataFrame with window statistics
        experiment_id: Experiment identifier
        dataset_name: Dataset name
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)
    
    # Main title
    fig.suptitle(f'Sliding Window Validation: {experiment_id}\nDataset: {dataset_name}',
                fontsize=14, fontweight='bold', y=0.995)
    
    # 1. Data distribution
    ax1 = fig.add_subplot(gs[0, 0])
    plot_window_data_distribution(stats_df, ax=ax1, 
                                  title='(a) Data Distribution Across Windows')
    
    # 2. Cold-start ratios
    ax2 = fig.add_subplot(gs[1, 0])
    plot_cold_start_ratios(stats_df, ax=ax2,
                          title='(b) Cold-Start Ratios: New Users and Items in Test Set')
    
    # 3. Overlap analysis
    ax3 = fig.add_subplot(gs[2, 0])
    plot_window_overlap_analysis(stats_df, ax=ax3,
                                 title='(c) Train-Test Overlap: Percentage of Test Entities Seen During Training')
    
    return fig


def plot_interaction_volume_stability(stats_df: pd.DataFrame,
                                     ax: Optional[plt.Axes] = None,
                                     title: str = 'Interaction Volume Stability') -> plt.Axes:
    """
    Plot coefficient of variation to assess data volume stability.
    
    Args:
        stats_df: DataFrame with window statistics
        ax: Optional matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate rolling CV
    window_size = 3
    if len(stats_df) >= window_size:
        train_rolling_cv = stats_df['train_interactions'].rolling(window_size).std() / \
                          stats_df['train_interactions'].rolling(window_size).mean()
        test_rolling_cv = stats_df['test_interactions'].rolling(window_size).std() / \
                         stats_df['test_interactions'].rolling(window_size).mean()
    else:
        train_rolling_cv = pd.Series([np.nan] * len(stats_df))
        test_rolling_cv = pd.Series([np.nan] * len(stats_df))
    
    windows = stats_df['window_number'].values
    x = np.arange(len(windows))
    
    # Plot
    ax.plot(x, train_rolling_cv, 'o-', label=f'Train (rolling CV, w={window_size})', 
           color='#2E86AB', linewidth=2, markersize=7)
    ax.plot(x, test_rolling_cv, 's-', label=f'Test (rolling CV, w={window_size})', 
           color='#A23B72', linewidth=2, markersize=7)
    
    # Add stability zones
    ax.axhspan(0, 0.1, alpha=0.1, color='green', label='Stable (CV < 0.1)')
    ax.axhspan(0.1, 0.3, alpha=0.1, color='yellow', label='Moderate (CV 0.1-0.3)')
    ax.axhspan(0.3, 1.0, alpha=0.1, color='red', label='High Variation (CV > 0.3)')
    
    # Formatting
    ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([int(w) for w in windows])
    ax.set_ylim(0, min(1.0, max(train_rolling_cv.max(), test_rolling_cv.max()) * 1.2))
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    
    return ax
