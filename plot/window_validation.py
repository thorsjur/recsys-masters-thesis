"""
Sliding window validation and statistics visualization.

This script generates comprehensive validation reports and plots to confirm
that the sliding window methodology functioned as intended. It analyzes:
- Data distribution across windows
- Cold-start ratios (new items/users in test)
- Train-test overlap statistics
- Volume stability assessment

Usage:
    python -m plot.window_validation --experiment-id exp001
"""

import argparse
from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.experiment_data import load_experiment_results
from util.dataset_analysis import load_temporal_interaction_data
from util.window_validation import (
    compute_all_window_statistics,
    generate_validation_report,
    export_statistics_table
)
from util.window_plots import (
    plot_comprehensive_window_validation,
    plot_window_data_distribution,
    plot_cold_start_ratios,
    plot_window_overlap_analysis,
    plot_interaction_volume_stability
)


def validate_sliding_windows(experiment_id: str,
                            jsonl_path: str = 'output/results/experiments.jsonl',
                            dataset_path: str = 'datasets/atomic_files',
                            output_dir: Optional[str] = None,
                            generate_plots: bool = True,
                            export_tables: bool = True) -> dict:
    """
    Generate comprehensive validation for sliding window methodology.
    
    Args:
        experiment_id: Experiment ID to validate
        jsonl_path: Path to experiments.jsonl file
        dataset_path: Path to dataset directory
        output_dir: Output directory (default: plot/output/)
        generate_plots: Whether to generate plots
        export_tables: Whether to export statistics tables
        
    Returns:
        Dictionary with validation results and output paths
    """
    # Load experiment results
    results = load_experiment_results(jsonl_path, experiment_id)
    
    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")
    
    # Extract configuration
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
    
    print(f"Validating {len(all_windows)} windows from experiment {experiment_id}...")
    
    # Determine time range
    min_unit = min(w.get('start_unit', 0) for w in all_windows)
    max_unit = max(w.get('end_unit', 0) for w in all_windows)
    
    # Load interaction data
    print(f"Loading {dataset_name} data ({granularity} {min_unit} to {max_unit})...")
    df = load_temporal_interaction_data(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        granularity=granularity,
        time_units=range(min_unit, max_unit + 1)
    )
    
    print(f"Loaded {len(df)} interactions from {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")
    
    # Compute statistics for all windows
    print("Computing window statistics...")
    stats_df = compute_all_window_statistics(df, all_windows, granularity)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path('plot/output')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate text report
    print("\nGenerating validation report...")
    report = generate_validation_report(stats_df, experiment_id, dataset_name, granularity)
    
    # Print to console
    print("\n" + report)
    
    # Save report to file
    report_path = output_dir / f"{experiment_id}_window_validation.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Saved validation report: {report_path}")
    
    output_paths = {'report': str(report_path)}
    
    # Export tables
    if export_tables:
        print("\nExporting statistics tables...")
        
        # LaTeX table
        latex_path = output_dir / f"{experiment_id}_window_stats.tex"
        export_statistics_table(stats_df, latex_path, format='latex')
        print(f"✓ Saved LaTeX table: {latex_path}")
        output_paths['latex_table'] = str(latex_path)
        
        # CSV table
        csv_path = output_dir / f"{experiment_id}_window_stats.csv"
        export_statistics_table(stats_df, csv_path, format='csv')
        print(f"✓ Saved CSV table: {csv_path}")
        output_paths['csv_table'] = str(csv_path)
        
        # Markdown table
        md_path = output_dir / f"{experiment_id}_window_stats.md"
        export_statistics_table(stats_df, md_path, format='markdown')
        print(f"✓ Saved Markdown table: {md_path}")
        output_paths['markdown_table'] = str(md_path)
    
    # Generate plots
    if generate_plots:
        print("\nGenerating validation plots...")
        
        # Comprehensive validation figure
        fig = plot_comprehensive_window_validation(stats_df, experiment_id, dataset_name)
        comprehensive_path = output_dir / f"{experiment_id}_window_validation.pdf"
        fig.savefig(comprehensive_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved comprehensive plot: {comprehensive_path}")
        output_paths['comprehensive_plot'] = str(comprehensive_path)
        
        # Individual plots for publication
        
        # 1. Data distribution
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        plot_window_data_distribution(stats_df, ax=ax1)
        dist_path = output_dir / f"{experiment_id}_data_distribution.pdf"
        fig1.savefig(dist_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"✓ Saved data distribution plot: {dist_path}")
        output_paths['distribution_plot'] = str(dist_path)
        
        # 2. Cold-start ratios
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        plot_cold_start_ratios(stats_df, ax=ax2)
        cs_path = output_dir / f"{experiment_id}_cold_start.pdf"
        fig2.savefig(cs_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"✓ Saved cold-start plot: {cs_path}")
        output_paths['cold_start_plot'] = str(cs_path)
        
        # 3. Overlap analysis
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        plot_window_overlap_analysis(stats_df, ax=ax3)
        overlap_path = output_dir / f"{experiment_id}_overlap_analysis.pdf"
        fig3.savefig(overlap_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"✓ Saved overlap analysis plot: {overlap_path}")
        output_paths['overlap_plot'] = str(overlap_path)
        
        # 4. Volume stability
        if len(stats_df) >= 3:  # Need at least 3 windows for rolling stats
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            plot_interaction_volume_stability(stats_df, ax=ax4)
            stability_path = output_dir / f"{experiment_id}_volume_stability.pdf"
            fig4.savefig(stability_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close(fig4)
            print(f"✓ Saved volume stability plot: {stability_path}")
            output_paths['stability_plot'] = str(stability_path)
    
    print(f"\n{'='*80}")
    print("Validation complete!")
    print(f"{'='*80}\n")
    
    return {
        'statistics': stats_df,
        'output_paths': output_paths,
        'experiment_id': experiment_id,
        'dataset_name': dataset_name,
        'num_windows': len(all_windows)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate sliding window methodology and generate statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate full validation report and plots
  python -m plot.window_validation --experiment-id exp_tfidf_temporal
  
  # Generate only the report (no plots)
  python -m plot.window_validation --experiment-id exp001 --no-plots
  
  # Custom output directory
  python -m plot.window_validation --experiment-id exp001 \\
      --output-dir validation_results/
  
  # Export tables only
  python -m plot.window_validation --experiment-id exp001 \\
      --no-plots --tables-only

Purpose:
  This script validates that the sliding window methodology (θ_data) 
  functioned as intended by analyzing:
  
  1. Data Distribution: Number of interactions and items per window
  2. Cold-Start Ratios: Percentage of new items/users in test sets
  3. Overlap Analysis: Train-test entity overlap
  4. Volume Stability: Consistency of data volume across windows
  
  The output includes text reports, LaTeX/CSV tables, and publication-ready
  plots suitable for Section 5.1 (Experimental Validation and Statistics).
        """
    )
    
    parser.add_argument('--experiment-id', required=True,
                       help='Experiment ID to validate (must be temporal experiment)')
    parser.add_argument('--jsonl-path', default='output/results/experiments.jsonl',
                       help='Path to experiments.jsonl file')
    parser.add_argument('--dataset-path', default='datasets/atomic_files',
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', '-o',
                       help='Output directory (default: plot/output/)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation, only produce text report')
    parser.add_argument('--no-tables', action='store_true',
                       help='Skip table export')
    
    args = parser.parse_args()
    
    try:
        validate_sliding_windows(
            experiment_id=args.experiment_id,
            jsonl_path=args.jsonl_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            generate_plots=not args.no_plots,
            export_tables=not args.no_tables
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
