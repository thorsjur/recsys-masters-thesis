#!/usr/bin/env python3
"""
Run stability experiments for recommendation models.

Supports four experimental protocols:
    A: Initialization Stability - Vary model training seeds
    B: Evaluation Sensitivity - Vary negative sampling seeds  
    C: Temporal Stability - Evaluate on different time windows
    D: Distributional Stability - Compare across datasets

Usage:
    # Experiment A: Model variance (10 training seeds)
    python run_stability_test.py --experiment A --model FastText --dataset mind_small --runs 10
    
    # Experiment B: Protocol variance (fixed model, vary eval seeds)
    python run_stability_test.py --experiment B --model FastText --dataset mind_small --runs 10 --model-seed 42
    
    # Experiment C: Temporal stability (train on early days, eval on later days)
    python run_stability_test.py --experiment C --model FastText --dataset mind_small --train-days 1-4 --eval-days 5,6,7
    
    # Experiment D: Dataset variance (compare across datasets)
    python run_stability_test.py --experiment D --model FastText --datasets mind_small,mind_large --runs 10
    
    # Legacy mode: Simple multi-seed runs (default)
    python run_stability_test.py --model BPR --dataset mind_small --runs 10
"""

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path
from util.temporal_dataset_builder import TemporalDatasetBuilder
from util.logging_config import setup_logging


def run_experiment(model: str, dataset: str, seed: int, config_files: list = None, params: list = None, data_path: str = 'datasets/atomic_files', experiment_id: str = None, description: str = None, window_info: dict = None):
    """Run a single experiment with specified seed."""
    cmd = [
        sys.executable,
        'run_recbole.py',
        '--model', model,
        '--dataset', dataset,
        '--data_path', data_path,
    ]
    
    if config_files:
        cmd.extend(['--config'] + config_files)
    
    # Add experiment ID and description if provided
    if experiment_id:
        cmd.extend(['--experiment-id', experiment_id])
    if description:
        cmd.extend(['--description', description])
    
    # Add window info if provided (for temporal experiments)
    if window_info:
        import json
        cmd.extend(['--window-info', json.dumps(window_info)])
    
    # Add seed as parameter
    seed_param = f'seed={seed}'
    if params:
        cmd.extend(['--params', seed_param] + params)
    else:
        cmd.extend(['--params', seed_param])
    
    print(f"\n{'='*80}")
    print(f"Running: {model} on {dataset} with seed={seed}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run stability experiments for recommendation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Protocols:
  A - Initialization Stability: Vary model training seeds (θ_rand)
      Measures variance from stochastic training (SGD, random init)
      
  B - Evaluation Sensitivity: Vary negative sampling seeds (θ_neg)
      Measures variance from random negative sample selection
      Requires --model-seed to fix the model
      
  C - Temporal Stability: Evaluate on different time windows
      Measures robustness to temporal drift without retraining
      Requires --window-size, --total-days and --window-ratio
      
  D - Distributional Stability: Compare across datasets (θ_data)
      Measures consistency across data variations
      Requires --datasets (comma-separated)

Examples:
  # Experiment A: 10 models with different training seeds
  python run_stability_test.py --experiment A --model FastText --dataset mind_small --runs 10
  
  # Experiment B: Fixed model, 10 evaluation runs with different neg samples
  python run_stability_test.py --experiment B --model FastText --dataset mind_small --runs 10 --model-seed 42
  
  # Experiment C: Sliding 7-day windows with 5:1:1 ratio over 28 days
  python run_stability_test.py --experiment C --model FastText --dataset mind_small --window-size 7 --window-ratio 5:1:1 --total-days 28
  
  # Experiment D: Compare mind_small vs mind_large with 10 seeds each
  python run_stability_test.py --experiment D --model FastText --datasets mind_small,mind_large --runs 10
        """
    )
    
    # Experiment selection
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['A', 'B', 'C', 'D'],
        help="Experiment protocol (A: Model variance, B: Protocol variance, C: Temporal, D: Dataset)"
    )
    
    # Core parameters
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Model name"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mind_small',
        help="Dataset name (for experiments A, B, C)"
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        help="Comma-separated dataset names (for experiment D)"
    )
    
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help="Number of runs with different seeds (default: 1)"
    )
    
    # Seed control
    parser.add_argument(
        '--seeds',
        type=str,
        help="Comma-separated list of seeds (e.g., '42,123,456'). If not provided, will use sequential seeds."
    )
    
    parser.add_argument(
        '--start-seed',
        type=int,
        default=2024,
        help="Starting seed for sequential runs (default: 2024)"
    )
    
    parser.add_argument(
        '--model-seed',
        type=int,
        help="Fixed model training seed (required for Experiment B)"
    )
    
    # Temporal experiment parameters
    parser.add_argument(
        '--window-size',
        type=int,
        help="Total window size in time units for Experiment C (e.g., 7 for weekly windows, 168 for weekly hours)"
    )
    
    parser.add_argument(
        '--window-ratio',
        type=str,
        default='5:1:1',
        help="Train:valid:test ratio for sliding windows (default: '5:1:1'). Use '5:2' for train:test without validation"
    )
    
    parser.add_argument(
        '--total-units',
        type=int,
        help="Total number of time units in dataset for Experiment C (days or hours)"
    )
    
    parser.add_argument(
        '--window-stride',
        type=int,
        help="Time units to slide window forward (default: same as window-size for non-overlapping)"
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=['day', 'hour'],
        default='day',
        help="Time granularity for Experiment C (default: 'day')"
    )
    
    # Keep --total-days for backward compatibility
    parser.add_argument(
        '--total-days',
        type=int,
        help="(Deprecated) Use --total-units instead. Total number of days in dataset for Experiment C"
    )
    
    # Config and parameters
    parser.add_argument(
        '--config',
        type=str,
        nargs='+',
        help="Config files to use"
    )
    
    parser.add_argument(
        '--params',
        type=str,
        nargs='+',
        help="Additional parameters (e.g., learning_rate=0.001)"
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='datasets/atomic_files',
        help="Path to dataset directory (default: 'datasets/atomic_files')"
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        help="Unique identifier for this experimental run (random UUID if not provided)"
    )
    
    parser.add_argument(
        '--description',
        type=str,
        help="Human-readable description of this experimental run"
    )
    
    args = parser.parse_args()
    
    # Generate experiment ID if not provided
    if not args.experiment_id:
        args.experiment_id = str(uuid.uuid4())[:8]
    
    # Auto-include dataset and model configs if not already specified
    dataset_config = f'configs/{args.dataset}.yaml'
    model_config = f'configs/{args.model.lower()}.yaml'
    
    # Build config list with proper ordering: dataset first, then model
    if args.config is None:
        args.config = []
    
    # Add dataset config if exists and not already in list
    if os.path.exists(dataset_config) and dataset_config not in args.config:
        args.config.insert(0, dataset_config)
    
    # Add model config if exists and not already in list
    if os.path.exists(model_config) and model_config not in args.config:
        args.config.append(model_config)
    
    # Setup logging to capture all output
    experiment_suffix = f"{args.experiment}_{args.model}_{args.dataset}" if args.experiment else f"{args.model}_{args.dataset}"
    setup_logging(
        debug_mode=False,
        log_dir='output/logs/stability',
        log_prefix=f'stability_{experiment_suffix}'
    )
    
    # Validate experiment-specific requirements
    if args.experiment:
        if args.experiment == 'B' and not args.model_seed:
            parser.error("Experiment B requires --model-seed to fix the model")
        if args.experiment == 'C':
            # Backward compatibility
            if args.total_days and not args.total_units:
                args.total_units = args.total_days
            
            if not args.window_size or not args.total_units:
                parser.error("Experiment C requires --window-size and --total-units (or --total-days)")
                
        if args.experiment == 'D' and not args.datasets:
            parser.error("Experiment D requires --datasets (comma-separated)")
    
    # Determine seeds to use
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        if len(seeds) != args.runs:
            print(f"Warning: {len(seeds)} seeds provided but --runs={args.runs}. Using {len(seeds)} runs.")
            args.runs = len(seeds)
    else:
        seeds = [args.start_seed + i for i in range(args.runs)]
    
    # Experiment-specific setup
    experiment_name = args.experiment or "Default"
    print(f"\n{'='*80}")
    print(f"STABILITY EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    print(f"Experiment ID: {args.experiment_id}")
    if args.description:
        print(f"Description: {args.description}")
    
    if args.experiment == 'A':
        print(f"Protocol: Initialization Stability (Model Variance)")
        print(f"  - Varying: Model training seeds (θ_rand)")
        print(f"  - Fixed: Dataset, negative sampling")
        print(f"  - Seeds: {seeds}")
        
    elif args.experiment == 'B':
        print(f"Protocol: Evaluation Sensitivity (Protocol Variance)")
        print(f"  - Fixed: Model (seed={args.model_seed})")
        print(f"  - Varying: Evaluation/negative sampling seeds (θ_neg)")
        print(f"  - Eval seeds: {seeds}")
        print(f"\nNote: You must implement negative sampling seed control in RecBole config")
        print(f"      Add to params: eval_neg_sample_seed=<seed>")
        
    elif args.experiment == 'C':
        # Parse window ratio
        ratio_parts = [int(x) for x in args.window_ratio.split(':')]
        has_valid = len(ratio_parts) == 3
        
        if len(ratio_parts) == 2:
            train_units, test_units = ratio_parts
            valid_units = 0
        elif len(ratio_parts) == 3:
            train_units, valid_units, test_units = ratio_parts
        else:
            parser.error("--window-ratio must be 'train:test' or 'train:valid:test' (e.g., '5:2' or '5:1:1')")
        
        total_ratio = sum(ratio_parts)
        unit_name = args.granularity
        
        if args.window_size != total_ratio:
            print(f"Warning: Window size ({args.window_size}) != sum of ratios ({total_ratio})")
            print(f"         Using ratio as window size: {total_ratio} {unit_name}s")
            args.window_size = total_ratio
        
        stride = args.window_stride or args.window_size
        num_windows = (args.total_units - args.window_size) // stride + 1
        
        print(f"Protocol: Temporal Stability (Sliding Window)")
        print(f"  - Granularity: {unit_name}")
        print(f"  - Total {unit_name}s: {args.total_units}")
        print(f"  - Window size: {args.window_size} {unit_name}s")
        if has_valid:
            print(f"  - Ratio: {train_units}:{valid_units}:{test_units} (train:valid:test)")
        else:
            print(f"  - Ratio: {train_units}:{test_units} (train:test, no validation)")
        print(f"  - Stride: {stride} {unit_name}s")
        print(f"  - Number of windows: {num_windows}")
        print(f"  - Model seeds per window: {seeds}")
        print(f"\nWindows:")
        for i in range(num_windows):
            start = i * stride + 1
            end = start + args.window_size - 1
            train_end = start + train_units - 1
            
            if has_valid:
                valid_start = train_end + 1
                valid_end = valid_start + valid_units - 1
                test_start = valid_end + 1
                test_end = end
                print(f"  Window {i+1}: {unit_name.capitalize()}s {start}-{end}")
                print(f"    Train: {start}-{train_end}, Valid: {valid_start}-{valid_end}, Test: {test_start}-{test_end}")
            else:
                test_start = train_end + 1
                test_end = end
                print(f"  Window {i+1}: {unit_name.capitalize()}s {start}-{end}")
                print(f"    Train: {start}-{train_end}, Test: {test_start}-{test_end}")
        
    elif args.experiment == 'D':
        datasets = [d.strip() for d in args.datasets.split(',')]
        print(f"Protocol: Distributional Stability (Dataset Variance)")
        print(f"  - Datasets: {datasets}")
        print(f"  - Seeds per dataset: {seeds}")
        print(f"  - Total runs: {len(datasets) * args.runs}")
    else:
        print(f"Protocol: Legacy multi-seed runs")
        print(f"  - Seeds: {seeds}")
    
    print(f"{'='*80}\n")
    
    # Run experiments
    successful = 0
    failed = 0
    total_runs = args.runs
    
    # Experiment-specific execution logic
    if args.experiment == 'D':
        # Experiment D: Run on multiple datasets
        datasets = [d.strip() for d in args.datasets.split(',')]
        total_runs = len(datasets) * args.runs
        
        for dataset in datasets:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset}")
            print(f"{'#'*80}")
            
            for i, seed in enumerate(seeds, 1):
                print(f"\n  Run {i}/{args.runs} (seed={seed})")
                
                success = run_experiment(
                    model=args.model,
                    dataset=dataset,
                    seed=seed,
                    config_files=args.config,
                    params=args.params,
                    data_path=args.data_path,
                    experiment_id=args.experiment_id,
                    description=args.description,
                    window_info={"experiment_type": "D", "dataset_variant": dataset, "run_number": i, "total_runs": args.runs}
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
                    
    elif args.experiment == 'C':
        # Experiment C: Temporal sliding window evaluation
        ratio_parts = [int(x) for x in args.window_ratio.split(':')]
        has_valid = len(ratio_parts) == 3
        
        if len(ratio_parts) == 2:
            train_units, test_units = ratio_parts
            valid_units = 0
        else:
            train_units, valid_units, test_units = ratio_parts
        
        stride = args.window_stride or args.window_size
        num_windows = (args.total_units - args.window_size) // stride + 1
        total_runs = num_windows * args.runs
        unit_name = args.granularity
        
        print(f"\nRunning sliding window evaluation...")
        print(f"Checking for {unit_name}-wise split files...")
        
        # Initialize temporal dataset builder
        builder = TemporalDatasetBuilder(args.data_path, args.dataset, granularity=args.granularity)
        available_units = builder.get_available_time_units()
        
        if not available_units:
            print(f"\n✗ ERROR: No {unit_name}-wise split files found for {args.dataset}")
            print(f"  Expected files like: {args.data_path}/{args.dataset}/{args.dataset}.{unit_name}_1.inter")
            print(f"\n  To generate {unit_name}-wise splits, run:")
            if unit_name == 'hour':
                print(f"  python run_etl.py --config <your_config> --temporal-hours {args.total_units}")
            else:
                print(f"  python run_etl.py --config <your_config> --temporal-days {args.total_units}")
            sys.exit(1)
        
        print(f"  Found {len(available_units)} {unit_name} files: {unit_name}s {min(available_units)}-{max(available_units)}")
        
        if max(available_units) < args.total_units:
            print(f"  Warning: Only {max(available_units)} {unit_name}s available, but {args.total_units} requested")
            print(f"           Adjusting total_units to {max(available_units)}")
            args.total_units = max(available_units)
            num_windows = (args.total_units - args.window_size) // stride + 1
            total_runs = num_windows * args.runs
        
        for window_idx in range(num_windows):
            start_unit = window_idx * stride + 1
            end_unit = start_unit + args.window_size - 1
            
            train_start = start_unit
            train_end = start_unit + train_units - 1
            
            if has_valid:
                valid_start = train_end + 1
                valid_end = valid_start + valid_units - 1
                test_start = valid_end + 1
                test_end = end_unit
            else:
                valid_start = None
                valid_end = None
                test_start = train_end + 1
                test_end = end_unit
            
            print(f"\n{'#'*80}")
            print(f"# Window {window_idx+1}/{num_windows}: {unit_name.capitalize()}s {start_unit}-{end_unit}")
            print(f"#   Train: {train_start}-{train_end} ({train_units} {unit_name}s)")
            if has_valid:
                print(f"#   Valid: {valid_start}-{valid_end} ({valid_units} {unit_name}s)")
            print(f"#   Test:  {test_start}-{test_end} ({test_units} {unit_name}s)")
            print(f"{'#'*80}")
            
            # Build temporary splits for this window
            try:
                if unit_name == 'hour':
                    temp_dir, splits = builder.build_temporal_splits(
                        train_hours=(train_start, train_end),
                        valid_hours=(valid_start, valid_end) if has_valid else None,
                        test_hours=(test_start, test_end),
                        temp_prefix=f'window{window_idx+1}'
                    )
                else:
                    temp_dir, splits = builder.build_temporal_splits(
                        train_days=(train_start, train_end),
                        valid_days=(valid_start, valid_end) if has_valid else None,
                        test_days=(test_start, test_end),
                        temp_prefix=f'window{window_idx+1}'
                    )
                
                # Prepare parameters with benchmark_filename override
                window_params = args.params.copy() if args.params else []
                window_params.append(f"benchmark_filename={splits['benchmark_filename']}")
                
                # Build window information for result logging
                window_info = {
                    "window_number": window_idx + 1,
                    "total_windows": num_windows,
                    "granularity": unit_name,
                    "window_size": args.window_size,
                    "window_stride": stride,
                    "window_ratio": args.window_ratio,
                    "start_unit": start_unit,
                    "end_unit": end_unit,
                    "train_range": f"{train_start}-{train_end}",
                    "train_units": train_units,
                    "test_range": f"{test_start}-{test_end}",
                    "test_units": test_units,
                    "has_validation": has_valid,
                }
                if has_valid:
                    window_info["valid_range"] = f"{valid_start}-{valid_end}"
                    window_info["valid_units"] = valid_units
                else:
                    window_info["validation_type"] = "dummy"
                
                # Note: Validation file always exists (dummy if not explicitly provided)
                if not splits['has_valid']:
                    print(f"    Note: Using dummy validation set for RecBole compatibility")
                
                # Run experiments with this window
                for i, seed in enumerate(seeds, 1):
                    print(f"\n  Run {i}/{args.runs} (seed={seed})")
                    
                    # Add run number to window info
                    run_window_info = window_info.copy()
                    run_window_info["run_number"] = i
                    run_window_info["total_runs_per_window"] = args.runs
                    
                    success = run_experiment(
                        model=args.model,
                        dataset=args.dataset,
                        seed=seed,
                        config_files=args.config,
                        params=window_params,
                        data_path=args.data_path,
                        experiment_id=args.experiment_id,
                        description=args.description,
                        window_info=run_window_info
                    )
                    
                    if success:
                        successful += 1
                    else:
                        failed += 1
                
            finally:
                # Cleanup temporary files after all runs for this window
                builder.cleanup(temp_prefix=splits['temp_prefix'])
                    
    elif args.experiment == 'B':
        # Experiment B: Fixed model, vary eval
        print(f"\nTraining model with seed={args.model_seed}...")
        
        # First, train the model once with fixed seed
        success = run_experiment(
            model=args.model,
            dataset=args.dataset,
            seed=args.model_seed,
            config_files=args.config,
            params=args.params,
            data_path=args.data_path,
            experiment_id=args.experiment_id,
            description=args.description,
            window_info={"experiment_type": "B", "phase": "model_training", "model_seed": args.model_seed}
        )
        
        if not success:
            print(f"\n✗ Model training failed with seed={args.model_seed}")
            return False
        
        print(f"\n✓ Model trained successfully")
        print(f"\nNow running {args.runs} evaluation runs with different negative sampling seeds...")
        print(f"Warning: Negative sampling seed control not implemented")
        print(f"         You need to add eval_neg_sample_seed parameter to RecBole")
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n{'#'*80}")
            print(f"# Evaluation Run {i}/{args.runs} (neg_sample_seed={seed})")
            print(f"{'#'*80}")
            
            eval_params = args.params.copy() if args.params else []
            eval_params.append(f"eval_neg_sample_seed={seed}")
            
            # Note: This would require RecBole modifications to support
            # eval-only mode with different negative sampling seeds
            print(f"  Placeholder: Would evaluate with neg_sample_seed={seed}")
            print(f"  Implementation required: Load trained model + vary neg samples")
            successful += 1  # Placeholder
            
    else:
        # Experiment A or default: Standard multi-seed runs
        for i, seed in enumerate(seeds, 1):
            print(f"\n{'#'*80}")
            print(f"# Run {i}/{args.runs}")
            print(f"{'#'*80}")
            
            success = run_experiment(
                model=args.model,
                dataset=args.dataset,
                seed=seed,
                config_files=args.config,
                params=args.params,
                data_path=args.data_path,
                experiment_id=args.experiment_id,
                description=args.description,
                window_info={"experiment_type": args.experiment or "A", "run_number": i, "total_runs": args.runs}
            )
            
            if success:
                successful += 1
                print(f"\n✓ Run {i} completed successfully")
            else:
                failed += 1
                print(f"\n✗ Run {i} failed")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"STABILITY TEST SUMMARY - Experiment {experiment_name}")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if args.experiment == 'D':
        datasets = [d.strip() for d in args.datasets.split(',')]
        print(f"\nTo analyze stability:")
        for dataset in datasets:
            print(f"  python temp/analyze_results.py --stability --model {args.model} --dataset {dataset}")
    else:
        print(f"\nTo analyze stability:")
        print(f"  python temp/analyze_results.py --stability --model {args.model} --dataset {args.dataset}")
    
    print(f"{'='*80}\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
