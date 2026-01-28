#!/usr/bin/env python3

import argparse
import sys
import uuid

from util.logging_config import setup_logging
from stability.base import parse_seeds
from stability.experiment_temporal import run_temporal_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run stability experiments for recommendation models",
    )

    # Experiment selection, currently only Experiment A (temporal stability) is implemented
    parser.add_argument(
        "--experiment",
        type=str,
        default="A",
        choices=["A"],
        help="Experiment protocol (A: Temporal Stability)",
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")

    parser.add_argument("--dataset", type=str, default="mind_small", help="Dataset name (for experiment A)")

    parser.add_argument("--runs", type=int, default=1, help="Number of runs with different seeds (default: 1)")

    # Seed control
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '42,123,456'). If not provided, will use sequential seeds.",
    )

    parser.add_argument(
        "--start-seed", type=int, default=2024, help="Starting seed for sequential runs (default: 2024)"
    )

    # Temporal experiment parameters
    parser.add_argument(
        "--window-size",
        type=int,
        help="Total window size in time units for Experiment A (e.g., 7 for weekly windows, 168 for weekly hours)",
    )

    parser.add_argument(
        "--window-ratio",
        type=str,
        default="5:1:1",
        help="Train:valid:test ratio for sliding windows (default: '5:1:1'). Use '5:2' for train:test without validation",
    )

    parser.add_argument(
        "--total-units", type=int, help="Total number of time units in dataset for Experiment A (days or hours)"
    )

    parser.add_argument(
        "--window-stride",
        type=int,
        help="Time units to slide window forward (default: same as window-size for non-overlapping)",
    )

    parser.add_argument(
        "--granularity",
        type=str,
        choices=["day", "hour"],
        default="day",
        help="Time granularity for Experiment A (default: 'day')",
    )

    # Config and parameters
    parser.add_argument("--config", type=str, nargs="+", help="Config files to use")

    parser.add_argument("--params", type=str, nargs="+", help="Additional parameters")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/atomic_files",
        help="Path to dataset directory (default: 'data/atomic_files')",
    )

    parser.add_argument(
        "--experiment-id", type=str, help="Unique identifier for this experimental run (random UUID if not provided)"
    )

    parser.add_argument("--description", type=str, help="Human-readable description of this experimental run")

    args = parser.parse_args()

    # Generate experiment ID if not provided
    if not args.experiment_id:
        args.experiment_id = str(uuid.uuid4())[:8]

    # Setup logging to capture output
    experiment_suffix = (
        f"{args.experiment}_{args.model}_{args.dataset}" if args.experiment else f"{args.model}_{args.dataset}"
    )
    setup_logging(debug_mode=False, log_dir="output/logs/stability", log_prefix=f"stability_{experiment_suffix}")

    # Validate experiment-specific requirements
    if args.experiment and args.experiment == "A":
        if not args.window_size or not args.total_units:
            parser.error("Experiment A requires --window-size and --total-units")

    # Set seeds to use
    seeds = parse_seeds(args.seeds, args.runs, args.start_seed)
    if args.seeds and len(seeds) != args.runs:
        print(f"Warning: {len(seeds)} seeds provided but --runs={args.runs}. Using {len(seeds)} runs.")
        args.runs = len(seeds)

    elif args.experiment == "A":
        summary = run_temporal_experiment(
            model=args.model,
            dataset=args.dataset,
            seeds=seeds,
            window_size=args.window_size,
            total_units=args.total_units,
            window_ratio=args.window_ratio,
            window_stride=args.window_stride,
            granularity=args.granularity,
            config_files=args.config,
            params=args.params,
            data_path=args.data_path,
            experiment_id=args.experiment_id,
            description=args.description,
        )

    return summary.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
