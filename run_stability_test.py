import argparse
import sys

from util.logging_config import setup_logging
from stability.base import parse_seeds
from stability.experiment_temporal import run_temporal_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run temporal stability experiments for recommendation models",
    )

    parser.add_argument("--model", type=str, required=True, help="Model name")

    parser.add_argument("--dataset", type=str, default="mind_small", help="Dataset name")

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help=("Number of runs per window, each with a different seed (default: 3). " "Ignored if --seeds is provided."),
    )

    # Seed control
    parser.add_argument(
        "--seeds",
        type=str,
        help="Comma-separated list of seeds (e.g., '42,123,456'). If not provided, will use sequential seeds.",
    )

    parser.add_argument("--start-seed", type=int, default=42, help=argparse.SUPPRESS)

    # Temporal experiment parameters
    parser.add_argument(
        "--window-size",
        type=int,
        help="Total window size in time units (e.g., 7 for weekly windows, 168 for weekly hours)",
    )

    parser.add_argument(
        "--window-ratio",
        type=str,
        default="5:1:1",
        help="Train:valid:test ratio for sliding windows (default: '5:1:1'). Use '5:2' for train:test without validation",
    )

    parser.add_argument("--total-units", type=int, help="Total number of time units in dataset (days or hours)")

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
        help="Time granularity (default: 'day')",
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

    parser.add_argument("--experiment-id", type=str, required=True, help="Unique identifier for this experiment")

    parser.add_argument("--description", type=str, help="Human-readable description of this experiment")

    args = parser.parse_args()

    # Setup logging to capture output
    experiment_suffix = f"{args.model}_{args.dataset}"
    setup_logging(debug_mode=False, log_dir="output/logs/stability", log_prefix=f"stability_{experiment_suffix}")

    if not args.window_size or not args.total_units:
        parser.error("Temporal stability requires --window-size and --total-units")

    # Set seeds to use
    seeds = parse_seeds(args.seeds, args.runs, args.start_seed)

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
