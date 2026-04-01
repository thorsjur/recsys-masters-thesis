import logging
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple
from stability.base import (
    run_experiment,
    ExperimentResult,
    ExperimentSummary,
    print_experiment_header,
)
from util.temporal_dataset_builder import TemporalDatasetBuilder

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for a single evaluation window."""

    window_idx: int
    start_unit: int
    end_unit: int
    train_start: int
    train_end: int
    valid_start: Optional[int]
    valid_end: Optional[int]
    test_start: int
    test_end: int
    has_valid: bool


@dataclass
class TemporalConfig:
    """Configuration for temporal stability experiment."""

    train_units: int
    valid_units: int
    test_units: int
    has_valid: bool
    window_size: int
    stride: int
    granularity: str

    @classmethod
    def from_ratio(
        cls, ratio_str: str, window_size: int, window_stride: Optional[int], granularity: str
    ) -> "TemporalConfig":
        """Create config from ratio string like '5:1:1' or '5:2'."""
        ratio_parts = [int(x) for x in ratio_str.split(":")]

        if len(ratio_parts) == 2:
            train_units, test_units = ratio_parts
            valid_units = 0
            has_valid = False
        elif len(ratio_parts) == 3:
            train_units, valid_units, test_units = ratio_parts
            has_valid = True
        else:
            raise ValueError("--window-ratio must be 'train:test' or 'train:valid:test'")

        total_ratio = train_units + valid_units + test_units
        if window_size != total_ratio:
            logger.warning(
                f"Window size ({window_size}) != sum of ratios ({total_ratio}). "
                f"Using ratio as window size: {total_ratio} {granularity}s"
            )
            window_size = total_ratio

        return cls(
            train_units=train_units,
            valid_units=valid_units,
            test_units=test_units,
            has_valid=has_valid,
            window_size=window_size,
            stride=window_stride or window_size,
            granularity=granularity,
        )

    @property
    def unit_name(self) -> str:
        return self.granularity


def calculate_windows(total_units: int, config: TemporalConfig) -> List[WindowConfig]:
    """Calculate all window configurations."""
    num_windows = (total_units - config.window_size) // config.stride + 1
    windows = []

    for window_idx in range(num_windows):
        start_unit = window_idx * config.stride + 1
        end_unit = start_unit + config.window_size - 1
        train_start = start_unit
        train_end = start_unit + config.train_units - 1

        if config.has_valid:
            valid_start = train_end + 1
            valid_end = valid_start + config.valid_units - 1
            test_start = valid_end + 1
        else:
            valid_start = None
            valid_end = None
            test_start = train_end + 1

        windows.append(
            WindowConfig(
                window_idx=window_idx,
                start_unit=start_unit,
                end_unit=end_unit,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=end_unit,
                has_valid=config.has_valid,
            )
        )

    return windows


def print_protocol_info(config: TemporalConfig, total_units: int, num_windows: int, seeds: List[int]):
    """Log protocol configuration information."""
    ratio_str = (
        f"{config.train_units}:{config.valid_units}:{config.test_units} (train:valid:test)"
        if config.has_valid
        else f"{config.train_units}:{config.test_units} (train:test, no validation)"
    )
    logger.info(
        f"Protocol: Temporal Stability (Sliding Window) | "
        f"Granularity: {config.unit_name} | "
        f"Total {config.unit_name}s: {total_units} | "
        f"Window size: {config.window_size} {config.unit_name}s | "
        f"Ratio: {ratio_str} | "
        f"Stride: {config.stride} {config.unit_name}s | "
        f"Windows: {num_windows} | "
        f"Seeds per window: {seeds}"
    )


def print_window_plan(windows: List[WindowConfig], config: TemporalConfig):
    """Log the planned windows before execution."""
    logger.info("Planned windows:")
    for w in windows:
        if w.has_valid:
            detail = (
                f"Window {w.window_idx+1}: {config.unit_name.capitalize()}s {w.start_unit}-{w.end_unit} | "
                f"Train: {w.train_start}-{w.train_end}, Valid: {w.valid_start}-{w.valid_end}, Test: {w.test_start}-{w.test_end}"
            )
        else:
            detail = (
                f"Window {w.window_idx+1}: {config.unit_name.capitalize()}s {w.start_unit}-{w.end_unit} | "
                f"Train: {w.train_start}-{w.train_end}, Test: {w.test_start}-{w.test_end}"
            )
        logger.info(f"  {detail}")


def print_window_header(w: WindowConfig, num_windows: int, config: TemporalConfig):
    """Log header for a single window execution."""
    valid_info = (
        f" | Valid: {w.valid_start}-{w.valid_end} ({config.valid_units} {config.unit_name}s)"
        if w.has_valid
        else ""
    )
    logger.info(
        f"Window {w.window_idx+1}/{num_windows}: {config.unit_name.capitalize()}s {w.start_unit}-{w.end_unit} | "
        f"Train: {w.train_start}-{w.train_end} ({config.train_units} {config.unit_name}s){valid_info} | "
        f"Test: {w.test_start}-{w.test_end} ({config.test_units} {config.unit_name}s)"
    )


def validate_available_units(
    builder: TemporalDatasetBuilder,
    total_units: int,
    config: TemporalConfig,
    dataset: str,
    data_path: str,
) -> Tuple[int, List[int]]:
    """Validate and return available time units, adjusting total if needed."""
    available_units = builder.get_available_time_units()

    if not available_units:
        etl_flag = "--temporal-hours" if config.unit_name == "hour" else "--temporal-days"
        logger.error(
            f"No {config.unit_name}-wise split files found for {dataset}. "
            f"Expected files like: {data_path}/{dataset}/{dataset}.{config.unit_name}_1.inter. "
            f"To generate {config.unit_name}-wise splits, run: "
            f"python run_etl.py --config <your_config> {etl_flag} {total_units}"
        )
        sys.exit(1)

    logger.info(
        f"Found {len(available_units)} {config.unit_name} files: "
        f"{config.unit_name}s {min(available_units)}-{max(available_units)}"
    )

    if max(available_units) < total_units:
        logger.warning(
            f"Only {max(available_units)} {config.unit_name}s available, but {total_units} requested. "
            f"Adjusting total_units to {max(available_units)}"
        )
        total_units = max(available_units)

    return total_units, available_units


def build_window_info(w: WindowConfig, config: TemporalConfig, num_windows: int, window_ratio: str) -> dict:
    """Build window information dictionary for result logging."""
    window_info = {
        "window_number": w.window_idx + 1,
        "total_windows": num_windows,
        "granularity": config.granularity,
        "window_size": config.window_size,
        "window_stride": config.stride,
        "window_ratio": window_ratio,
        "start_unit": w.start_unit,
        "end_unit": w.end_unit,
        "train_range": f"{w.train_start}-{w.train_end}",
        "train_units": config.train_units,
        "test_range": f"{w.test_start}-{w.test_end}",
        "test_units": config.test_units,
        "has_validation": w.has_valid,
    }
    if w.has_valid:
        window_info["valid_range"] = f"{w.valid_start}-{w.valid_end}"
        window_info["valid_units"] = config.valid_units
    else:
        window_info["validation_type"] = "dummy"
    return window_info


def run_window_experiments(
    w: WindowConfig,
    config: TemporalConfig,
    builder: TemporalDatasetBuilder,
    model: str,
    dataset: str,
    seeds: List[int],
    config_files: Optional[List[str]],
    params: Optional[List[str]],
    data_path: str,
    experiment_id: Optional[str],
    description: Optional[str],
    num_windows: int,
    window_ratio: str,
    summary: ExperimentSummary,
):
    """Run all seed experiments for a single window."""
    # Build temporal splits
    valid_range: Optional[Tuple[int, int]] = None
    if w.has_valid and w.valid_start is not None and w.valid_end is not None:
        valid_range = (w.valid_start, w.valid_end)

    if config.granularity == "hour":
        temp_dir, splits = builder.build_temporal_splits(
            train_hours=(w.train_start, w.train_end),
            valid_hours=valid_range,
            test_hours=(w.test_start, w.test_end),
            temp_prefix=f"window{w.window_idx+1}",
        )
    else:
        temp_dir, splits = builder.build_temporal_splits(
            train_days=(w.train_start, w.train_end),
            valid_days=valid_range,
            test_days=(w.test_start, w.test_end),
            temp_prefix=f"window{w.window_idx+1}",
        )

    # Prepare parameters
    window_params = params.copy() if params else []
    window_params.append(f"benchmark_filename={splits['benchmark_filename']}")

    window_info = build_window_info(w, config, num_windows, window_ratio)
    window_info["temp_prefix"] = splits["temp_prefix"]
    window_info["temp_dir"] = str(temp_dir)

    if not splits["has_valid"]:
        logger.info("Using dummy validation set for RecBole compatibility")

    # Run experiments for each seed
    for i, seed in enumerate(seeds, 1):
        logger.info(f"Run {i}/{len(seeds)} (seed={seed})")

        run_window_info = window_info.copy()
        run_window_info["run_number"] = i
        run_window_info["total_runs_per_window"] = len(seeds)

        success = run_experiment(
            model=model,
            dataset=dataset,
            seed=seed,
            config_files=config_files,
            params=window_params,
            data_path=data_path,
            experiment_id=experiment_id,
            description=description,
            window_info=run_window_info,
        )

        result = ExperimentResult(
            success=success,
            seed=seed,
            model=model,
            dataset=dataset,
            window_info=run_window_info,
        )
        summary.add_result(result)

    return splits["temp_prefix"]


def run_temporal_experiment(
    model: str,
    dataset: str,
    seeds: List[int],
    window_size: int,
    total_units: int,
    window_ratio: str = "5:1:1",
    window_stride: Optional[int] = None,
    granularity: str = "day",
    config_files: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    data_path: str = "data/atomic_files",
    experiment_id: Optional[str] = None,
    description: Optional[str] = None,
) -> ExperimentSummary:
    """
    Run Temporal Stability experiment.
    Uses sliding windows to evaluate temporal drift robustness.

    Returns:
        ExperimentSummary with results
    """
    print_experiment_header("Temporal Stability", experiment_id or "N/A", description)

    # Parse and validate configuration
    config = TemporalConfig.from_ratio(window_ratio, window_size, window_stride, granularity)
    num_windows = (total_units - config.window_size) // config.stride + 1

    # Log configuration
    print_protocol_info(config, total_units, num_windows, seeds)
    windows = calculate_windows(total_units, config)
    print_window_plan(windows, config)

    # Initialize and validate temporal dataset builder
    logger.info(f"Running sliding window evaluation...")
    logger.info(f"Checking for {config.unit_name}-wise split files...")

    builder = TemporalDatasetBuilder(data_path, dataset, granularity=granularity)
    total_units, _ = validate_available_units(builder, total_units, config, dataset, data_path)

    # Recalculate windows if total_units was adjusted
    num_windows = (total_units - config.window_size) // config.stride + 1
    windows = calculate_windows(total_units, config)

    summary = ExperimentSummary()

    # Execute each window
    for w in windows:
        print_window_header(w, num_windows, config)

        try:
            temp_prefix = run_window_experiments(
                w=w,
                config=config,
                builder=builder,
                model=model,
                dataset=dataset,
                seeds=seeds,
                config_files=config_files,
                params=params,
                data_path=data_path,
                experiment_id=experiment_id,
                description=description,
                num_windows=num_windows,
                window_ratio=window_ratio,
                summary=summary,
            )
        finally:
            builder.cleanup(temp_prefix=temp_prefix)

    summary.print_summary("Temporal Stability", model, dataset)
    return summary
