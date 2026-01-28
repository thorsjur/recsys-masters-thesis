import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    success: bool
    seed: int
    model: str
    dataset: str
    window_info: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentSummary:
    """Summary of an experiment batch."""

    total_runs: int = 0
    successful: int = 0
    failed: int = 0
    results: List[ExperimentResult] = field(default_factory=list)

    def add_result(self, result: ExperimentResult):
        """Add a result and update counts."""
        self.results.append(result)
        self.total_runs += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1

    def print_summary(self, experiment_name: str, model: str, dataset: str):
        """Log experiment summary."""
        logger.info(
            f"STABILITY TEST SUMMARY - {experiment_name} | "
            f"Total runs: {self.total_runs} | "
            f"Successful: {self.successful} | "
            f"Failed: {self.failed}"
        )
        logger.info(
            f"To analyze stability: " f"python temp/analyze_results.py --stability --model {model} --dataset {dataset}"
        )


def parse_seeds(seeds_str: Optional[str], runs: int, start_seed: int = 2024) -> List[int]:
    """
    Parse seeds from string or generate sequential seeds.

    Returns:
        List of integer seeds
    """
    if seeds_str:
        seeds = [int(s.strip()) for s in seeds_str.split(",")]
        return seeds
    return [start_seed + i for i in range(runs)]


def run_experiment(
    model: str,
    dataset: str,
    seed: int,
    config_files: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    data_path: str = "data/atomic_files",
    experiment_id: Optional[str] = None,
    description: Optional[str] = None,
    window_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Run a single experiment with specified seed.

    Returns:
        True if experiment succeeded, False otherwise
    """
    cmd = [
        sys.executable,
        "run_recbole.py",
        "--model",
        model,
        "--dataset",
        dataset,
        "--data_path",
        data_path,
    ]

    if config_files:
        cmd.extend(["--config"] + config_files)

    if experiment_id:
        cmd.extend(["--experiment-id", experiment_id])
    if description:
        cmd.extend(["--description", description])

    if window_info:
        cmd.extend(["--window-info", json.dumps(window_info)])

    # Add seed as parameter
    seed_param = f"seed={seed}"
    if params:
        cmd.extend(["--params", seed_param] + params)
    else:
        cmd.extend(["--params", seed_param])

    logger.info(f"Running: {model} on {dataset} with seed={seed}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def print_experiment_header(experiment_name: str, experiment_id: str, description: Optional[str] = None):
    """Log experiment header information."""
    header = f"STABILITY EXPERIMENT: {experiment_name} | Experiment ID: {experiment_id}"
    if description:
        header += f" | Description: {description}"
    logger.info(header)
