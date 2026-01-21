"""
Experiment orchestrator for Slurm-based parallel execution.

Handles the high-level workflow of:
- Generating experiment tasks from configuration
- Building temporal datasets
- Coordinating parallel job submission
- Monitoring progress
- Recovery and restart operations
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from slurm.slurm_constants import DEFAULT_CONDA_ENV
from slurm.state import (
    JobState,
    ExperimentState,
    TaskInfo,
    ExperimentConfig,
    StateManager,
)
from slurm.job_manager import SlurmJobManager
from stability.experiment_temporal import TemporalConfig, calculate_windows
from util.temporal_dataset_builder import TemporalDatasetBuilder

logger = logging.getLogger(__name__)


@dataclass
class SubmissionStats:
    """Statistics from a job submission batch."""

    submitted: int = 0
    skipped: int = 0
    failed: int = 0
    job_ids: List[str] = field(default_factory=list)


class ExperimentOrchestrator:
    """
    Orchestrates experiment execution on Slurm.

    Handles the complete lifecycle:
    1. Initialize experiment and generate tasks
    2. Build temporal dataset splits
    3. Submit jobs (individual or array)
    4. Monitor progress
    5. Handle failures and retries
    6. Cleanup
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        job_manager: Optional[SlurmJobManager] = None,
        dry_run: bool = False,
    ):
        self.state_manager = state_manager or StateManager()
        self.job_manager = job_manager or SlurmJobManager(self.state_manager, dry_run=dry_run)
        self.dry_run = dry_run
        self._window_configs: Dict[str, Dict] = {}
        self._prep_job_id: Optional[str] = None

    def submit_prep_job(self, experiment_id: str) -> Optional[str]:
        """
        Submit the dataset preparation job.

        Returns:
            Slurm job ID if successful, None otherwise
        """
        prep_job_id = self.job_manager.submit_prep_job(experiment_id)
        if prep_job_id:
            self._prep_job_id = prep_job_id
            logger.info(f"Submitted prep job {prep_job_id}")
        return prep_job_id

    def submit_with_prep(
        self,
        experiment_id: str,
        use_array_jobs: bool = True,
        max_parallel: int = 10,
        warmup: bool = False,
    ) -> SubmissionStats:
        """
        Submit prep job first, then all tasks with dependency.
        """
        # Submit prep job first
        prep_job_id = self.submit_prep_job(experiment_id)
        if not prep_job_id:
            logger.error("Failed to submit prep job")
            return SubmissionStats(failed=1)

        logger.info(f"Prep job submitted: {prep_job_id}")
        logger.info("Submitting experiment tasks with dependency on prep job...")

        # Submit all tasks with dependency on prep job
        stats = self.submit_all(
            experiment_id,
            use_array_jobs=use_array_jobs,
            max_parallel=max_parallel,
            prep_job_id=prep_job_id,
            warmup=warmup,
        )

        # Add prep job to stats
        stats.job_ids.insert(0, f"{prep_job_id} (prep)")
        return stats

    def create_experiment(
        self,
        experiment_id: str,
        model: str,
        dataset: str,
        window_size: int,
        total_units: int,
        seeds: List[int],
        window_ratio: str = "5:1:1",
        window_stride: Optional[int] = None,
        granularity: str = "day",
        config_files: Optional[List[str]] = None,
        params: Optional[List[str]] = None,
        data_path: str = "datasets/atomic_files",
        description: Optional[str] = None,
        # IDUN Slurm options
        partition: str = "CPUQ",  # CPUQ, GPUQ, or short
        time_limit: str = "48:00:00",
        memory: str = "16G",
        cpus_per_task: int = 4,
        ntasks_per_node: int = 1,
        nodes: int = 1,
        # GPU configuration (IDUN style)
        gpu_count: int = 0,
        gpu_type: Optional[str] = None,  # p100, v100, a100, h100
        gpu_constraint: Optional[str] = None,  # gpu16g, gpu32g, gpu40g, gpu80g, sxm4
        # Account is REQUIRED on IDUN
        account: str = "",
        modules: Optional[List[str]] = None,
        conda_env: Optional[str] = DEFAULT_CONDA_ENV,
        # Email notifications
        mail_user: Optional[str] = None,
        mail_type: str = "FAIL",
        force: bool = False,
    ) -> Tuple[ExperimentConfig, List[TaskInfo]]:
        """
        Create a new experiment with all task definitions for IDUN cluster.

        Returns:
            Tuple of (ExperimentConfig, List[TaskInfo])
        """
        # Validate account (required on IDUN)
        if not account:
            raise ValueError(
                "Slurm account is required on IDUN. "
                "Use --account to specify your allocation account. "
                "Run 'sacctmgr show assoc format=Account%15,User,QOS | grep $USER' to find your accounts."
            )

        # Check for existing experiment
        if self.state_manager.experiment_exists(experiment_id):
            if force:
                logger.warning(f"Overwriting existing experiment: {experiment_id}")
                self.state_manager.backup_experiment(experiment_id)
                self.state_manager.delete_experiment(experiment_id)
            else:
                raise ValueError(
                    f"Experiment '{experiment_id}' already exists. "
                    f"Use --force to overwrite or --resume to continue."
                )

        # Auto-select partition based on GPU usage
        if gpu_count > 0 and partition == "CPUQ":
            partition = "GPUQ"
            logger.info(f"Auto-selected GPUQ partition for GPU job")

        # Create config
        config = ExperimentConfig(
            experiment_id=experiment_id,
            model=model,
            dataset=dataset,
            window_size=window_size,
            total_units=total_units,
            window_ratio=window_ratio,
            window_stride=window_stride or window_size,
            granularity=granularity,
            seeds=seeds,
            config_files=config_files or [],
            params=params or [],
            data_path=data_path,
            description=description,
            partition=partition,
            time_limit=time_limit,
            memory=memory,
            cpus_per_task=cpus_per_task,
            ntasks_per_node=ntasks_per_node,
            nodes=nodes,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            gpu_constraint=gpu_constraint,
            account=account,
            modules=modules or [],
            conda_env=conda_env,
            mail_user=mail_user,
            mail_type=mail_type,
        )

        # Generate temporal config and windows
        temporal_config = TemporalConfig.from_ratio(window_ratio, window_size, window_stride, granularity)
        windows = calculate_windows(total_units, temporal_config)

        # Generate tasks (one per window-seed combination)
        tasks = []
        for window in windows:
            for seed in seeds:
                task_id = f"w{window.window_idx + 1}_s{seed}"
                task = TaskInfo(
                    task_id=task_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    state=JobState.PENDING,
                )
                tasks.append(task)

        # Create state file
        self.state_manager.create_experiment(config, tasks)

        logger.info(
            f"Created experiment '{experiment_id}' with {len(tasks)} tasks "
            f"({len(windows)} windows × {len(seeds)} seeds)"
        )

        return config, tasks

    def prepare_temporal_datasets(
        self,
        experiment_id: str,
        save_to_file: bool = True,
    ) -> Dict[str, Dict]:
        """
        Prepare temporal dataset splits for all windows.

        Args:
            experiment_id: Experiment identifier
            save_to_file: If True, save window configs to JSON file for array jobs

        Returns:
            Dictionary mapping task_id to window configuration
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)

        # Initialize builder
        builder = TemporalDatasetBuilder(config.data_path, config.dataset, granularity=config.granularity)

        # Verify available time units
        available_units = builder.get_available_time_units()
        if not available_units:
            raise RuntimeError(
                f"No {config.granularity}-wise split files found for {config.dataset}. "
                f"Run ETL first: python run_etl.py --config <config> "
                f"--temporal-{config.granularity}s {config.total_units}"
            )

        max_available = max(available_units)
        if max_available < config.total_units:
            logger.warning(
                f"Only {max_available} {config.granularity}s available, "
                f"but {config.total_units} requested. Adjusting."
            )
            config.total_units = max_available
            self.state_manager.save_experiment(config, progress, tasks)

        # Generate temporal config and windows
        temporal_config = TemporalConfig.from_ratio(
            config.window_ratio,
            config.window_size,
            config.window_stride,
            config.granularity,
        )
        windows = calculate_windows(config.total_units, temporal_config)

        # Build splits for each window
        window_configs = {}
        for window in windows:
            # Determine ranges
            valid_range = None
            if window.has_valid and window.valid_start and window.valid_end:
                valid_range = (window.valid_start, window.valid_end)

            if config.granularity == "hour":
                temp_dir, splits = builder.build_temporal_splits(
                    train_hours=(window.train_start, window.train_end),
                    valid_hours=valid_range,
                    test_hours=(window.test_start, window.test_end),
                    temp_prefix=f"window{window.window_idx + 1}",
                )
            else:
                temp_dir, splits = builder.build_temporal_splits(
                    train_days=(window.train_start, window.train_end),
                    valid_days=valid_range,
                    test_days=(window.test_start, window.test_end),
                    temp_prefix=f"window{window.window_idx + 1}",
                )

            # Build window info dict for each seed's task
            num_windows = len(windows)
            for seed in config.seeds:
                task_id = f"w{window.window_idx + 1}_s{seed}"

                window_info = {
                    "window_number": window.window_idx + 1,
                    "total_windows": num_windows,
                    "granularity": config.granularity,
                    "window_size": config.window_size,
                    "window_stride": config.window_stride,
                    "window_ratio": config.window_ratio,
                    "start_unit": window.start_unit,
                    "end_unit": window.end_unit,
                    "train_range": f"{window.train_start}-{window.train_end}",
                    "train_units": temporal_config.train_units,
                    "test_range": f"{window.test_start}-{window.test_end}",
                    "test_units": temporal_config.test_units,
                    "has_validation": window.has_valid,
                    "benchmark_filename": splits["benchmark_filename"],
                    "temp_prefix": splits["temp_prefix"],
                    "temp_dir": str(temp_dir),
                }

                if window.has_valid:
                    window_info["valid_range"] = f"{window.valid_start}-{window.valid_end}"
                    window_info["valid_units"] = temporal_config.valid_units
                else:
                    window_info["validation_type"] = "dummy"

                window_configs[task_id] = window_info

        self._window_configs = window_configs

        # Persist configs to JSON file for array jobs
        if save_to_file:
            self._save_window_configs_to_file(experiment_id, config, tasks, window_configs)

        logger.info(f"Prepared {len(windows)} temporal windows")
        return window_configs

    def _save_window_configs_to_file(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        tasks: List[TaskInfo],
        window_configs: Dict[str, Dict],
    ) -> Path:
        """
        Save window configs to JSON file for array job consumption.
        """
        import json

        scripts_dir = Path("output/slurm_scripts")
        config_file = scripts_dir / experiment_id / "window_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Build task configs keyed by task_id for easy lookup
        task_configs = {}
        for task in tasks:
            task_configs[task.task_id] = {
                "task_id": task.task_id,
                "seed": task.seed,
                "window_idx": task.window_idx,
                "window_config": window_configs.get(task.task_id, {}),
            }

        with open(config_file, "w") as f:
            json.dump(task_configs, f, indent=2)

        logger.info(f"Saved window configs to {config_file}")
        return config_file

    def submit_all(
        self,
        experiment_id: str,
        use_array_jobs: bool = True,
        max_parallel: int = 10,
        skip_prepared: bool = True,
        prep_job_id: Optional[str] = None,
        warmup: bool = False,
    ) -> SubmissionStats:
        """
        Submit all pending tasks to Slurm.
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        stats = SubmissionStats()

        # Filter to pending tasks
        pending_tasks = [t for t in tasks if t.state == JobState.PENDING]

        if skip_prepared:
            # Skip any that are already submitted, running, or completed
            skip_states = [JobState.SUBMITTED, JobState.RUNNING, JobState.COMPLETED]
            skipped = [t for t in tasks if t.state in skip_states]
            stats.skipped = len(skipped)
            logger.info(f"Skipping {stats.skipped} already processed tasks")

        if not pending_tasks:
            logger.info("No pending tasks to submit")
            return stats

        # Build dependency list
        dependency_job_ids = [prep_job_id] if prep_job_id else None

        # If no prep job provided, prepare datasets locally (for local runs)
        if not prep_job_id and not self._window_configs:
            logger.info("No prep job dependency - preparing datasets locally")
            self._window_configs = self.prepare_temporal_datasets(experiment_id)

        if use_array_jobs:
            if prep_job_id:
                # Generate placeholder configs - actual data prepared by prep job
                all_task_configs = [
                    {"task_id": t.task_id, "window_idx": t.window_idx, "seed": t.seed} for t in pending_tasks
                ]
            else:
                all_task_configs = [self._window_configs[t.task_id] for t in pending_tasks]

            job_id = self.job_manager.submit_array_job(
                experiment_id,
                pending_tasks,
                all_task_configs,
                max_parallel=max_parallel,
                dependency_job_ids=dependency_job_ids,
                warmup=warmup,
            )

            if job_id:
                stats.submitted += len(pending_tasks)
                stats.job_ids.append(job_id)
            else:
                stats.failed += len(pending_tasks)
        else:
            # Submit individual jobs
            for task in pending_tasks:
                if not prep_job_id and task.task_id not in self._window_configs:
                    logger.warning(f"No window config for task {task.task_id}")
                    stats.failed += 1
                    continue

                # For jobs with prep dependency, use minimal config
                if prep_job_id:
                    window_config = {"task_id": task.task_id, "window_idx": task.window_idx, "seed": task.seed}
                else:
                    window_config = self._window_configs[task.task_id]

                job_id = self.job_manager.submit_task(
                    experiment_id,
                    task,
                    window_config,
                    dependency_job_ids=dependency_job_ids,
                )

                if job_id:
                    stats.submitted += 1
                    stats.job_ids.append(job_id)
                else:
                    stats.failed += 1

        logger.info(
            f"Submission complete: {stats.submitted} submitted, " f"{stats.skipped} skipped, {stats.failed} failed"
        )
        return stats

    def submit_tasks(
        self,
        experiment_id: str,
        task_ids: List[str],
    ) -> SubmissionStats:
        """Submit specific tasks by their IDs."""
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        stats = SubmissionStats()

        task_map = {t.task_id: t for t in tasks}

        if not self._window_configs:
            self._window_configs = self.prepare_temporal_datasets(experiment_id)

        for task_id in task_ids:
            if task_id not in task_map:
                logger.warning(f"Task {task_id} not found")
                stats.failed += 1
                continue

            task = task_map[task_id]

            if task.state == JobState.COMPLETED:
                logger.info(f"Task {task_id} already completed, skipping")
                stats.skipped += 1
                continue

            if task.state in [JobState.SUBMITTED, JobState.RUNNING]:
                logger.info(f"Task {task_id} already {task.state.value}, skipping")
                stats.skipped += 1
                continue

            # Reset state if needed
            if task.state in [JobState.FAILED, JobState.CANCELLED, JobState.PAUSED]:
                self.state_manager.update_task(
                    experiment_id,
                    task_id,
                    state=JobState.PENDING,
                )

            if task_id not in self._window_configs:
                logger.warning(f"No window config for task {task_id}")
                stats.failed += 1
                continue

            job_id = self.job_manager.submit_task(
                experiment_id,
                task,
                self._window_configs[task_id],
            )

            if job_id:
                stats.submitted += 1
                stats.job_ids.append(job_id)
            else:
                stats.failed += 1

        return stats

    def pause_experiment(self, experiment_id: str) -> int:
        """
        Pause an experiment - mark pending tasks as paused.
        Running tasks will complete but no new ones will start.

        Returns:
            Number of tasks paused
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        paused_count = 0

        for task in tasks:
            if task.state == JobState.PENDING:
                self.state_manager.update_task(
                    experiment_id,
                    task.task_id,
                    state=JobState.PAUSED,
                )
                paused_count += 1

        logger.info(f"Paused {paused_count} pending tasks")
        return paused_count

    def resume_experiment(self, experiment_id: str) -> int:
        """
        Resume a paused experiment - mark paused tasks as pending.

        Returns:
            Number of tasks resumed
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        resumed_count = 0

        for task in tasks:
            if task.state == JobState.PAUSED:
                self.state_manager.update_task(
                    experiment_id,
                    task.task_id,
                    state=JobState.PENDING,
                )
                resumed_count += 1

        logger.info(f"Resumed {resumed_count} paused tasks")
        return resumed_count

    def retry_failed(
        self,
        experiment_id: str,
        max_retries: int = 3,
        prep_job_id: Optional[str] = None,
    ) -> SubmissionStats:
        """
        Retry all failed/cancelled tasks.

        Args:
            experiment_id: Experiment identifier
            max_retries: Maximum retry attempts per task
            prep_job_id: Job ID of prep job to wait for (dependency)

        Returns:
            SubmissionStats for retry attempt
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        stats = SubmissionStats()

        # Only prepare locally if no prep job dependency
        if not prep_job_id and not self._window_configs:
            self._window_configs = self.prepare_temporal_datasets(experiment_id)

        failed_tasks = [
            t
            for t in tasks
            if t.state in [JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT] and t.retry_count < max_retries
        ]

        if not failed_tasks:
            logger.info("No failed tasks eligible for retry")
            return stats

        # Build dependency list
        dependency_job_ids = [prep_job_id] if prep_job_id else None

        job_ids = self.job_manager.resubmit_failed_tasks(
            experiment_id,
            self._window_configs if not prep_job_id else {},
            max_retries=max_retries,
            dependency_job_ids=dependency_job_ids,
        )

        stats.submitted = len(job_ids)
        stats.job_ids = job_ids
        stats.failed = len(failed_tasks) - len(job_ids)

        logger.info(f"Retried {stats.submitted} failed tasks")
        return stats

    def cancel_all(self, experiment_id: str) -> int:
        """Cancel all running/submitted jobs."""
        return self.job_manager.cancel_experiment(experiment_id)

    def get_status(self, experiment_id: str, update_from_slurm: bool = True) -> Dict[str, Any]:
        """
        Get current experiment status.

        Args:
            experiment_id: Experiment identifier
            update_from_slurm: Whether to query Slurm for latest status

        Returns:
            Status dictionary with progress and task details
        """
        if update_from_slurm:
            self.job_manager.update_all_job_statuses(experiment_id)

        config, progress, tasks = self.state_manager.load_experiment(experiment_id)

        # Group tasks by state
        by_state = {}
        for state in JobState:
            state_tasks = [t for t in tasks if t.state == state]
            if state_tasks:
                by_state[state.value] = [
                    {
                        "task_id": t.task_id,
                        "window": t.window_idx + 1,
                        "seed": t.seed,
                        "slurm_job_id": t.slurm_job_id,
                        "retry_count": t.retry_count,
                    }
                    for t in state_tasks
                ]

        return {
            "experiment_id": experiment_id,
            "model": config.model,
            "dataset": config.dataset,
            "description": config.description,
            "state": progress.state.value,
            "progress": {
                "total": progress.total_tasks,
                "completed": progress.completed,
                "running": progress.running,
                "submitted": progress.submitted,
                "pending": progress.pending,
                "failed": progress.failed,
                "cancelled": progress.cancelled,
                "paused": progress.paused,
            },
            "timing": {
                "created": progress.created_at,
                "started": progress.started_at,
                "updated": progress.updated_at,
                "completed": progress.completed_at,
            },
            "tasks_by_state": by_state,
        }

    def print_status(self, experiment_id: str, verbose: bool = False):
        """Print human-readable status."""
        status = self.get_status(experiment_id)

        print("\n" + "=" * 70)
        print(f"Experiment: {status['experiment_id']}")
        print(f"Model: {status['model']} | Dataset: {status['dataset']}")
        if status["description"]:
            print(f"Description: {status['description']}")
        print(f"State: {status['state'].upper()}")
        print("-" * 70)

        p = status["progress"]
        total = p["total"]
        print(f"Progress: {p['completed']}/{total} completed ({100*p['completed']/total:.1f}%)")
        print(
            f"  ✓ Completed: {p['completed']} | "
            f"⟳ Running: {p['running']} | "
            f"◷ Submitted: {p['submitted']} | "
            f"○ Pending: {p['pending']}"
        )
        print(f"  ✗ Failed: {p['failed']} | " f"⊘ Cancelled: {p['cancelled']} | " f"⏸ Paused: {p['paused']}")

        if verbose and status["tasks_by_state"]:
            print("-" * 70)
            for state, task_list in status["tasks_by_state"].items():
                if state in ["failed", "cancelled", "running"]:
                    print(f"\n{state.upper()} tasks:")
                    for t in task_list:
                        print(
                            f"  - {t['task_id']} (window {t['window']}, seed {t['seed']}, "
                            f"job {t['slurm_job_id']}, retries {t['retry_count']})"
                        )

        print("=" * 70 + "\n")

    def monitor(
        self,
        experiment_id: str,
        interval: int = 60,
        timeout: Optional[int] = None,
    ):
        """
        Monitor experiment progress until completion.

        Args:
            experiment_id: Experiment identifier
            interval: Seconds between status checks
            timeout: Maximum seconds to monitor (None for no limit)
        """
        start_time = time.time()

        while True:
            self.print_status(experiment_id)

            config, progress, tasks = self.state_manager.load_experiment(experiment_id)

            if progress.state in [ExperimentState.COMPLETED, ExperimentState.PARTIALLY_COMPLETED]:
                print(f"Experiment {progress.state.value}")
                break

            if timeout and (time.time() - start_time) > timeout:
                print(f"Monitoring timeout after {timeout}s")
                break

            print(f"Next update in {interval}s (Ctrl+C to stop monitoring)")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
