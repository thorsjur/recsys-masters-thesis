import json
import logging
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from filelock import FileLock

from slurm.slurm_constants import DEFAULT_CONDA_ENV

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    PAUSED = "paused"


class ExperimentState(str, Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"


@dataclass
class TaskInfo:
    """Information about a single experiment task."""

    task_id: str
    window_idx: int
    seed: int
    state: JobState = JobState.PENDING
    slurm_job_id: Optional[str] = None
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    output_file: Optional[str] = None
    error_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskInfo":
        """Create from dictionary."""
        data["state"] = JobState(data["state"])
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    experiment_id: str
    model: str
    dataset: str
    window_size: int
    total_units: int
    window_ratio: str
    window_stride: int
    granularity: str
    seeds: List[int]
    config_files: List[str] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    data_path: str = "data/atomic_files"
    description: Optional[str] = None

    # IDUN Slurm-specific configuration
    partition: str = "CPUQ"  # CPUQ, GPUQ, or short
    time_limit: str = "48:00:00"
    memory: str = "16G"
    cpus_per_task: int = 4
    ntasks_per_node: int = 1
    nodes: int = 1

    # GPU configuration
    gpu_count: int = 0
    gpu_type: Optional[str] = None  # p100, v100, a100, h100
    gpu_constraint: Optional[str] = None  # gpu16g, gpu32g, gpu40g, gpu80g, sxm4

    # Account is REQUIRED on IDUN
    account: str = ""

    # Module loading
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = DEFAULT_CONDA_ENV

    # Email notifications
    mail_user: Optional[str] = None
    mail_type: str = "FAIL"  # NONE, BEGIN, END, FAIL, ALL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExperimentProgress:
    """Progress tracking for an experiment."""

    state: ExperimentState = ExperimentState.INITIALIZING
    total_tasks: int = 0
    pending: int = 0
    submitted: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    paused: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentProgress":
        """Create from dictionary."""
        data["state"] = ExperimentState(data["state"])
        return cls(**data)

    def update_counts(self, tasks: List[TaskInfo]):
        """Update counts from task list."""
        self.total_tasks = len(tasks)
        self.pending = sum(1 for t in tasks if t.state == JobState.PENDING)
        self.submitted = sum(1 for t in tasks if t.state == JobState.SUBMITTED)
        self.running = sum(1 for t in tasks if t.state == JobState.RUNNING)
        self.completed = sum(1 for t in tasks if t.state == JobState.COMPLETED)
        self.failed = sum(1 for t in tasks if t.state == JobState.FAILED)
        self.cancelled = sum(1 for t in tasks if t.state == JobState.CANCELLED)
        self.paused = sum(1 for t in tasks if t.state == JobState.PAUSED)
        self.updated_at = datetime.now().isoformat()

        # Update overall state
        if self.completed == self.total_tasks:
            self.state = ExperimentState.COMPLETED
            if not self.completed_at:
                self.completed_at = datetime.now().isoformat()
        elif self.failed > 0 and self.pending == 0 and self.submitted == 0 and self.running == 0:
            self.state = ExperimentState.PARTIALLY_COMPLETED
        elif self.paused == self.total_tasks - self.completed:
            self.state = ExperimentState.PAUSED
        elif self.running > 0 or self.submitted > 0:
            self.state = ExperimentState.RUNNING
            if not self.started_at:
                self.started_at = datetime.now().isoformat()


class StateManager:
    def __init__(self, state_dir: str = "output/slurm_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_path(self, experiment_id: str) -> Path:
        """Get path to state file for an experiment."""
        return self.state_dir / f"{experiment_id}.json"

    def _get_lock_path(self, experiment_id: str) -> Path:
        """Get path to lock file for an experiment."""
        return self.state_dir / f"{experiment_id}.lock"

    def experiment_exists(self, experiment_id: str) -> bool:
        """Check if an experiment state exists."""
        return self._get_state_path(experiment_id).exists()

    def list_experiments(self) -> List[str]:
        """List all experiment IDs with state files."""
        return [p.stem for p in self.state_dir.glob("*.json")]

    def create_experiment(
        self,
        config: ExperimentConfig,
        tasks: List[TaskInfo],
    ) -> None:
        """Create a new experiment state file."""
        state_path = self._get_state_path(config.experiment_id)
        lock_path = self._get_lock_path(config.experiment_id)

        if state_path.exists():
            raise ValueError(
                f"Experiment '{config.experiment_id}' already exists. "
                f"Use --resume to continue or --force to overwrite."
            )

        progress = ExperimentProgress(total_tasks=len(tasks))
        progress.update_counts(tasks)

        state_data = {
            "config": config.to_dict(),
            "progress": progress.to_dict(),
            "tasks": [t.to_dict() for t in tasks],
        }

        with FileLock(lock_path):
            with open(state_path, "w") as f:
                json.dump(state_data, f, indent=2)

        logger.info(f"Created experiment state: {state_path}")

    def load_experiment(self, experiment_id: str) -> tuple[ExperimentConfig, ExperimentProgress, List[TaskInfo]]:
        """Load experiment state from file."""
        state_path = self._get_state_path(experiment_id)
        lock_path = self._get_lock_path(experiment_id)

        if not state_path.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found")

        with FileLock(lock_path):
            with open(state_path, "r") as f:
                state_data = json.load(f)

        config = ExperimentConfig.from_dict(state_data["config"])
        progress = ExperimentProgress.from_dict(state_data["progress"])
        tasks = [TaskInfo.from_dict(t) for t in state_data["tasks"]]

        return config, progress, tasks

    def save_experiment(
        self,
        config: ExperimentConfig,
        progress: ExperimentProgress,
        tasks: List[TaskInfo],
    ) -> None:
        """Save experiment state to file."""
        state_path = self._get_state_path(config.experiment_id)
        lock_path = self._get_lock_path(config.experiment_id)

        progress.update_counts(tasks)

        state_data = {
            "config": config.to_dict(),
            "progress": progress.to_dict(),
            "tasks": [t.to_dict() for t in tasks],
        }

        with FileLock(lock_path):
            # Write to temp file first, then rename to ensure atomic operation
            temp_path = state_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(state_data, f, indent=2)
            shutil.move(temp_path, state_path)

    def update_task(
        self,
        experiment_id: str,
        task_id: str,
        **updates,
    ) -> None:
        """Update a single task's state."""
        config, progress, tasks = self.load_experiment(experiment_id)

        for task in tasks:
            if task.task_id == task_id:
                for key, value in updates.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                break
        else:
            raise ValueError(f"Task '{task_id}' not found")

        self.save_experiment(config, progress, tasks)

    def get_tasks_by_state(self, experiment_id: str, states: List[JobState]) -> List[TaskInfo]:
        """Get tasks in specific states."""
        _, _, tasks = self.load_experiment(experiment_id)
        return [t for t in tasks if t.state in states]

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete experiment state file."""
        state_path = self._get_state_path(experiment_id)
        lock_path = self._get_lock_path(experiment_id)

        if state_path.exists():
            state_path.unlink()
            logger.info(f"Deleted experiment state: {state_path}")

        if lock_path.exists():
            lock_path.unlink()

    def backup_experiment(self, experiment_id: str) -> Path:
        """Create a backup of experiment state."""
        state_path = self._get_state_path(experiment_id)
        if not state_path.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.state_dir / "backups" / f"{experiment_id}_{timestamp}.json"
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(state_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
