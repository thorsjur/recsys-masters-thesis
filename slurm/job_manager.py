"""
Slurm job submission and management.

Handles interaction with the Slurm workload manager:
- Job submission
- Status monitoring
- Job cancellation
- Output retrieval
"""

import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from slurm.state import (
    JobState,
    TaskInfo,
    ExperimentConfig,
    ExperimentProgress,
    StateManager,
)

logger = logging.getLogger(__name__)


class SlurmJobManager:
    """
    Manages Slurm job submission and monitoring.
    """

    # Map Slurm job states to our JobState enum
    SLURM_STATE_MAP = {
        "PENDING": JobState.SUBMITTED,
        "RUNNING": JobState.RUNNING,
        "COMPLETED": JobState.COMPLETED,
        "FAILED": JobState.FAILED,
        "CANCELLED": JobState.CANCELLED,
        "TIMEOUT": JobState.TIMEOUT,
        "NODE_FAIL": JobState.FAILED,
        "PREEMPTED": JobState.CANCELLED,
        "OUT_OF_MEMORY": JobState.FAILED,
    }

    def __init__(
        self,
        state_manager: StateManager,
        scripts_dir: str = "output/slurm_scripts",
        logs_dir: str = "output/slurm_logs",
        dry_run: bool = False,
    ):
        self.state_manager = state_manager
        self.scripts_dir = Path(scripts_dir)
        self.logs_dir = Path(logs_dir)
        self.dry_run = dry_run
        self.templates_dir = Path(__file__).parent / "templates"

        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _load_template(self, template_name: str) -> Template:
        """Load a template file from the templates directory."""
        template_path = self.templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return Template(template_path.read_text())

    def generate_job_script(
        self,
        config: ExperimentConfig,
        task: TaskInfo,
        window_config: Dict,
    ) -> Path:
        """Generate a Slurm job script for a single task (IDUN optimized)."""
        script_name = f"{config.experiment_id}_{task.task_id}.sh"
        script_path = self.scripts_dir / config.experiment_id / script_name
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the Python command
        python_cmd = self._build_python_command(config, task, window_config)

        # Build Slurm directives
        slurm_directives = self._build_slurm_directives(config, task)

        # Build module loads and environment setup
        env_setup = self._build_environment_setup(config)

        # Load and render template
        template = self._load_template("single_job.sh.template")
        script_content = template.substitute(
            experiment_id=config.experiment_id,
            task_id=task.task_id,
            model=config.model,
            memory=config.memory,
            slurm_directives=slurm_directives,
            env_setup=env_setup,
            python_cmd=python_cmd,
        )
        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)
        logger.debug(f"Generated script: {script_path}")
        return script_path

    def _build_slurm_directives(self, config: ExperimentConfig, task: TaskInfo) -> str:
        """Build Slurm SBATCH directives for IDUN cluster."""
        log_dir = self.logs_dir / config.experiment_id
        log_dir.mkdir(parents=True, exist_ok=True)

        output_file = log_dir / f"{task.task_id}_%j.out"
        error_file = log_dir / f"{task.task_id}_%j.err"

        # Account is REQUIRED on IDUN
        if not config.account:
            raise ValueError(
                "Slurm account is required on IDUN. "
                "Use --account to specify your allocation account. "
                "Run 'sacctmgr show assoc format=Account%15,User,QOS | grep $USER' to find your accounts."
            )

        directives = [
            f"#SBATCH --account={config.account}",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --time={config.time_limit}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem={config.memory}",
            f"#SBATCH --output={output_file}",
            f"#SBATCH --error={error_file}",
        ]

        # GPU configuration for IDUN (uses --gres=gpu:TYPE:COUNT)
        if config.gpu_count > 0:
            if config.gpu_type:
                # Specific GPU type requested (p100, v100, a100, h100)
                directives.append(f"#SBATCH --gres=gpu:{config.gpu_type}:{config.gpu_count}")
            else:
                # Any GPU
                directives.append(f"#SBATCH --gres=gpu:{config.gpu_count}")

            # GPU memory/feature constraint (gpu16g, gpu32g, gpu40g, gpu80g, sxm4)
            if config.gpu_constraint:
                directives.append(f"#SBATCH --constraint={config.gpu_constraint}")

        # Email notifications
        if config.mail_user:
            directives.append(f"#SBATCH --mail-user={config.mail_user}")
            directives.append(f"#SBATCH --mail-type={config.mail_type}")

        return "\n".join(directives)

    def _build_environment_setup(self, config: ExperimentConfig) -> str:
        """Build environment setup commands for IDUN cluster.

        Note: These strings are inserted directly into the final script,
        so use regular $ (not $$) for bash variables.
        """
        lines = [
            "# Working directory",
            "WORKDIR=${SLURM_SUBMIT_DIR}",
            "cd ${WORKDIR}",
            "",
            'echo "Running from directory: $SLURM_SUBMIT_DIR"',
            'echo "Job name: $SLURM_JOB_NAME"',
            'echo "Job ID: $SLURM_JOB_ID"',
            'echo "Nodes: $SLURM_JOB_NODELIST"',
            'echo "CPUs per task: $SLURM_CPUS_PER_TASK"',
            "",
        ]

        # Module loading - always purge first on IDUN
        lines.extend(
            [
                "# Load modules",
                "module purge",
            ]
        )

        if config.modules:
            for module in config.modules:
                lines.append(f"module load {module}")
        else:
            # Default Python module for IDUN if none specified
            lines.append("# No modules specified - using conda environment")

        lines.extend(["module list", ""])

        # Activate conda environment
        if config.conda_env:
            lines.extend(
                [
                    "# Activate conda environment",
                    "source $(conda info --base)/etc/profile.d/conda.sh",
                    f"conda activate {config.conda_env}",
                    'echo "Python: $(which python)"',
                    'echo "Python version: $(python --version)"',
                    "",
                ]
            )

        return "\n".join(lines)

    def _build_prep_slurm_directives(self, config: ExperimentConfig) -> str:
        """Build Slurm SBATCH directives for prep job (CPU-only, shorter time)."""
        log_dir = self.logs_dir / config.experiment_id
        log_dir.mkdir(parents=True, exist_ok=True)

        output_file = log_dir / "prep_%j.out"
        error_file = log_dir / "prep_%j.err"

        if not config.account:
            raise ValueError("Slurm account is required on IDUN.")

        # Prep job uses CPU partition with modest resources
        directives = [
            f"#SBATCH --account={config.account}",
            "#SBATCH --partition=CPUQ",
            "#SBATCH --time=01:00:00",  # 1 hour should be plenty
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=16G",
            f"#SBATCH --output={output_file}",
            f"#SBATCH --error={error_file}",
        ]

        if config.mail_user:
            directives.append(f"#SBATCH --mail-user={config.mail_user}")
            directives.append("#SBATCH --mail-type=FAIL")

        return "\n".join(directives)

    def generate_prep_job_script(self, config: ExperimentConfig) -> Path:
        """Generate a Slurm job script for dataset preparation."""
        script_name = f"{config.experiment_id}_prep.sh"
        script_path = self.scripts_dir / config.experiment_id / script_name
        script_path.parent.mkdir(parents=True, exist_ok=True)

        slurm_directives = self._build_prep_slurm_directives(config)
        env_setup = self._build_environment_setup(config)

        template = self._load_template("prep_job.sh.template")
        script_content = template.substitute(
            experiment_id=config.experiment_id,
            slurm_directives=slurm_directives,
            env_setup=env_setup,
        )

        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)
        logger.info(f"Generated prep script: {script_path}")
        return script_path

    def submit_prep_job(self, experiment_id: str) -> Optional[str]:
        """
        Submit the dataset preparation job.

        Returns:
            Slurm job ID if successful, None otherwise
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        script_path = self.generate_prep_job_script(config)

        sbatch_cmd = ["sbatch", str(script_path)]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would submit prep job: {' '.join(sbatch_cmd)}")
            return "dry_run_prep"

        try:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                logger.info(f"Submitted prep job as {job_id}")
                return job_id
            else:
                logger.error(f"Could not parse job ID from: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit prep job: {e.stderr}")
            return None

    def _build_python_command(
        self,
        config: ExperimentConfig,
        task: TaskInfo,
        window_config: Dict,
    ) -> str:
        """Build the Python command to run."""
        import json

        cmd_parts = [
            "python run_recbole.py",
            f"--model {config.model}",
            f"--dataset {config.dataset}",
            f"--data_path {config.data_path}",
            f"--experiment-id {config.experiment_id}",
        ]

        if config.description:
            cmd_parts.append(f'--description "{config.description}"')

        if config.config_files:
            cmd_parts.append(f"--config {' '.join(config.config_files)}")

        # Add window info as JSON
        window_info_json = json.dumps(window_config)
        cmd_parts.append(f"--window-info '{window_info_json}'")

        # Add seed and other params
        params = [f"seed={task.seed}"]
        if config.params:
            params.extend(config.params)

        # Add benchmark filename from window config
        if "benchmark_filename" in window_config:
            params.append(f"benchmark_filename={window_config['benchmark_filename']}")

        cmd_parts.append(f"--params {' '.join(params)}")

        return " \\\n    ".join(cmd_parts)

    def submit_task(
        self,
        experiment_id: str,
        task: TaskInfo,
        window_config: Dict,
        dependency_job_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Submit a single task to Slurm.

        Returns:
            Slurm job ID if successful, None otherwise
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)

        # Generate job script
        script_path = self.generate_job_script(config, task, window_config)

        # Build sbatch command
        sbatch_cmd = ["sbatch"]

        if dependency_job_ids:
            dep_str = ":".join(dependency_job_ids)
            sbatch_cmd.extend(["--dependency", f"afterany:{dep_str}"])

        sbatch_cmd.append(str(script_path))

        if self.dry_run:
            logger.info(f"[DRY RUN] Would submit: {' '.join(sbatch_cmd)}")
            return f"dry_run_{task.task_id}"

        try:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse job ID from output (e.g., "Submitted batch job 12345")
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                logger.info(f"Submitted task {task.task_id} as job {job_id}")

                # Update task state
                self.state_manager.update_task(
                    experiment_id,
                    task.task_id,
                    state=JobState.SUBMITTED,
                    slurm_job_id=job_id,
                    submit_time=datetime.now().isoformat(),
                    output_file=str(self.logs_dir / config.experiment_id / f"{task.task_id}_{job_id}.out"),
                    error_file=str(self.logs_dir / config.experiment_id / f"{task.task_id}_{job_id}.err"),
                )
                return job_id
            else:
                logger.error(f"Could not parse job ID from: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit task {task.task_id}: {e.stderr}")
            self.state_manager.update_task(
                experiment_id,
                task.task_id,
                state=JobState.FAILED,
                error_message=f"Submission failed: {e.stderr}",
            )
            return None

    def submit_array_job(
        self,
        experiment_id: str,
        tasks: List[TaskInfo],
        window_configs: List[Dict],
        max_parallel: int = 10,
        dependency_job_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Submit multiple tasks as a Slurm array job.

        Returns:
            Slurm job ID if successful, None otherwise
        """
        if len(tasks) != len(window_configs):
            raise ValueError("Number of tasks must match number of window configs")

        config, progress, all_tasks = self.state_manager.load_experiment(experiment_id)

        # Generate array job script
        script_path = self._generate_array_script(config, tasks, window_configs)

        array_spec = f"0-{len(tasks)-1}"
        if max_parallel and max_parallel < len(tasks):
            array_spec += f"%{max_parallel}"

        sbatch_cmd = ["sbatch", f"--array={array_spec}"]

        # Add dependency if specified
        if dependency_job_ids:
            dep_str = ":".join(dependency_job_ids)
            sbatch_cmd.extend(["--dependency", f"afterok:{dep_str}"])

        sbatch_cmd.append(str(script_path))

        if self.dry_run:
            logger.info(f"[DRY RUN] Would submit array job: {' '.join(sbatch_cmd)}")
            return f"dry_run_array_{experiment_id}"

        try:
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                logger.info(f"Submitted array job {job_id} with {len(tasks)} tasks")

                # Update all task states
                for i, task in enumerate(tasks):
                    self.state_manager.update_task(
                        experiment_id,
                        task.task_id,
                        state=JobState.SUBMITTED,
                        slurm_job_id=f"{job_id}_{i}",
                        submit_time=datetime.now().isoformat(),
                    )
                return job_id
            else:
                logger.error(f"Could not parse job ID from: {result.stdout}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit array job: {e.stderr}")
            return None

    def _generate_array_script(
        self,
        config: ExperimentConfig,
        tasks: List[TaskInfo],
        window_configs: List[Dict],
    ) -> Path:
        """Generate a Slurm array job script for IDUN cluster."""
        import json

        script_name = f"{config.experiment_id}_array.sh"
        script_path = self.scripts_dir / config.experiment_id / script_name
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # Create task config file (JSON array)
        task_configs = []
        for task, window_config in zip(tasks, window_configs):
            task_configs.append(
                {
                    "task_id": task.task_id,
                    "seed": task.seed,
                    "window_idx": task.window_idx,
                    "window_config": window_config,
                }
            )

        config_file = script_path.with_suffix(".json")
        with open(config_file, "w") as f:
            json.dump(task_configs, f, indent=2)

        log_dir = self.logs_dir / config.experiment_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # Account is REQUIRED on IDUN
        if not config.account:
            raise ValueError("Slurm account is required on IDUN. " "Use --account to specify your allocation account.")

        slurm_directives = [
            f"#SBATCH --account={config.account}",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --time={config.time_limit}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem={config.memory}",
            f"#SBATCH --job-name={config.experiment_id}_array",
            f"#SBATCH --output={log_dir}/%x_%A_%a.out",
            f"#SBATCH --error={log_dir}/%x_%A_%a.err",
        ]

        # GPU configuration for IDUN
        if config.gpu_count > 0:
            if config.gpu_type:
                slurm_directives.append(f"#SBATCH --gres=gpu:{config.gpu_type}:{config.gpu_count}")
            else:
                slurm_directives.append(f"#SBATCH --gres=gpu:{config.gpu_count}")
            if config.gpu_constraint:
                slurm_directives.append(f"#SBATCH --constraint={config.gpu_constraint}")

        # Email notifications
        if config.mail_user:
            slurm_directives.append(f"#SBATCH --mail-user={config.mail_user}")
            slurm_directives.append(f"#SBATCH --mail-type={config.mail_type}")

        env_setup = self._build_environment_setup(config)

        # Build optional arguments
        description_arg = f'--description "{config.description}"' if config.description else ""
        config_arg = f"--config {' '.join(config.config_files)}" if config.config_files else ""
        extra_params = " ".join(config.params) if config.params else ""

        # Load and render template
        template = self._load_template("array_job.sh.template")
        script_content = template.substitute(
            experiment_id=config.experiment_id,
            model=config.model,
            num_tasks=len(tasks),
            slurm_directives="\n".join(slurm_directives),
            env_setup=env_setup,
            config_file=config_file,
            dataset=config.dataset,
            data_path=config.data_path,
            description_arg=description_arg,
            config_arg=config_arg,
            extra_params=extra_params,
        )
        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)
        logger.info(f"Generated array script: {script_path}")
        return script_path

    def get_job_status(self, job_id: str) -> Optional[JobState]:
        """Query Slurm for job status."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
                check=True,
            )

            states = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
            if states:
                # Get the most recent state (last in list for array jobs)
                slurm_state = states[-1].split("|")[0] if "|" in states[-1] else states[-1]
                # Handle states like "CANCELLED by 12345"
                slurm_state = slurm_state.split()[0]
                return self.SLURM_STATE_MAP.get(slurm_state, JobState.FAILED)

            return None

        except subprocess.CalledProcessError:
            return None

    def update_all_job_statuses(self, experiment_id: str) -> Dict[str, JobState]:
        """Update status of all submitted/running jobs."""
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        updated = {}

        for task in tasks:
            if task.state in [JobState.SUBMITTED, JobState.RUNNING] and task.slurm_job_id:
                new_state = self.get_job_status(task.slurm_job_id)
                if new_state and new_state != task.state:
                    self.state_manager.update_task(
                        experiment_id,
                        task.task_id,
                        state=new_state,
                        end_time=(
                            datetime.now().isoformat()
                            if new_state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT]
                            else None
                        ),
                    )
                    updated[task.task_id] = new_state
                    logger.info(f"Task {task.task_id}: {task.state.value} -> {new_state.value}")

        return updated

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a Slurm job."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would cancel job {job_id}")
            return True

        try:
            subprocess.run(
                ["scancel", job_id],
                check=True,
                capture_output=True,
            )
            logger.info(f"Cancelled job {job_id}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e.stderr}")
            return False

    def cancel_experiment(self, experiment_id: str) -> int:
        """Cancel all running jobs for an experiment."""
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        cancelled = 0

        for task in tasks:
            if task.state in [JobState.SUBMITTED, JobState.RUNNING] and task.slurm_job_id:
                if self.cancel_job(task.slurm_job_id):
                    self.state_manager.update_task(
                        experiment_id,
                        task.task_id,
                        state=JobState.CANCELLED,
                        end_time=datetime.now().isoformat(),
                    )
                    cancelled += 1

        logger.info(f"Cancelled {cancelled} jobs for experiment {experiment_id}")
        return cancelled

    def get_job_output(self, experiment_id: str, task_id: str) -> Tuple[str, str]:
        """Get stdout and stderr from a completed job."""
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)

        task = next((t for t in tasks if t.task_id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        stdout = ""
        stderr = ""

        if task.output_file and Path(task.output_file).exists():
            stdout = Path(task.output_file).read_text()

        if task.error_file and Path(task.error_file).exists():
            stderr = Path(task.error_file).read_text()

        return stdout, stderr

    def resubmit_failed_tasks(
        self,
        experiment_id: str,
        window_configs: Dict[str, Dict],
        max_retries: int = 3,
        dependency_job_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Resubmit failed/cancelled tasks.

        Args:
            experiment_id: Experiment identifier
            window_configs: Window configurations (can be empty if using prep job)
            max_retries: Maximum retry attempts per task
            dependency_job_ids: Job IDs to wait for before running

        Returns:
            List of newly submitted job IDs
        """
        config, progress, tasks = self.state_manager.load_experiment(experiment_id)
        submitted_jobs = []

        failed_tasks = [
            t
            for t in tasks
            if t.state in [JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT] and t.retry_count < max_retries
        ]

        for task in failed_tasks:
            # Use placeholder config if prep job handles preparation
            if window_configs and task.task_id not in window_configs:
                logger.warning(f"No window config for task {task.task_id}, skipping")
                continue

            task_config = window_configs.get(
                task.task_id,
                {
                    "task_id": task.task_id,
                    "window_idx": task.window_idx,
                    "seed": task.seed,
                },
            )

            # Increment retry count
            self.state_manager.update_task(
                experiment_id,
                task.task_id,
                state=JobState.PENDING,
                retry_count=task.retry_count + 1,
                slurm_job_id=None,
                error_message=None,
            )

            # Reload task with updated retry count
            _, _, updated_tasks = self.state_manager.load_experiment(experiment_id)
            updated_task = next(t for t in updated_tasks if t.task_id == task.task_id)

            job_id = self.submit_task(
                experiment_id,
                updated_task,
                task_config,
                dependency_job_ids=dependency_job_ids,
            )
            if job_id:
                submitted_jobs.append(job_id)

        return submitted_jobs
