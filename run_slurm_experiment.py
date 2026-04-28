import argparse
import json
import logging
import sys
from pathlib import Path

from slurm.slurm_constants import DEFAULT_CONDA_ENV, DEFAULT_EMAIL
from util.logging_config import setup_logging
from slurm.state import StateManager
from slurm.job_manager import SlurmJobManager
from slurm.orchestrator import ExperimentOrchestrator
from stability.base import parse_seeds


def setup_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run stability experiments on Slurm HPC clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # === CREATE command ===
    create_parser = subparsers.add_parser(
        "create",
        help="Create a new experiment",
    )
    _add_experiment_args(create_parser)
    _add_slurm_args(create_parser)
    create_parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs immediately after creation",
    )
    create_parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run first task of each window first to warm caches, then parallelize",
    )
    create_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing experiment with same ID",
    )

    # === SUBMIT command ===
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit pending tasks to Slurm",
    )
    submit_parser.add_argument("experiment_id", help="Experiment identifier")
    submit_parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of specific task IDs to submit",
    )
    submit_parser.add_argument(
        "--no-array",
        action="store_true",
        help="Submit individual jobs instead of array jobs",
    )
    submit_parser.add_argument(
        "--max-parallel",
        type=int,
        default=10,
        help="Maximum concurrent jobs for array jobs (default: 10)",
    )
    submit_parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run first task of each window first to warm caches, then parallelize",
    )

    # === STATUS command ===
    status_parser = subparsers.add_parser(
        "status",
        help="Show experiment status",
    )
    status_parser.add_argument("experiment_id", nargs="?", help="Experiment identifier (omit to list all)")
    status_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed task information",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output status as JSON",
    )
    status_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't query Slurm for latest status",
    )

    # === CANCEL command ===
    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel all running/submitted jobs",
    )
    cancel_parser.add_argument("experiment_id", help="Experiment identifier")

    # === LIST command ===
    list_parser = subparsers.add_parser(
        "list",
        help="List all experiments",
    )

    # === DELETE command ===
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete experiment state file",
    )
    delete_parser.add_argument("experiment_id", help="Experiment identifier")
    delete_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete without confirmation",
    )
    
    # === CLEANUP command ===
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Cleanup temporary split files",
    )
    cleanup_parser.add_argument("dataset", help="Dataset name")
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files to be deleted without deleting them",
    )
    
    cleanup_parser.add_argument(
        "--data-path",
        type=str,
        default="data/atomic_files",
        help="Path to dataset directory (default: 'data/atomic_files')",
    )
    
    cleanup_parser.add_argument(
        "--tmp-prefix",
        type=str,
        default="window",
        help="Prefix of temporary files to delete (default: 'window')",
    )
    
    # === OUTPUT command ===
    output_parser = subparsers.add_parser(
        "output",
        help="View output from a task",
    )
    output_parser.add_argument("experiment_id", help="Experiment identifier")
    output_parser.add_argument("task_id", help="Task identifier")
    output_parser.add_argument(
        "--stderr",
        action="store_true",
        help="Show stderr instead of stdout",
    )
    output_parser.add_argument(
        "-p", "--pager",
        action="store_true",
        help="Open output in less -R for interactive viewing",
    )
    output_parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Follow output (like tail -f)",
    )

    return parser


def _add_experiment_args(parser: argparse.ArgumentParser):
    """Add experiment configuration arguments."""
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Unique identifier for this experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., BERT, TFIDF, FastText)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mind_small",
        help="Dataset name (default: mind_small)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        required=True,
        help="Total window size in time units",
    )
    parser.add_argument(
        "--total-units",
        type=int,
        required=True,
        help="Total number of time units in dataset",
    )
    parser.add_argument(
        "--window-ratio",
        type=str,
        default="5:1:1",
        help="Train:valid:test ratio (default: '5:1:1', or '5:2' for no validation)",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        help="Time units to slide window forward (default: window-size)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["day", "hour"],
        default="day",
        help="Time granularity (default: 'day')",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds (default: auto-generated from --runs)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per window, each with a different seed (default: 3). "
             "Ignored if --seeds is provided.",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        help="Config files to use",
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="+",
        help="Additional parameters (e.g., 'learning_rate=0.001')",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/atomic_files",
        help="Path to dataset directory (default: 'data/atomic_files')",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Human-readable description of this experiment",
    )


def _add_slurm_args(parser: argparse.ArgumentParser):
    """Add IDUN Slurm configuration arguments."""
    # Required account
    parser.add_argument(
        "--account",
        type=str,
        required=True,
        help="IDUN allocation account (REQUIRED). Find with: sacctmgr show assoc format=Account%%15,User | grep $USER",
    )

    # Partition
    parser.add_argument(
        "--partition",
        type=str,
        choices=["CPUQ", "GPUQ", "short"],
        default="CPUQ",
        help="IDUN partition: CPUQ (CPU jobs), GPUQ (GPU jobs), short (20min test). Default: CPUQ",
    )

    # Time and resources
    parser.add_argument(
        "--time-limit",
        type=str,
        default="48:00:00",
        help="Job time limit HH:MM:SS (default: '48:00:00'). Max: CPUQ=30d, GPUQ=14d",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="16G",
        help="Memory per job (default: '16G')",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=4,
        help="CPUs per task (default: 4)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)",
    )
    parser.add_argument(
        "--ntasks-per-node",
        type=int,
        default=1,
        help="Tasks per node (default: 1)",
    )

    # GPU configuration (IDUN style with --gres)
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=0,
        help="Number of GPUs (default: 0). Auto-selects GPUQ partition if > 0",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        choices=["p100", "v100", "a100", "h100"],
        help="GPU type: p100, v100, a100, h100 (optional, any GPU if not specified)",
    )
    parser.add_argument(
        "--gpu-constraint",
        type=str,
        help="GPU constraint: gpu16g, gpu32g, gpu40g, gpu80g, sxm4 (optional)",
    )

    # Module loading
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        help="Modules to load (e.g., 'Python/3.10.4-GCCcore-11.3.0' 'CUDA/11.8')",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=DEFAULT_CONDA_ENV,
        help="Conda environment to activate",
    )

    # Email notifications
    parser.add_argument(
        "--mail-user",
        type=str,
        default=DEFAULT_EMAIL,
        help="Email address for job notifications",
    )
    parser.add_argument(
        "--mail-type",
        type=str,
        default="FAIL",
        choices=["NONE", "BEGIN", "END", "FAIL", "ALL"],
        help="When to send email notifications (default: FAIL)",
    )


def cmd_create(args, orchestrator: ExperimentOrchestrator):
    """Handle create command."""
    seeds = parse_seeds(args.seeds, runs=args.runs, start_seed=42)

    config, tasks = orchestrator.create_experiment(
        experiment_id=args.experiment_id,
        model=args.model,
        dataset=args.dataset,
        window_size=args.window_size,
        total_units=args.total_units,
        seeds=seeds,
        window_ratio=args.window_ratio,
        window_stride=args.window_stride,
        granularity=args.granularity,
        config_files=args.config,
        params=args.params,
        data_path=args.data_path,
        description=args.description,
        partition=args.partition,
        time_limit=args.time_limit,
        memory=args.memory,
        cpus_per_task=args.cpus_per_task,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        gpu_count=args.gpu_count,
        gpu_type=args.gpu_type,
        gpu_constraint=args.gpu_constraint,
        account=args.account,
        modules=args.modules,
        conda_env=args.conda_env,
        mail_user=args.mail_user,
        mail_type=args.mail_type,
        force=args.force,
    )

    print(f"Created experiment '{args.experiment_id}' with {len(tasks)} tasks")

    if args.submit:
        print("Submitting prep job and experiment tasks...")
        stats = orchestrator.submit_with_prep(
            args.experiment_id,
            warmup=args.warmup,
        )
        print(f"Submitted {stats.submitted} tasks (prep job + {stats.submitted} experiment jobs)")
        if stats.job_ids:
            print(f"Job IDs: {', '.join(stats.job_ids)}")


def cmd_submit(args, orchestrator: ExperimentOrchestrator):
    """Handle submit command."""
    if args.tasks:
        # Specific tasks - prepare locally since we're not doing full experiment
        print(f"Preparing temporal datasets for {args.experiment_id}...")
        orchestrator.prepare_temporal_datasets(args.experiment_id)

        task_ids = [t.strip() for t in args.tasks.split(",")]
        print(f"Submitting {len(task_ids)} specific tasks...")
        stats = orchestrator.submit_tasks(args.experiment_id, task_ids)
    else:
        # Full experiment - use prep job with dependencies
        print("Submitting prep job and experiment tasks...")
        stats = orchestrator.submit_with_prep(
            args.experiment_id,
            use_array_jobs=not args.no_array,
            max_parallel=args.max_parallel,
            warmup=args.warmup,
        )

    print(f"Submitted: {stats.submitted}, Skipped: {stats.skipped}, Failed: {stats.failed}")
    if stats.job_ids:
        print(f"Job IDs: {', '.join(stats.job_ids)}")


def cmd_status(args, orchestrator: ExperimentOrchestrator):
    """Handle status command."""
    state_manager = orchestrator.state_manager

    if not args.experiment_id:
        # List all experiments
        experiments = state_manager.list_experiments()
        if not experiments:
            print("No experiments found")
            return

        print(f"\nFound {len(experiments)} experiment(s):\n")
        for exp_id in experiments:
            try:
                config, progress, tasks = state_manager.load_experiment(exp_id)
                status = f"{progress.completed}/{progress.total_tasks} completed"
                print(f"  {exp_id}: {config.model}/{config.dataset} - {status} ({progress.state.value})")
            except Exception as e:
                print(f"  {exp_id}: Error loading - {e}")
        return

    if args.json:
        status = orchestrator.get_status(
            args.experiment_id,
            update_from_slurm=not args.no_update,
        )
        print(json.dumps(status, indent=2))
    else:
        orchestrator.print_status(
            args.experiment_id,
            verbose=args.verbose,
            update_from_slurm=not args.no_update,
        )


def cmd_cancel(args, orchestrator: ExperimentOrchestrator):
    """Handle cancel command."""
    count = orchestrator.cancel_all(args.experiment_id)
    print(f"Cancelled {count} jobs")


def cmd_list(args, orchestrator: ExperimentOrchestrator):
    """Handle list command."""
    experiments = orchestrator.state_manager.list_experiments()

    if not experiments:
        print("No experiments found")
        return

    print(f"\nFound {len(experiments)} experiment(s):\n")
    for exp_id in experiments:
        try:
            config, progress, tasks = orchestrator.state_manager.load_experiment(exp_id)
            print(f"  {exp_id}")
            print(f"    Model: {config.model} | Dataset: {config.dataset}")
            print(f"    Progress: {progress.completed}/{progress.total_tasks} ({progress.state.value})")
            print()
        except Exception as e:
            print(f"  {exp_id}: Error - {e}\n")


def cmd_delete(args, orchestrator: ExperimentOrchestrator):
    """Handle delete command."""
    if not args.force:
        response = input(f"Delete experiment '{args.experiment_id}'? [y/N] ")
        if response.lower() != "y":
            print("Cancelled")
            return

    orchestrator.state_manager.delete_experiment(args.experiment_id)
    print(f"Deleted experiment '{args.experiment_id}'")

def cmd_cleanup(args, orchestrator: ExperimentOrchestrator):
    """Handle cleanup command."""
    file_prefix = args.tmp_prefix
    assert file_prefix, "Temporary file prefix must be specified"
    
    file_name_prefix = args.dataset + "." + file_prefix
    dataset = args.dataset
    data_path = Path(args.data_path) / dataset
    temp_files = []
    
    # Find all files in data_path that start with file_name_prefix
    for f in data_path.iterdir():
        if f.is_file() and f.name.startswith(file_name_prefix):
            temp_files.append(str(f))

    if not temp_files:
        print(f"No temporary files found for dataset '{dataset}'")
        return

    print(f"Found {len(temp_files)} temporary files to delete:")
    for f in temp_files:
        print(f"  {f}")

    if args.dry_run:
        print("Dry run enabled, no files deleted")
        return

    for f in temp_files:
        try:
            Path(f).unlink()
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")

    print(f"Cleanup completed for dataset '{dataset}'")

def cmd_output(args, orchestrator: ExperimentOrchestrator):
    """Handle output command."""
    import subprocess
    
    # Get file paths directly from task info
    config, progress, tasks = orchestrator.state_manager.load_experiment(args.experiment_id)
    task = next((t for t in tasks if t.task_id == args.task_id), None)
    if not task:
        print(f"Task {args.task_id} not found")
        return

    file_path = task.error_file if args.stderr else task.output_file
    file_type = "stderr" if args.stderr else "stdout"

    if not file_path or not Path(file_path).exists():
        print(f"No {file_type} file found for task {args.task_id}")
        return

    if args.follow:
        subprocess.run(["tail", "-f", "-n", "50", file_path])
        return
    if args.pager:
        subprocess.run(["less", "-R", file_path])
    else:
        print(Path(file_path).read_text())


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(
        debug_mode=args.debug,
        log_dir="output/logs/slurm",
        log_prefix="slurm_experiment",
    )

    # Initialize components
    state_manager = StateManager()
    job_manager = SlurmJobManager(state_manager, dry_run=args.dry_run)
    orchestrator = ExperimentOrchestrator(
        state_manager=state_manager,
        job_manager=job_manager,
        dry_run=args.dry_run,
    )

    # Dispatch to command handler
    commands = {
        "create": cmd_create,
        "submit": cmd_submit,
        "status": cmd_status,
        "cancel": cmd_cancel,
        "list": cmd_list,
        "delete": cmd_delete,
        "cleanup": cmd_cleanup,
        "output": cmd_output,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args, orchestrator)
        except Exception as e:
            logging.error(f"Error: {e}")
            if args.debug:
                raise
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
