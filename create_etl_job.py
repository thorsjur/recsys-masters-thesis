#!/usr/bin/env python3
"""
Generate Slurm job script for ETL pipeline.

Creates a .sh script for running run_etl.py on an HPC cluster.
The generated script can be submitted manually with sbatch.

Usage:
    python create_etl_job.py \\
        --config mind_small_no_preprocessing \\
        --temporal-hours 168 \\
        --account <your_account> \\
        --output etl_mind_small.sh

    # Then submit manually:
    sbatch etl_mind_small.sh
"""

import argparse
from pathlib import Path
from string import Template
from typing import Optional

from dataset_registry import get_available_datasets
from slurm.slurm_constants import DEFAULT_CONDA_ENV, DEFAULT_CONDA_MODULE


AVAILABLE_CONFIGS = get_available_datasets()
TEMPLATE_DIR = Path(__file__).parent / "slurm" / "templates"


def _load_template(name: str) -> Template:
    """Load a template file from the templates directory."""
    template_path = TEMPLATE_DIR / name
    with open(template_path, "r") as f:
        return Template(f.read())


def create_etl_job_script(
    config: str,
    output_path: str,
    temporal_hours: Optional[int] = None,
    temporal_days: Optional[int] = None,
    account: str = "",
    partition: str = "CPUQ",
    time_limit: str = "04:00:00",
    memory: str = "32G",
    cpus: int = 4,
    conda_env: Optional[str] = DEFAULT_CONDA_ENV,
    modules: Optional[list] = None,
    debug: bool = False,
) -> Path:
    """Generate a Slurm job script for ETL."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    job_name = f"etl_{config}"
    log_dir = Path("output/slurm_logs/etl")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build ETL command
    etl_cmd_parts = [
        "python run_etl.py",
        f"--config {config}",
    ]

    if temporal_hours:
        etl_cmd_parts.append(f"--temporal-hours {temporal_hours}")
    elif temporal_days:
        etl_cmd_parts.append(f"--temporal-days {temporal_days}")

    if debug:
        etl_cmd_parts.append("--debug")

    etl_cmd = " \\\n    ".join(etl_cmd_parts)

    # Build SLURM directives
    slurm_lines = [
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --output={log_dir}/{job_name}_%j.out",
        f"#SBATCH --error={log_dir}/{job_name}_%j.err",
    ]
    slurm_directives = "\n".join(slurm_lines)

    # Build environment setup
    env_lines = ["# Load modules", "module purge"]

    # Always load default conda module
    all_modules = [DEFAULT_CONDA_MODULE]
    if modules:
        all_modules.extend(modules)

    for mod in all_modules:
        env_lines.append(f"module load {mod}")
    env_lines.append("module list")

    # Conda activation
    if conda_env:
        env_lines.extend(
            [
                "",
                "# Activate conda environment",
                "eval \"$(conda shell.bash hook)\"",
                f"conda activate {conda_env}",
                'echo "Python: $(which python)"',
                'echo "Python version: $(python --version)"',
            ]
        )

    env_setup = "\n".join(env_lines)

    # Load and fill template
    template = _load_template("etl_job.sh.template")
    script_content = template.substitute(
        config=config,
        job_name=job_name,
        slurm_directives=slurm_directives,
        env_setup=env_setup,
        etl_cmd=etl_cmd,
    )

    with open(output_file, "w") as f:
        f.write(script_content)

    output_file.chmod(0o755)
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate Slurm job script for ETL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ETL arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=AVAILABLE_CONFIGS,
        help="Dataset configuration key",
    )
    parser.add_argument(
        "--temporal-hours",
        type=int,
        help="Generate hour-wise temporal splits",
    )
    parser.add_argument(
        "--temporal-days",
        type=int,
        help="Generate day-wise temporal splits",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging in ETL",
    )

    # Slurm arguments
    parser.add_argument(
        "--account",
        type=str,
        required=True,
        help="IDUN allocation account",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="CPUQ",
        help="Slurm partition (default: CPUQ)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="04:00:00",
        help="Time limit HH:MM:SS (default: 04:00:00)",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="32G",
        help="Memory allocation (default: 32G)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs (default: 4)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default=DEFAULT_CONDA_ENV,
        help="Conda environment to activate",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        help="Modules to load",
    )

    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output script path (default: output/slurm_scripts/etl_<config>.sh)",
    )

    args = parser.parse_args()

    # Validate temporal args
    if args.temporal_hours and args.temporal_days:
        parser.error("Cannot specify both --temporal-hours and --temporal-days")

    # Default output path
    if args.output is None:
        args.output = f"output/slurm_scripts/etl_{args.config}.sh"

    # Generate script
    output_path = create_etl_job_script(
        config=args.config,
        output_path=args.output,
        temporal_hours=args.temporal_hours,
        temporal_days=args.temporal_days,
        account=args.account,
        partition=args.partition,
        time_limit=args.time,
        memory=args.memory,
        cpus=args.cpus,
        conda_env=args.conda_env,
        modules=args.modules,
        debug=args.debug,
    )

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
