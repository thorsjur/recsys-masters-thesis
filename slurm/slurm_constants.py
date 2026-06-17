import os


# Defaults for Slurm-based runs. Environment variables keep local cluster
# details out of the source tree while preserving convenient command-line
# defaults for repeated experiments.
DEFAULT_CONDA_ENV = os.environ.get("RECSYS_CONDA_ENV", "recsys_stability")
DEFAULT_CONDA_MODULE = os.environ.get("RECSYS_CONDA_MODULE", "Miniconda3/24.7.1-0")
DEFAULT_EMAIL = os.environ.get("SLURM_MAIL_USER") or None
