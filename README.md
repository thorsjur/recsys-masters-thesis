# Stability Experiments for News Recommendation

Repository for a master's thesis project on temporal stability in news recommendation. It contains data preparation pipelines for MIND and EB-NeRD, a model execution layer built around RecBole, and data-loading extensions, sliding-window experiment orchestration, and analysis utilities for completed experiments. Experiments are primarily run on a Slurm cluster, but the code is designed to be runnable locally as well.

## Overview

![Architecture overview](./docs/diagrams/recsys-masters-thesis-overview-light.svg#gh-light-mode-only)
![Architecture overview](./docs/diagrams/recsys-masters-thesis-overview.svg#gh-dark-mode-only)

The workflow has four main parts:

- `run_etl.py` prepares raw datasets into RecBole-compatible atomic files.
- `configs/`, `models/`, `dataloaders/`, and `custom_datasets/` define the RecBole configuration and extensions used by the experiments.
- `experiments/` and `run_slurm_experiment.py` define and submit the main experiment runs.
- `data_analysis/` generates summaries, diagnostics, statistical models, and figures from completed runs.

The experiments use sliding time windows to compare recommendation quality across changing news periods. Each window is prepared from atomic files (file format as used by the RecBole framework), evaluated with one or more seeds, and logged with enough metadata to reproduce or analyze the run later.

## Setup

```bash
conda env create -f environment.yml
conda activate recsys_stability
```

Raw datasets are not included in the repository. Place them under:

- `data/MINDlarge`
- `data/EBNeRDlarge`

Available dataset keys are listed in `dataset_registry.py`.

## Prepare Data

```bash
python run_etl.py --config mind --temporal-hours 168
python run_etl.py --config ebnerd --temporal-hours 168
```

Current dataset variants:

- `mind`, `mind_cleaned`, `mind_tokenized`, `mind_cleaned_tokenized`
- `ebnerd`, `ebnerd_cleaned`, `ebnerd_tokenized`, `ebnerd_cleaned_tokenized`

Prepared files are written to `data/atomic_files/`.

## Run Experiments

The primary experiment definitions live in `experiments/`.

These scripts serve as the primary definition of the experiments. They set the experiment ID, dataset, method, temporal window, resources, and short description in the same place, allowing easily reproducible runs.

For Slurm runs, use the scripts in `experiments/slurm_experiments/`:

```bash
export SLURM_ACCOUNT=<account>
export RECSYS_CONDA_ENV=recsys_stability
bash experiments/slurm_experiments/E00_random_mind.sh
python run_slurm_experiment.py submit E00_random_mind
```

`RECSYS_CONDA_ENV`, `RECSYS_CONDA_MODULE`, and `SLURM_MAIL_USER` can be set to match the local cluster environment.

For a local run, call the stability runner directly:

```bash
python run_stability_test.py \
  --experiment A \
  --model RANDOM \
  --dataset mind \
  --window-size 48 \
  --window-ratio 36:12 \
  --total-units 168 \
  --granularity hour \
  --window-stride 12 \
  --seeds 42,123 \
  --experiment-id E00_random_mind \
  --description "Random baseline on MIND"
```

Results are appended to `output/results/experiments.jsonl`.

Each row contains the experiment ID, run metadata, window information, metrics, and timing information.

## Repository Layout

```text
configs/          RecBole and dataset configuration files
dataloaders/      Impression-aware data loading helpers
etl/              Dataset conversion, cleaning, sampling, and splitting
experiments/      Main experiment definitions and runners
models/           Recommendation models used by RecBole
slurm/            Slurm orchestration, state, and job templates
stability/        Sliding-window experiment protocol
data_analysis/    Dataset and result analysis utilities
output/results/   Committed experiment result logs used for analysis
```
