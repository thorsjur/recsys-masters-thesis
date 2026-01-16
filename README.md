# Stability Experiments for News Recommender Systems

Codebase for evaluating temporal stability of content-based news recommender models

> [!IMPORTANT]
> DISCLAIMER: For this codebase I have at times used GitHub Copilot (primarily with the *Claude Opus 4.5* model) for code generation and suggestions. While I have reviewed, modified, and tested all generated code, please be aware that some parts may contain inaccuracies or suboptimal implementations.

## How It Works

![Architecture overview](./docs/diagrams/recsys-masters-thesis-overview-light.svg#gh-light-mode-only)
![Architecture overview](./docs/diagrams/recsys-masters-thesis-overview.svg#gh-dark-mode-only)

> **Figure 1:** High-level architecture of the codebase.

1. **Data preparation** — `run_etl.py` uses the `loaders/` pipeline to convert raw MIND data into RecBole atomic files, optionally creating hour/day-level temporal splits. See [`dataset_registry.py`](./dataset_registry.py) for available datasets and configurations
2. **Model execution** — `run_recbole.py` loads a model from `models/`, builds item embeddings from news text, and evaluates on the prepared dataset using RecBole's evaluation framework
3. **Stability evaluation** — `run_stability_test.py` orchestrates sliding-window experiments: for each window, it builds a temporal benchmark file, runs the model with multiple seeds, and aggregates metrics



## Components

### Data Pipeline — `loaders/`

ETL framework for converting raw datasets to RecBole's atomic format.

| Component | Description |
|-----------|-------------|
| `base_loader.py` | Abstract loader with ETL pipeline orchestration |
| `converters/` | Dataset → atomic file converters |
| `processing/` | Text cleaning (SpaCy), recursive k-core pruning, etc. |
| `splitters/` | Global/temporal train-valid-test splitting, etc. |

![Loaders class diagram light](./docs/diagrams/loaders-class-diagram-light.svg#gh-light-mode-only)
![Loaders class diagram](./docs/diagrams/loaders-class-diagram.svg#gh-dark-mode-only)
> **Figure 2:** Class diagram of the data loading pipeline. Shows main classes and their relationships. Some information, such as methods and some attributes have been omitted for clarity.

### Models — `models/`

Content-based recommenders using text embeddings for item similarity.

| Model | Description |
|-------|-------------|
| `BERT` | Transformer embeddings (bert-base-uncased) |
| `FastText` | Subword embeddings |
| `TFIDF` | Sparse TF-IDF vectors |
| `Random` | Baseline (random recommendations) |
| `Pop` | Baseline (popularity-based) |


### Stability Framework — `stability/`

Temporal stability experiment protocol.

| File | Description |
|------|-------------|
| `experiment_temporal.py` | Sliding window evaluation loop, used by `run_stability_test.py` |
| `base.py` | Common experiment utilities |


## Usage

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate recsys
```

### 2. Data Preparation

```bash
# Download dataset to datasets/

# Run ETL with temporal splits (168 hours = 1 week)
python run_etl.py --config mind_small_no_preprocessing --temporal-hours 168
```

See `configs/` for dataset configurations.

### 3. Run Single Experiment

```bash
# Run BERT model example
python run_recbole.py --model BERT --dataset mind_small --config configs/bert.yaml

# With parameter overrides example
python run_recbole.py --model TFIDF --dataset mind_small --params seed=42 epochs=1
```

### 4. Run Stability Experiment (Local)

```bash
python run_stability_test.py \
    --model BERT \
    --dataset mind_no_preprocessing \
    --window-size 48 \
    --window-ratio 36:12 \
    --total-units 168 \
    --granularity hour \
    --window-stride 12 \
    --seeds 42,123,456
```

> [!NOTE]
> Even though the experiments can be ran locally, I define the individual experiments using shell files for better replicability and some extra info prints, with unique IDs. see `experiments/` for examples, and `experiments/experiment_runner.sh` for the common runner script.

## Configuration

Model configs in `configs/`:

```yaml
# configs/bert.yaml example
bert_model_name: 'bert-base-uncased'
bert_max_length: 128
bert_pooling: 'cls'
bert_batch_size: 32
bert_use_cache: true
```

---

## Output Structure

```
output/
├── slurm_scripts/          # Generated job scripts
├── slurm_logs/             # Stdout/stderr from jobs
├── experiments/            # Experimental definitions
└── results/                # Experiment results
    └── experiments.jsonl   # The full results log

log/                        # RecBole logs per model
```
