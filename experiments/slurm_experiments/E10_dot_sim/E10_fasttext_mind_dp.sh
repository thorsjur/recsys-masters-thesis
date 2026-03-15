#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E10_fasttext_mind_dp"
MODEL="FastText_dp"
DATASET="mind_cleaned"

cd "$(dirname "$0")/../../.."

echo "Experiment: $EXPERIMENT_ID"
echo "Model: $MODEL"
echo "Dataset: $DATASET"

PARTITION="CPUQ"

python run_slurm_experiment.py create \
    --experiment-id "$EXPERIMENT_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --window-size 48 \
    --window-ratio "36:12" \
    --total-units 168 \
    --granularity hour \
    --window-stride 12 \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --time-limit "24:00:00" \
    --memory "24G" \
    --description "E10 FastText subword embedding baseline on MIND with dot similarity" \
    --seeds "42,123" \
    --params "embedding_path=~/fasttext/cc.en.300.bin" "fasttext_cache=~/fasttext/cache/cc.en.300.bin.kv" "similarity=dot"
