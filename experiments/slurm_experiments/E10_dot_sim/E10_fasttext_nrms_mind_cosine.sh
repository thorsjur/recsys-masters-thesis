#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E10_fasttext_nrms_mind_cosine"
MODEL="NRMS_ft_cosine"
DATASET="mind_tokenized"

cd "$(dirname "$0")/../../.."

echo "Experiment: $EXPERIMENT_ID"
echo "Model: $MODEL"
echo "Dataset: $DATASET"

PARTITION="GPUQ"

python run_slurm_experiment.py create \
    --experiment-id "$EXPERIMENT_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --window-size 48 \
    --window-ratio "30:6:12" \
    --total-units 168 \
    --granularity hour \
    --window-stride 12 \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --time-limit "24:00:00" \
    --memory "32G" \
    --description "E10 NRMS neural baseline on MIND, FastText initialized embeddings (instead of GloVe) with cosine similarity" \
    --seeds "42,123,456,789,999" \
    --gpu-count 1 \
    --params "embedding_source=fasttext" "embedding_path=~/fasttext/cc.en.300.bin" "fasttext_cache=~/fasttext/cache/cc.en.300.bin.kv" "similarity=cosine"
