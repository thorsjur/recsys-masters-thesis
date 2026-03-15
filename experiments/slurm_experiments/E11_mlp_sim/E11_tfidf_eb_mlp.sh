#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

EXPERIMENT_ID="E11_tfidf_eb_mlp"
MODEL="TFIDF_mlp"
DATASET="ebnerd_cleaned"

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
    --total-units 336 \
    --granularity hour \
    --window-stride 12 \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --time-limit "24:00:00" \
    --memory "24G" \
    --description "E11 TF-IDF lexical similarity baseline on EB-NeRD. using MLP for similarity" \
    --seeds "42,123" \
    --params "similarity=mlp" "epochs=200" "train_batch_size=128" \
    --gpu-count 1
