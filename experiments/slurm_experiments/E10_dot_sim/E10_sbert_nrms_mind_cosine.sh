#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E10_sbert_nrms_mind_cosine"
MODEL="NRMS_sbert_cosine"
DATASET="mind"

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
    --description "E10 NRMS on MIND, using SBERT to encode news content with cosine similarity" \
    --seeds "42,123,456,789,999" \
    --gpu-count 1 \
    --params "sentence_embedding_source=sbert" "sentence_embedding_model=sentence-transformers/distiluse-base-multilingual-cased-v2" "sentence_embedding_dim=512" "similarity=cosine"

