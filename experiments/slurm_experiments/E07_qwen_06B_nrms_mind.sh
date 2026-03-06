#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E07_qwen_0.6b_nrms_mind"
MODEL="NRMS_qwen_0.6B"
DATASET="mind"

cd "$(dirname "$0")/../.."

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
    --description "E07 NRMS on MIND, using Qwen 0.6B to encode news content" \
    --seeds "42,123,456,789,999" \
    --gpu-count 1 \
    --params "sentence_embedding_model=Qwen/Qwen3-Embedding-0.6B" "sentence_embedding_model_kwargs={'device_map':'auto'}" "sentence_embedding_tokenizer_kwargs={'padding_side':'left'}" "sentence_embedding_dim=1024" "eval_batch_size=128"
