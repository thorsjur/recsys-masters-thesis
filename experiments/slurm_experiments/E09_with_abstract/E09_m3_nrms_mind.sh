#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E07_bge_m3_nrms_mind"
MODEL="NRMS_m3"
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
    --description "E07 NRMS on MIND, using BAAi's multilingual bge m3 to encode news content with abstract" \
    --seeds "42,123,456,789,999" \
    --gpu-count 1 \
    --params "sentence_embedding_model=BAAI/bge-m3" "sentence_embedding_dim=1024" "eval_batch_size=128" "use_abstract=true"
