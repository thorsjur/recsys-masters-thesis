#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E07_fasttext_nrms_eb"
MODEL="NRMS_eb"
DATASET="ebnerd_tokenized"

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
    --memory "48G" \
    --description "E07 NRMS neural baseline on EB-NeRD, Danish FastText initialized embeddings" \
    --seeds "42,123,456,789,999" \
    --gpu-count 1 \
    --params "embedding_source=fasttext" "embedding_path=~/fasttext/cc.da.300.bin" "fasttext_cache=~/fasttext/cache/cc.da.300.bin.kv" "use_abstract=false"
