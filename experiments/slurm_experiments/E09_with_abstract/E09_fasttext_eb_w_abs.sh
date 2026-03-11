#!/bin/bash
set -euo pipefail

ACCOUNT="${SLURM_ACCOUNT:-}"
if [ -z "$ACCOUNT" ]; then
    echo "SLURM_ACCOUNT not set"
    exit 1
fi

export EXPERIMENT_ID="E09_fasttext_eb_w_abs"
MODEL="FastText_abs"
DATASET="ebnerd_cleaned"

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
    --total-units 336 \
    --granularity hour \
    --window-stride 12 \
    --account "$ACCOUNT" \
    --partition "$PARTITION" \
    --time-limit "24:00:00" \
    --memory "64G" \
    --description "E09 FastText subword embedding baseline on EB-NeRD with abstract" \
    --seeds "42,123" \
    --params "embedding_path=~/fasttext/cc.da.300.bin" "fasttext_cache=~/fasttext/cache/cc.da.300.bin.kv" "use_abstract=true"
