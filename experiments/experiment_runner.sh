#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT_ID:?EXPERIMENT_ID is required}"
: "${DESCRIPTION:?DESCRIPTION is required}"
: "${MODEL:?MODEL is required}"
: "${DATASET:?DATASET is required}"

WINDOW_SIZE="${WINDOW_SIZE:-48}"
WINDOW_RATIO="${WINDOW_RATIO:-36:12}"
TOTAL_UNITS="${TOTAL_UNITS:-144}"
GRANULARITY="${GRANULARITY:-hour}"
WINDOW_STRIDE="${WINDOW_STRIDE:-12}"
RUNS="${RUNS:-3}"

echo "================================================================================"
echo "Starting Experiment: $EXPERIMENT_ID"
echo "Description: $DESCRIPTION"
echo "================================================================================"
echo "Model:       $MODEL"
echo "Dataset:     $DATASET"
echo "Window size: $WINDOW_SIZE"
echo "Window ratio: $WINDOW_RATIO"
echo "Total units: $TOTAL_UNITS $GRANULARITY"
echo "Stride:      $WINDOW_STRIDE"
echo "Runs:        $RUNS"
echo "================================================================================"

# Change to project root
cd "$(dirname "$0")/.."

python run_stability_test.py \
  --experiment C \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --window-size "$WINDOW_SIZE" \
  --window-ratio "$WINDOW_RATIO" \
  --total-units "$TOTAL_UNITS" \
  --granularity "$GRANULARITY" \
  --window-stride "$WINDOW_STRIDE" \
  --runs "$RUNS" \
  --experiment-id "$EXPERIMENT_ID" \
  --description "$DESCRIPTION"

echo "================================================================================"
echo "Experiment $EXPERIMENT_ID completed!"
echo "Results saved to: output/results/experiments.jsonl"
echo "Filter results with: grep \"$EXPERIMENT_ID\" output/results/experiments.jsonl"
echo "================================================================================"
