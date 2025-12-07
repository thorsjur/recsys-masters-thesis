#!/bin/bash

# Experiment: Random Temporal Stability - 36h training windows
# ID: exp_random_temporal_36h
# Description: Evaluate random baseline stability across temporal windows in news recommendation
# Date: 2025-12-06
# Configuration:
#   - Model: Random
#   - Dataset: mind_small (144 hours total)
#   - Training window: 36 hours
#   - Test window: 12 hours
#   - Window stride: 12 hours (50% overlap)
#   - Expected windows: 9
#   - Runs per window: 3
#   - Total runs: 27

set -e  # Exit on error

EXPERIMENT_ID="exp_random_temporal_36h"
DESCRIPTION="Random baseline temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset"
MODEL="Random"
DATASET="mind_small"
WINDOW_SIZE=48        # 36h train + 12h test
WINDOW_RATIO="36:12"  # 36h train, 12h test
TOTAL_UNITS=144       # 144 hours in MIND small dataset
GRANULARITY="hour"
WINDOW_STRIDE=12      # 12h stride = 50% overlap
RUNS=3                # 3 runs per window for statistical stability

echo "================================================================================"
echo "Starting Experiment: $EXPERIMENT_ID"
echo "Description: $DESCRIPTION"
echo "================================================================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Window configuration:"
echo "  - Training: 36 hours"
echo "  - Testing: 12 hours"
echo "  - Stride: 12 hours (50% overlap)"
echo "  - Total dataset: 144 hours"
echo "  - Expected windows: 9"
echo "  - Runs per window: $RUNS"
echo "================================================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Run the temporal stability experiment
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
