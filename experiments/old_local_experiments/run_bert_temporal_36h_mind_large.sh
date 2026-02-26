#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_bert_temporal_36h_mind_large"
DESCRIPTION="BERT temporal stability with 36h train, 12h test, 12h stride across 168h MIND Large dataset - bert_base_uncased, cls pooling, 128 max length"
MODEL="BERT"
DATASET="mind_large"
WINDOW_SIZE=48
WINDOW_RATIO="36:12"
TOTAL_UNITS=168
GRANULARITY="hour"
WINDOW_STRIDE=12
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"