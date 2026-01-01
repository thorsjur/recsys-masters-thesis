#!/bin/bash
set -euo pipefail

EXPERIMENT_ID="exp_bert_temporal_36h"
DESCRIPTION="BERT temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset - bert_base_uncased, cls pooling, 128 max length"
MODEL="BERT"
DATASET="mind_no_preprocessing"
WINDOW_SIZE=48
WINDOW_RATIO="36:12"
TOTAL_UNITS=144
GRANULARITY="hour"
WINDOW_STRIDE=12
RUNS=3

SCRIPT_DIR="$(dirname "$0")"
source "$SCRIPT_DIR/experiment_runner.sh"