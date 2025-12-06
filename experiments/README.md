# Experiments Tracking

This file tracks all experiments run in this project. Use this to find the correct results in `output/results/experiments.jsonl`.

## Experiment Log

| ID | Date | Description | Model | Dataset | Config | Status |
|----|------|-------------|-------|---------|--------|--------|
| exp_tfidf_temporal_36h | 2025-12-06 | TFIDF temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset | TFIDF | mind_small | 36h train, 12h test, 12h stride, 3 runs per window | Planned |
| exp_fasttext_temporal_36h | 2025-12-06 | FastText temporal stability with 36h train, 12h test, 12h stride across 144h MIND dataset | FastText | mind_small | 36h train, 12h test, 12h stride, 3 runs per window | Planned |

## Experiment Details

### exp_tfidf_temporal_36h
- **Purpose**: Evaluate TFIDF baseline stability across temporal windows in news recommendation
- **Configuration**:
  - Training window: 36 hours
  - Test window: 12 hours  
  - Window stride: 12 hours (50% overlap)
  - Total dataset: 144 hours (6 days of MIND small dataset)
  - Expected windows: 9 windows (sliding from hour 1-48 → 109-144 with 12h stride)
  - Runs per window: 3 (for statistical stability)
  - Total runs: 27
- **Script**: `run_tfidf_temporal_36h.sh`
- **Expected output**: Results grouped by experiment ID in `experiments.jsonl`
- **Rationale**: 
  - 36h training captures ~1.5 days of user behavior patterns
  - 12h test window represents half-day prediction horizon
  - 50% overlap allows dense temporal coverage
  - 3 runs measure algorithmic stability (TFIDF is deterministic but data splits may vary)

### exp_fasttext_temporal_36h
- **Purpose**: Evaluate FastText embedding-based model stability across temporal windows
- **Configuration**:
  - Training window: 36 hours
  - Test window: 12 hours
  - Window stride: 12 hours (50% overlap)
  - Total dataset: 144 hours (6 days of MIND small dataset)
  - Expected windows: 9 windows
  - Runs per window: 3 (different random seeds for embedding training)
  - Total runs: 27
  - Learning rate: 0.001
  - Embedding dimension: 64
  - Epochs: 20
- **Script**: `run_fasttext_temporal_36h.sh`
- **Expected output**: Results grouped by experiment ID in `experiments.jsonl`
- **Rationale**:
  - Same temporal configuration as TFIDF for direct comparison
  - 3 runs capture stochastic training variance
  - FastText embeddings adapt to temporal semantic shifts
  - Epochs=20 balances training time vs convergence

## Notes

- All experiments use the MIND small dataset (144 hours total)
- Window calculations: (144 - 36 - 12) / 12 + 1 = 9 windows
- Window ranges: 
  1. Train: 1-36, Test: 37-48
  2. Train: 13-48, Test: 49-60
  3. Train: 25-60, Test: 61-72
  4. Train: 37-72, Test: 73-84
  5. Train: 49-84, Test: 85-96
  6. Train: 61-96, Test: 97-108
  7. Train: 73-108, Test: 109-120
  8. Train: 85-120, Test: 121-132
  9. Train: 97-132, Test: 133-144
- Results can be filtered by experiment_id in experiments.jsonl
- Use `grep "exp_tfidf_temporal_36h" output/results/experiments.jsonl` to extract results

## Adding New Experiments

When adding new experiments:
1. Create a new shell script in this directory
2. Add entry to the table above with unique ID and description
3. Add detailed configuration in the "Experiment Details" section
4. Run the script and update status to "Running" or "Completed"
5. Document any issues or findings
