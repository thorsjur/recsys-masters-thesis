# TF-IDF Baseline for RecBole

## Overview

This directory contains a TF-IDF content-based baseline model for use with RecBole. The model uses TF-IDF vectorization of item text features (title and abstract) to compute item similarities and make recommendations.

## Model Description

The TF-IDF baseline (`models/tfidf.py`) works by:
1. Building a TF-IDF matrix from item text features (title and optionally abstract)
2. Computing item-to-item similarity using cosine similarity
3. Recommending items similar to those in the user's history

## Dataset Configuration

Two datasets are configured in `configs/`:

- **mind_small.yaml**: MIND Small dataset with preprocessing
- **mind_no_preprocessing.yaml**: MIND Small dataset without preprocessing

Both datasets use:
- Ratio-based splitting (80/10/10 train/valid/test)
- Temporal ordering (TO) by timestamp
- Full ranking evaluation mode
- Standard metrics: Recall, NDCG, Hit, Precision, MRR at k=[5, 10, 20, 50]

## Usage

### Running the TF-IDF Baseline

```bash
# Run on mind_small dataset
python run_tfidf.py --dataset mind_small

# Run on mind_no_preprocessing dataset
python run_tfidf.py --dataset mind_no_preprocessing

# Enable debug logging
python run_tfidf.py --dataset mind_small --debug
```

### Configuration Options

Edit `configs/tfidf.yaml` to adjust:
- `use_abstract`: Include abstract text in TF-IDF (default: true)
- `title_field`: Field name for title (default: 'title')
- `abstract_field`: Field name for abstract (default: 'abstract')

## Output

Logs are saved to `output/logs/recbole/tfidf_{dataset}_YYYYMMDD_HHMMSS.log`

Results include:
- Validation metrics during training
- Final test set performance
- Per-metric scores at different k values

## Requirements

The TF-IDF model requires:
- recbole
- scikit-learn
- torch
- numpy
