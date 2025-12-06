"""
Convenience wrapper for running TF-IDF model.
"""
import sys
from run_recbole import main as run_recbole

if __name__ == "__main__":
    sys.argv.insert(1, '--model')
    sys.argv.insert(2, 'TFIDF')
    
    if '--config' not in sys.argv:
        dataset_idx = sys.argv.index('--dataset') + 1 if '--dataset' in sys.argv else None
        dataset = sys.argv[dataset_idx] if dataset_idx else 'mind_small'
        sys.argv.extend(['--config', f'configs/{dataset}.yaml', 'configs/tfidf.yaml'])
    
    run_recbole()
