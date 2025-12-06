#!/usr/bin/env python3
"""Run FastText model on MIND dataset."""

import sys
sys.path.insert(0, '.')

from run_recbole import main as run_recbole

if __name__ == "__main__":
    import sys
    sys.argv = [
        'run_fasttext.py',
        '--model', 'FastText',
        '--dataset', 'mind_small',
        '--config', 'configs/mind_small.yaml', 'configs/fasttext.yaml'
    ] + sys.argv[1:]
    
    run_recbole()
