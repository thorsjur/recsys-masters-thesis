import sys
sys.path.insert(0, '.')

from run_recbole import main as run_recbole

if __name__ == "__main__":
    sys.argv = [
        'run_bert.py',
        '--model', 'BERT',
        '--dataset', 'mind_no_preprocessing',
        '--config', 'configs/mind_no_preprocessing.yaml', 'configs/bert.yaml'
    ] + sys.argv[1:]
    
    run_recbole()
