import sys
sys.path.insert(0, '.')

from run_recbole import main as run_recbole

if __name__ == "__main__":

    sys.argv = [
        'run_tfidf.py',
        '--model', 'TFIDF',
        '--dataset', 'mind_small',
        '--config', 'configs/mind_small.yaml', 'configs/tfidf.yaml'
    ] + sys.argv[1:]

    
    run_recbole()
