import argparse
import sys
from logging import getLogger
from importlib import import_module

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from util.logging_config import setup_logging


def get_model_class(model_name):
    """Dynamically import model class."""
    try:
        module = import_module(f'models.{model_name.lower()}')
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass
    
    try:
        module = import_module('recbole.model.general_recommender')
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass
    
    try:
        module = import_module('recbole.model.sequential_recommender')
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass
    
    try:
        module = import_module('recbole.model.context_aware_recommender')
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass
    
    try:
        module = import_module('recbole.model.knowledge_aware_recommender')
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass
    
    raise ImportError(f"Could not find model '{model_name}' in custom models or RecBole")


def main():
    """Run RecBole experiments with specified model and dataset."""
    parser = argparse.ArgumentParser(description="Run RecBole model on dataset.")
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Model name (e.g., TFIDF, BPR, Pop, ItemKNN)"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mind_small',
        help="Dataset name"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        nargs='+',
        default=None,
        help="Additional config files"
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='datasets/atomic_files',
        help="Path to dataset directory"
    )
    
    parser.add_argument('--debug', action='store_true', help="Enable debug logging")
    
    parser.add_argument(
        '--params',
        type=str,
        nargs='+',
        default=None,
        help="Override config parameters (e.g., --params learning_rate=0.001 epochs=100)"
    )
    
    args = parser.parse_args()
    
    setup_logging(
        debug_mode=args.debug,
        log_dir='output/logs/recbole',
        log_prefix=f'{args.model.lower()}_{args.dataset}'
    )
    
    config_file_list = []
    if args.config:
        config_file_list.extend(args.config)
    
    config_dict = {
        'data_path': args.data_path,
    }
    
    if args.params:
        for param in args.params:
            key, value = param.split('=')
            try:
                config_dict[key] = eval(value)
            except:
                config_dict[key] = value
    
    model_class = get_model_class(args.model)
    
    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict
    )
    
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    
    logger.info(config)
    
    dataset = create_dataset(config)
    logger.info(dataset)
    
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    trainer = Trainer(config, model)
    
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=True
    )
    
    test_result = trainer.evaluate(test_data, show_progress=True)
    
    logger.info(f'Best valid result: {best_valid_result}')
    logger.info(f'Test result: {test_result}')
    
    return test_result


if __name__ == "__main__":
    main()
