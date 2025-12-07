import argparse
import time
from logging import getLogger
from importlib import import_module
import importlib.util
import sys
from pathlib import Path

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader

from util.logging_config import setup_logging
from util.results_logger import ResultsLogger


def get_model_class(model_name):
    """Dynamically import model class."""
    try:
        module = import_module(f'models.{model_name.lower()}')
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass
    
    model_file_map = {
        'Pop': 'pop.py',
        'Random': 'random.py',
    }
    
    if model_name in model_file_map:
        try:
            import recbole
            recbole_path = Path(recbole.__file__).parent
            model_file = recbole_path / 'model' / 'general_recommender' / model_file_map[model_name]
            
            if model_file.exists():
                # Load module directly from file
                spec = importlib.util.spec_from_file_location(f"recbole_model_{model_name}", model_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"recbole_model_{model_name}"] = module
                    spec.loader.exec_module(module)
                    return getattr(module, model_name)
        except Exception as e:
            pass
    
    # Fallback to generic imports
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
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help="Unique identifier for this experimental run"
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help="Human-readable description of this experimental run"
    )
    
    parser.add_argument(
        '--window-info',
        type=str,
        default=None,
        help="JSON string with window/split information for temporal experiments"
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
    
    # Debug: Check benchmark_filename configuration
    if 'benchmark_filename' in config and config['benchmark_filename'] is not None:
        logger.info(f"benchmark_filename: {config['benchmark_filename']}")
        logger.info(f"Number of splits: {len(config['benchmark_filename'])}")
    
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # Collect dataset statistics
    dataset_stats = {
        "num_users": int(dataset.user_num),
        "num_items": int(dataset.item_num),
        "num_interactions": int(dataset.inter_num),
        "sparsity": float(1 - dataset.inter_num / (dataset.user_num * dataset.item_num)),
        "avg_interactions_per_user": float(dataset.inter_num / dataset.user_num),
        "avg_interactions_per_item": float(dataset.inter_num / dataset.item_num),
        "has_item_features": dataset.item_feat is not None,
    }
    
    # Debug: Check if item features are loaded
    logger.info(f"Original dataset has item_feat: {dataset.item_feat is not None}")
    
    # data_preparation should now always return 3 datasets
    train_data, valid_data, test_data = data_preparation(config, dataset)

    assert not isinstance(train_data, KnowledgeBasedDataLoader), \
        "Knowledge-based models are not currently supported"
    
    # Debug: Check train_data.dataset
    logger.info(f"train_data.dataset has item_feat: {train_data.dataset.item_feat is not None}")
    
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    trainer = Trainer(config, model)
    
    start_time = time.time()

    assert isinstance(config['epochs'], int) and config['epochs'] >= 0, "Number of epochs must be a non-negative integer" # type: ignore
    
    if  config['epochs'] > 0: # type: ignore
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=True
        )
        training_time = time.time() - start_time
        test_result = trainer.evaluate(test_data, show_progress=True)
    else:
        logger.info("Skipping training (epochs=0, non-trainable model)")
        training_time = time.time() - start_time
        
        if valid_data:
            logger.info("Evaluating on validation set")
            best_valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)
            assert best_valid_result is not None, "Validation result should not be None"

            best_valid_score = best_valid_result.get(config['valid_metric'], 0.0)
        else:
            best_valid_result = None
            best_valid_score = 0.0
        
        logger.info("Evaluating on test set")
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    
    logger.info(f'Best valid result: {best_valid_result}')
    logger.info(f'Test result: {test_result}')
    
    results_logger = ResultsLogger()
    additional_info = {
        'data_path': args.data_path,
        'config_files': config_file_list,
        'dataset_stats': dataset_stats,
    }
    if args.experiment_id:
        additional_info['experiment_id'] = args.experiment_id
    if args.description:
        additional_info['description'] = args.description
    if args.window_info:
        import json
        additional_info['window_info'] = json.loads(args.window_info)
    
    results_logger.log_experiment(
        model=args.model,
        dataset=args.dataset,
        config=config.final_config_dict,
        valid_results=best_valid_result,
        test_results=test_result,
        training_time=training_time,
        additional_info=additional_info
    )
    
    logger.info(f'Results saved to {results_logger.results_path}')
    
    return test_result


if __name__ == "__main__":
    main()
