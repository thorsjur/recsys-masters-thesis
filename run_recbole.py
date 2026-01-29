import argparse
import os
import time
from logging import getLogger
from importlib import import_module
import uuid

from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader

from util.recbole.data_preparation import data_preparation, create_dataset
from util.recbole.config import CConfig
from util.logging_config import setup_logging
from util.results_logger import ResultsLogger
from util.dataset_analysis import collect_recbole_dataset_stats, format_dataset_stats_summary


def get_model_class(model_name):
    """Dynamically import model class."""

    model_lower = model_name.lower()

    try:
        module = import_module(f"models.{model_lower}")
        return getattr(module, model_name)
    except (ImportError, AttributeError):
        pass

    # Try to load from recbole models, this needs to be expanded if other models
    # are to be used.
    model_file_map = {
        "pop": ("recbole.model.general_recommender", "Pop"),
    }

    if model_lower in model_file_map:
        try:
            category, model = model_file_map[model_lower]
            module = import_module(category)
            return getattr(module, model)
        except Exception as e:
            print(e)
            pass

    raise ImportError(f"Could not find model '{model_name}' in custom models or RecBole")


def main():
    """Run RecBole experiments with specified model and dataset."""
    parser = argparse.ArgumentParser(description="Run RecBole model on dataset.")

    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., TFIDF, BPR, Pop, ItemKNN)")

    parser.add_argument("--dataset", type=str, default="mind_small", help="Dataset name")

    parser.add_argument("--config", type=str, nargs="+", default=None, help="Additional config files")

    parser.add_argument("--data_path", type=str, default="data/atomic_files", help="Path to dataset directory")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--params",
        type=str,
        nargs="+",
        default=None,
        help="Override config parameters (e.g., --params learning_rate=0.001 epochs=100)",
    )

    parser.add_argument("--experiment-id", type=str, default=None, help="Unique identifier for this experimental run")

    parser.add_argument(
        "--description", type=str, default=None, help="Human-readable description of this experimental run"
    )

    parser.add_argument(
        "--window-info",
        type=str,
        default=None,
        help="JSON string with window/split information for temporal experiments",
    )

    args = parser.parse_args()

    setup_logging(
        debug_mode=args.debug, log_dir="output/logs/recbole", log_prefix=f"{args.model.lower()}_{args.dataset}"
    )

    # Auto-include dataset and model configs if not already specified
    config_file_list = []
    if args.config:
        config_file_list.extend(args.config)

    dataset_config = f"configs/{args.dataset}.yaml"
    model_config = f"configs/{args.model.lower()}.yaml"

    # Add dataset config if exists and not already in list
    if os.path.exists(dataset_config) and dataset_config not in config_file_list:
        config_file_list.insert(0, dataset_config)

    # Add model config if exists and not already in list
    if os.path.exists(model_config) and model_config not in config_file_list:
        config_file_list.append(model_config)

    config_dict = {
        "data_path": args.data_path,
        
        # To prevent clashes when running in parallel, we use unique checkpoint dirs
        "checkpoint_dir": f"output/checkpoints/{args.experiment_id}/{uuid.uuid4().hex[:8]}",
    }

    if args.params:
        for param in args.params:
            key, value = param.split("=")
            try:
                config_dict[key] = eval(value)
            except:
                config_dict[key] = value

    model_class = get_model_class(args.model)

    config = CConfig(
        model=model_class, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # Debug: Check benchmark_filename configuration
    if "benchmark_filename" in config and config["benchmark_filename"] is not None:
        logger.info(f"benchmark_filename: {config['benchmark_filename']}")
        logger.info(f"Number of splits: {len(config['benchmark_filename'])}")  # type: ignore

    dataset = create_dataset(config)
    logger.info(dataset)

    # Collect dataset statistics
    dataset_stats = collect_recbole_dataset_stats(dataset, config.final_config_dict)
    logger.info(format_dataset_stats_summary(dataset_stats))

    # Debug: Check if item features are loaded
    logger.info(f"Original dataset has item_feat: {dataset.item_feat is not None}")

    # data_preparation should now always return 3 datasets
    train_data, valid_data, test_data = data_preparation(config, dataset)

    assert not isinstance(train_data, KnowledgeBasedDataLoader), "Knowledge-based models are not currently supported"

    # Debug: Check train_data.dataset
    logger.info(f"train_data.dataset has item_feat: {train_data.dataset.item_feat is not None}")  # type: ignore

    model = model_class(config, train_data.dataset).to(config["device"])
    logger.info(model)

    trainer = Trainer(config, model)

    start_time = time.time()

    assert isinstance(config["epochs"], int) and config["epochs"] >= 0, "Number of epochs must be a non-negative integer"  # type: ignore

    if config["epochs"] > 0:  # type: ignore
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
        training_time = time.time() - start_time
        test_result = trainer.evaluate(test_data, show_progress=True)
    else:
        logger.info("Skipping training (epochs=0, non-trainable model)")
        training_time = time.time() - start_time

        if valid_data:
            logger.info("Evaluating on validation set")
            best_valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)
            assert best_valid_result is not None, "Validation result should not be None"

            best_valid_score = best_valid_result.get(config["valid_metric"], 0.0)
        else:
            best_valid_result = None
            best_valid_score = 0.0

        logger.info("Evaluating on test set")
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_result}")

    results_logger = ResultsLogger()
    additional_info = {
        "data_path": args.data_path,
        "config_files": config_file_list,
        "dataset_stats": dataset_stats,
    }
    if args.experiment_id:
        additional_info["experiment_id"] = args.experiment_id
    if args.description:
        additional_info["description"] = args.description
    if args.window_info:
        import json

        additional_info["window_info"] = json.loads(args.window_info)

    results_logger.log_experiment(
        model=args.model,
        dataset=args.dataset,
        config=config.final_config_dict,
        valid_results=best_valid_result,
        test_results=test_result,
        training_time=training_time,
        additional_info=additional_info,
    )

    logger.info(f"Results saved to {results_logger.results_path}")

    return test_result


if __name__ == "__main__":
    main()
