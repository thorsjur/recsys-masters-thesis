import argparse
import glob
import json
import os
import platform
import socket
import sys
import time
from logging import getLogger
from importlib import import_module
import uuid

import psutil
import torch

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
        if hasattr(module, model_name):
            return getattr(module, model_name)
        for attr_name in dir(module):
            if attr_name.lower() == model_name.lower() and not attr_name.startswith("_"):
                return getattr(module, attr_name)
        raise AttributeError(f"No class matching '{model_name}' found in module")
    except (ImportError, AttributeError) as e:
        print(f"Error importing model '{model_name}': {e}")
        print(f"Model '{model_name}' not found in custom models, trying RecBole models...")

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

    raise ImportError(f"Could not find model '{model_name}' in custom models or RecBole")


def _find_config(search_dir, filename):
    """Find a config file by name within a directory tree."""
    pattern = os.path.join(search_dir, "**", filename)
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    direct = os.path.join(search_dir, filename)
    if os.path.exists(direct):
        return direct
    return None


def _get_base_config(dataset_config_path):
    """Read the ``_base`` key from a dataset config to resolve its parent."""
    import yaml

    with open(dataset_config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg and "_base" in cfg:
        base_name = cfg["_base"]
        base_path = os.path.join("configs", "datasets", f"{base_name}.yaml")
        if os.path.exists(base_path):
            return base_path
    return None


def _get_environment_info(device):
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["cuda_version"] = torch.version.cuda

    return info


def _get_process_stats():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()

    stats = {
        "process_rss_mb": mem.rss / (1024 ** 2),
        "process_vms_mb": mem.vms / (1024 ** 2),
    }

    io = None
    try:
        io = process.io_counters()
    except Exception:
        pass

    if io is not None:
        stats["process_read_mb"] = io.read_bytes / (1024 ** 2)
        stats["process_write_mb"] = io.write_bytes / (1024 ** 2)

    try:
        stats["cpu_percent"] = process.cpu_percent(interval=0.1)
    except Exception:
        pass

    return stats


def _get_gpu_stats():
    if not torch.cuda.is_available():
        return {}

    stats = {}

    try:
        stats["gpu_peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        stats["gpu_peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 ** 2)
        stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
    except Exception:
        pass

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        stats["gpu_util_percent"] = float(util.gpu)
        stats["gpu_mem_util_percent"] = float(util.memory)
        stats["gpu_used_memory_mb"] = mem.used / (1024 ** 2)
        stats["gpu_total_memory_mb"] = mem.total / (1024 ** 2)

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            stats["gpu_power_w"] = power / 1000.0
        except Exception:
            pass

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats["gpu_temp_c"] = float(temp)
        except Exception:
            pass

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    except Exception:
        pass

    return stats



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
        debug_mode=args.debug,
        log_dir="output/logs/recbole",
        log_prefix=f"{args.model.lower()}_{args.dataset}",
    )

    config_file_list = []
    model_name = args.model.split("_")[0]

    env_config = "configs/env.yaml"
    if os.path.exists(env_config) and env_config not in config_file_list:
        config_file_list.append(env_config)

    dataset_config = _find_config("configs/datasets", f"{args.dataset}.yaml")
    if dataset_config:
        base_config = _get_base_config(dataset_config)
        if base_config and base_config not in config_file_list:
            config_file_list.append(base_config)
        config_file_list.append(dataset_config)

    model_config = _find_config("configs/models", f"{model_name.lower()}.yaml")
    if model_config and model_config not in config_file_list:
        config_file_list.append(model_config)

    if args.config:
        config_file_list.extend(args.config)

    run_id = args.experiment_id or uuid.uuid4().hex[:8]

    config_dict = {
        "data_path": args.data_path,
        "checkpoint_dir": f"output/checkpoints/{run_id}/{uuid.uuid4().hex[:8]}",
    }

    if args.params:
        for param in args.params:
            key, value = param.split("=", 1)
            try:
                config_dict[key] = eval(value)
            except Exception:
                config_dict[key] = value

    model_class = get_model_class(model_name)

    config = CConfig(
        model=model_class,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    if "benchmark_filename" in config and config["benchmark_filename"] is not None:
        logger.info(f"benchmark_filename: {config['benchmark_filename']}")
        logger.info(f"Number of splits: {len(config['benchmark_filename'])}")  # type: ignore

    run_start = time.perf_counter()

    env_info = _get_environment_info(config["device"])
    logger.info(f"Environment info: {env_info}")

    dataset_create_start = time.perf_counter()
    dataset = create_dataset(config)
    dataset_create_time = time.perf_counter() - dataset_create_start
    logger.info(dataset)

    dataset_stats = collect_recbole_dataset_stats(dataset, config.final_config_dict)
    logger.info(format_dataset_stats_summary(dataset_stats))

    logger.info(f"Original dataset has item_feat: {dataset.item_feat is not None}")

    data_prep_start = time.perf_counter()
    train_data, valid_data, test_data = data_preparation(config, dataset)
    data_prep_time = time.perf_counter() - data_prep_start

    assert not isinstance(
        train_data, KnowledgeBasedDataLoader
    ), "Knowledge-based models are not currently supported"

    logger.info(f"train_data.dataset has item_feat: {train_data.dataset.item_feat is not None}")  # type: ignore

    model_init_start = time.perf_counter()
    model = model_class(config, train_data.dataset).to(config["device"])
    model_init_time = time.perf_counter() - model_init_start
    logger.info(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    trainer = Trainer(config, model)

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    pre_train_process_stats = _get_process_stats()

    train_start = time.perf_counter()

    assert (
        isinstance(config["epochs"], int) and config["epochs"] >= 0
    ), "Number of epochs must be a non-negative integer"

    valid_eval_time = None

    if config["epochs"] > 0:  # type: ignore
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
        training_time = time.perf_counter() - train_start

        logger.info("Evaluating on test set")
        test_eval_start = time.perf_counter()
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        test_eval_time = time.perf_counter() - test_eval_start
    else:
        logger.info("Skipping training (epochs=0, non-trainable model)")
        training_time = time.perf_counter() - train_start

        if valid_data:
            logger.info("Evaluating on validation set")
            valid_eval_start = time.perf_counter()
            best_valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)
            valid_eval_time = time.perf_counter() - valid_eval_start

            assert best_valid_result is not None, "Validation result should not be None"
            best_valid_score = best_valid_result.get(config["valid_metric"], 0.0)
        else:
            best_valid_result = None
            best_valid_score = 0.0

        logger.info("Evaluating on test set")
        test_eval_start = time.perf_counter()
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
        test_eval_time = time.perf_counter() - test_eval_start

    total_runtime = time.perf_counter() - run_start

    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_result}")

    post_run_process_stats = _get_process_stats()
    gpu_stats = _get_gpu_stats()

    runtime_info = {
        "dataset_creation_sec": dataset_create_time,
        "data_preparation_sec": data_prep_time,
        "model_init_sec": model_init_time,
        "training_sec": training_time,
        "valid_eval_sec": valid_eval_time,
        "test_eval_sec": test_eval_time,
        "total_sec": total_runtime,
    }

    model_info = {
        "trainable_params": trainable_params,
        "total_params": total_params,
    }

    additional_info = {
        "data_path": args.data_path,
        "config_files": config_file_list,
        "dataset_stats": dataset_stats,
        "runtime": runtime_info,
        "environment": env_info,
        "model_info": model_info,
        "process_stats_pre_train": pre_train_process_stats,
        "process_stats_post_run": post_run_process_stats,
        "gpu_stats": gpu_stats,
        "checkpoint_dir": config["checkpoint_dir"],
    }

    if args.experiment_id:
        additional_info["experiment_id"] = args.experiment_id
    if args.description:
        additional_info["description"] = args.description
    if args.window_info:
        additional_info["window_info"] = json.loads(args.window_info)

    results_logger = ResultsLogger()
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