# Adapted from RecBole's data.utils to support custom dataloaders and datasets
import importlib
from logging import getLogger
from typing import Literal
from recbole.data.utils import data_preparation as recbole_data_preparation, create_samplers, create_dataset as recbole_create_dataset
from recbole.utils import set_color

from dataloaders.train_impression_data_loader import TrainImpressionDataLoader
from dataloaders.eval_impression_data_loader import EvalImpressionDataLoader, TestImpressionDataLoader, ValImpressionDataLoader

# ---
# DISCLAIMER:
# All of these functions are adapted from RecBole's data.utils to support custom dataloaders and datasets.

def create_dataset(config):
    """
    Adapted create_dataset function to support custom datasets.
    
    See RecBole's create_dataset for more details.
    """
    logger = getLogger()
    custom_dataset = config["custom_dataset"] or None
    
    c_dataset_registry = {
        "opt_sequential": "OptimizedSequentialDataset",
    }
    
    dataset_module = importlib.import_module("custom_datasets")
    if custom_dataset is None or custom_dataset not in c_dataset_registry or not hasattr(dataset_module, c_dataset_registry[custom_dataset]):
        logger.info("Using RecBole built-in dataset class.")
        return recbole_create_dataset(config)
    
    logger.info(f"Using custom dataset: {custom_dataset}")
    
    dataset_class = getattr(dataset_module, c_dataset_registry[custom_dataset])
    dataset = dataset_class(config)
    if config["save_dataset"]:
        raise NotImplementedError("Saving custom datasets is not supported yet")
    return dataset

def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    custom_dataloader = config["custom_dataloader"] or None
    if custom_dataloader is None:
      return recbole_data_preparation(config, dataset)
    
    logger = getLogger()
    logger.info(f"Using custom dataloader: {custom_dataloader}")
    logger.info("Building dataset")
    built_datasets = dataset.build()


    logger.info("Creating samplers and dataloaders...")
    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(
        config, dataset, built_datasets
    )

    train_data = get_dataloader(config, "train")(
        config, train_dataset, train_sampler, shuffle=config["shuffle"]
    )
    valid_data = get_dataloader(config, "valid")(
        config, valid_dataset, valid_sampler, shuffle=False
    )
    test_data = get_dataloader(config, "test")(
        config, test_dataset, test_sampler, shuffle=False
    )
    
    if config["save_dataloaders"]:
        raise NotImplementedError("Saving dataloaders is not supported yet")

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )

    register_table = {
        "impression_loader": _get_impression_dataloader,
    }

    if config["custom_dataloader"] in register_table:
        loader = register_table[config["custom_dataloader"]](config, phase)
        print(f"Using dataloader: {loader.__name__} for phase '{phase}'")
        return loader 
    else:
        raise ValueError(
            f"Dataloader '{config['custom_dataloader']}' for model '{config['model']}' is not registered."
        )
        

def _get_impression_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Customized function for ImpressionLoader models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
        
    if phase == "evaluation":
        raise ValueError("The 'evaluation' phase has been deprecated, please use 'valid' or 'test' instead.")

    if phase == "train":
        return TrainImpressionDataLoader
    elif phase == "valid":
        return ValImpressionDataLoader
    elif phase == "test":
        return TestImpressionDataLoader