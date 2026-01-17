from loaders.base_loader import DatasetConfig
from loaders.mind_loader import MINDDataLoader
from loaders.converters.mind_converter import MINDAtomicConverter

def get_mind_small_baseline():
    """MIND Small with standard preprocessing."""
    
    from loaders.processing.recursive_pruner import RecursivePruner
    from loaders.processing.spacy_text_cleaner import SpacyTextCleaner
    
    pipeline = [
        SpacyTextCleaner(target_col="title", output_col="title", batch_size=2000),
        SpacyTextCleaner(target_col="abstract", output_col="abstract", batch_size=100),
        RecursivePruner(min_user_hist=5, min_item_freq=10)
    ]
    
    config = DatasetConfig(
        raw_path='./datasets/MINDsmall_train',
        output_dir='./datasets/atomic_files',
        dataset_name='mind_small',
        version='small',
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None
    )
    
    return config, MINDDataLoader

def get_mind_no_preprocessing():
    """MIND Small without preprocessing."""
    pipeline = []
    
    config = DatasetConfig(
        raw_path='./datasets/MINDsmall_train',
        output_dir='./datasets/atomic_files',
        dataset_name='mind_no_preprocessing',
        version='small',
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None
    )
    
    return config, MINDDataLoader

def get_mind_large_no_preprocessing():
    """MIND Small with standard preprocessing."""
    pipeline = []
    
    config = DatasetConfig(
        raw_path='./datasets/MINDlarge',
        output_dir='./datasets/atomic_files',
        dataset_name='mind_large',
        version='large',
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        
        # Test is excluded, since MIND Large does not have labels for test set
        options={"subfolders": ["train", "dev"]}
    )
    
    return config, MINDDataLoader

DATASET_REGISTRY = {
    "mind_small_baseline": get_mind_small_baseline,
    "mind_small_no_preprocessing": get_mind_no_preprocessing,
    "mind_large_no_preprocessing": get_mind_large_no_preprocessing,
}

def get_available_datasets() -> list[str]:
    """Get list of available dataset configurations."""
    return list(DATASET_REGISTRY.keys())

def get_dataset_description(name: str) -> str:
    """Get description of a dataset configuration."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name].__doc__ or "No description available"