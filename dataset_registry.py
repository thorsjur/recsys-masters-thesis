from loaders.base_loader import DatasetConfig
from loaders.mind_loader import MINDDataLoader
from loaders.converters.mind_converter import MINDAtomicConverter
from loaders.processing.recursive_pruner import RecursivePruner
from loaders.processing.spacy_text_cleaner import SpacyTextCleaner

def get_mind_small_baseline():
    """MIND Small with standard preprocessing."""
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
    )
    
    return config, MINDDataLoader

DATASET_REGISTRY = {
    "mind_small_baseline": get_mind_small_baseline,
    "mind_small_no_preprocessing": get_mind_no_preprocessing,
}