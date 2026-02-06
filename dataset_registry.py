from etl.base_loader import DatasetConfig
from etl.mind_loader import MINDDataLoader
from etl.converters.mind_converter import MINDAtomicConverter
from etl.processing.nltk_tokenizer import NLTKTokenizer


def get_mind_small_baseline():
    """MIND Small with standard preprocessing."""

    from etl.processing.recursive_pruner import RecursivePruner
    from etl.processing.spacy_text_cleaner import SpacyTextCleaner

    pipeline = [
        SpacyTextCleaner(target_col="title", output_col="title", batch_size=2000),
        SpacyTextCleaner(target_col="abstract", output_col="abstract", batch_size=100),
        RecursivePruner(min_user_hist=5, min_item_freq=10),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDsmall_train",
        dataset_name="mind_small",
        version="small",
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
    )

    return config, MINDDataLoader


def get_mind_no_preprocessing():
    """MIND Small without preprocessing."""
    pipeline = []

    config = DatasetConfig(
        raw_path="./data/MINDsmall_train",
        dataset_name="mind_no_preprocessing",
        version="small",
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
    )

    return config, MINDDataLoader


def get_mind_large_no_preprocessing():
    """MIND Small with standard preprocessing."""
    pipeline = []

    config = DatasetConfig(
        raw_path="./data/MINDlarge",
        dataset_name="mind_large",
        version="large",
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        # Test is excluded, since MIND Large does not have labels for test set
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDDataLoader


def get_mind_small_minor_preprocessing():
    """MIND Small with minor preprocessing."""
    from etl.processing.recursive_pruner import RecursivePruner
    from etl.processing.value_filter import ValueFilter
    from etl.processing.base_preprocessor import BasePreprocessor

    pipeline: list[BasePreprocessor] = [
        # To reduce data set size, we only keep actual interactions and not negatives
        # in impressions. For future experiments we might want to include negative interactions.
        ValueFilter(col_name="label", valid_values=[1]),
        # Minimal pruning to ensure all items and users have at least one interaction
        RecursivePruner(min_user_hist=1, min_item_freq=1),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDsmall_train",
        dataset_name="mind_small_minor_preprocessing",
        version="small",
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
    )

    return config, MINDDataLoader


def get_mind_large_impressions():
    """MIND Small with negative interactions for each interaction."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.processing.base_preprocessor import BasePreprocessor
    from etl.mind_impression_loader import MINDImpressionDataLoader

    pipeline: list[BasePreprocessor] = [
        # We do not want to prune, as it might remove items present in the negatives list
        # (which has no interactions itself). Instead, I rely on the original MIND preprocessing.
        # RecursivePruner(min_user_hist=1, min_item_freq=1),
        # Tokenization: We only tokenize title field, as that is the only field currently used.
        NLTKTokenizer(to_lower=True, item_text_fields=["title"]),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDlarge",

        dataset_name="mind_large_impressions",
        version="large",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        
        # Test is excluded, since MIND Large does not have labels for test set
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDImpressionDataLoader


def get_mind_small_impressions():
    """MIND Small with negative interactions for each interaction."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.processing.base_preprocessor import BasePreprocessor
    from etl.mind_impression_loader import MINDImpressionDataLoader

    pipeline: list[BasePreprocessor] = [
        # We do not want to prune, as it might remove items present in the negatives list
        # (which has no interactions itself). Instead, I rely on the original MIND preprocessing.
        # RecursivePruner(min_user_hist=1, min_item_freq=1),
        # Tokenization: We only tokenize title field, as that is the only field currently used.
        NLTKTokenizer(to_lower=True, item_text_fields=["title"]),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDsmall_train",
        dataset_name="mind_small_impressions",
        version="small",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
    )

    return config, MINDImpressionDataLoader

def get_mind_mini_impressions():
    """Minified MIND with only 5000 users, for quick testing with impression negatives."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.processing.base_preprocessor import BasePreprocessor
    from etl.mind_impression_loader import MINDImpressionDataLoader

    pipeline: list[BasePreprocessor] = [
        # We do not want to prune, as it might remove items present in the negatives list
        # (which has no interactions itself). Instead, I rely on the original MIND preprocessing.
        # RecursivePruner(min_user_hist=1, min_item_freq=1),
        # Tokenization: We only tokenize title field, as that is the only field currently used.
        NLTKTokenizer(to_lower=True, item_text_fields=["title"]),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDmini",
        dataset_name="mind_mini_impressions",
        version="mini",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
    )

    return config, MINDImpressionDataLoader


def get_mind_large_minor_preprocessing():
    """MIND Large with minor preprocessing."""
    from etl.processing.recursive_pruner import RecursivePruner
    from etl.processing.value_filter import ValueFilter
    from etl.processing.base_preprocessor import BasePreprocessor

    pipeline: list[BasePreprocessor] = [
        # To reduce data set size, we only keep actual interactions and not negatives
        # in impressions. For future experiments we might want to include negative interactions.
        ValueFilter(col_name="label", valid_values=[1]),
        # Minimal pruning to ensure all items and users have at least one interaction
        # across full dataset
        RecursivePruner(min_user_hist=1, min_item_freq=1),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDlarge",
        dataset_name="mind_large_minor_preprocessing",
        version="large",
        converter_class=MINDAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        # Test is excluded, since MIND Large does not have labels for test set
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDDataLoader


DATASET_REGISTRY = {
    "mind_small_baseline": get_mind_small_baseline,
    "mind_small_no_preprocessing": get_mind_no_preprocessing,
    "mind_small_minor_preprocessing": get_mind_small_minor_preprocessing,
    "mind_large_no_preprocessing": get_mind_large_no_preprocessing,
    "mind_large_minor_preprocessing": get_mind_large_minor_preprocessing,
    "mind_large_impressions": get_mind_large_impressions,
    "mind_small_impressions": get_mind_small_impressions,
    "mind_mini_impressions": get_mind_mini_impressions,
}


def get_available_datasets() -> list[str]:
    """Get list of available dataset configurations."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_description(name: str) -> str:
    """Get description of a dataset configuration."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name].__doc__ or "No description available"
