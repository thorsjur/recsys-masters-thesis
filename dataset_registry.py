from etl.base_loader import DatasetConfig
from etl.processing.nltk_tokenizer import NLTKTokenizer
from etl.processing.base_preprocessor import BasePreprocessor
from typing import List


def get_ebnerd():
    """EB-NeRD medium (200k users from large) with no preprocessing."""
    from etl.converters.ebnerd_impression_converter import EBNeRDImpressionAtomicConverter
    from etl.ebnerd_impression_loader import EBNeRDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler

    pipeline: List[BasePreprocessor] = [
        UserSampler(n_users=200_000, seed=42),
    ]

    config = DatasetConfig(
        raw_path="./data/EBNeRDlarge",
        dataset_name="ebnerd",
        version="large_subset",
        converter_class=EBNeRDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "validation"], "max_history_items": 50},
    )

    return config, EBNeRDImpressionDataLoader

def get_ebnerd_cleaned():
    """EB-NeRD medium (200k users from large) with text cleaning on title and abstract."""
    from etl.converters.ebnerd_impression_converter import EBNeRDImpressionAtomicConverter
    from etl.ebnerd_impression_loader import EBNeRDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler
    from etl.processing.spacy_text_cleaner import SpacyTextCleaner

    pipeline = [
        UserSampler(n_users=200_000, seed=42),
        SpacyTextCleaner(target_col="title", output_col="title", batch_size=2000),
        SpacyTextCleaner(target_col="abstract", output_col="abstract", batch_size=2000),
    ]

    config = DatasetConfig(
        raw_path="./data/EBNeRDlarge",
        dataset_name="ebnerd_cleaned",
        version="large_subset",
        converter_class=EBNeRDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "validation"], "max_history_items": 50},
    )

    return config, EBNeRDImpressionDataLoader


def get_ebnerd_tokenized():
    """EB-NeRD medium (200k users from large) with NLTK tokenization on title and abstract."""
    from etl.converters.ebnerd_impression_converter import EBNeRDImpressionAtomicConverter
    from etl.ebnerd_impression_loader import EBNeRDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler

    pipeline = [
        UserSampler(n_users=200_000, seed=42),
        NLTKTokenizer(to_lower=True, item_text_fields=["title", "abstract"]),
    ]

    config = DatasetConfig(
        raw_path="./data/EBNeRDlarge",
        dataset_name="ebnerd_tokenized",
        version="large_subset",
        converter_class=EBNeRDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "validation"], "max_history_items": 50},
    )

    return config, EBNeRDImpressionDataLoader


def get_ebnerd_cleaned_tokenized():
    """EB-NeRD medium (200k users from large) with text cleaning and NLTK tokenization on title and abstract."""
    from etl.converters.ebnerd_impression_converter import EBNeRDImpressionAtomicConverter
    from etl.ebnerd_impression_loader import EBNeRDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler
    from etl.processing.spacy_text_cleaner import SpacyTextCleaner

    pipeline = [
        UserSampler(n_users=200_000, seed=42),
        SpacyTextCleaner(target_col="title", output_col="title", batch_size=2000),
        SpacyTextCleaner(target_col="abstract", output_col="abstract", batch_size=2000),
        NLTKTokenizer(to_lower=True, item_text_fields=["title", "abstract"]),
    ]

    config = DatasetConfig(
        raw_path="./data/EBNeRDlarge",
        dataset_name="ebnerd_cleaned_tokenized",
        version="large_subset",
        converter_class=EBNeRDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "validation"], "max_history_items": 50},
    )

    return config, EBNeRDImpressionDataLoader


def get_mind():
    """MIND medium (200k users from large), unprocessed."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.mind_impression_loader import MINDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler

    pipeline: List[BasePreprocessor] = [
        UserSampler(n_users=200_000, seed=42),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDlarge",
        dataset_name="mind",
        version="large_subset",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDImpressionDataLoader

def get_mind_cleaned():
    """MIND medium (200k users from large) with text cleaning on title and abstract."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.mind_impression_loader import MINDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler
    from etl.processing.spacy_text_cleaner import SpacyTextCleaner

    pipeline = [
        UserSampler(n_users=200_000, seed=42),
        SpacyTextCleaner(target_col="title", output_col="title", batch_size=2000),
        SpacyTextCleaner(target_col="abstract", output_col="abstract", batch_size=2000),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDlarge",
        dataset_name="mind_cleaned",
        version="large_subset",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDImpressionDataLoader


def get_mind_tokenized():
    """MIND medium (200k users from large) with NLTK tokenization on title and abstract."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.mind_impression_loader import MINDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler

    pipeline = [
        UserSampler(n_users=200_000, seed=42),
        NLTKTokenizer(to_lower=True, item_text_fields=["title", "abstract"]),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDlarge",
        dataset_name="mind_tokenized",
        version="large_subset",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDImpressionDataLoader


def get_mind_cleaned_tokenized():
    """MIND medium (200k users from large) with text cleaning and NLTK tokenization on title and abstract."""
    from etl.converters.mind_impression_converter import MINDImpressionAtomicConverter
    from etl.mind_impression_loader import MINDImpressionDataLoader
    from etl.processing.user_sampler import UserSampler
    from etl.processing.spacy_text_cleaner import SpacyTextCleaner

    pipeline = [
        UserSampler(n_users=200_000, seed=42),
        SpacyTextCleaner(target_col="title", output_col="title", batch_size=2000),
        SpacyTextCleaner(target_col="abstract", output_col="abstract", batch_size=2000),
        NLTKTokenizer(to_lower=True, item_text_fields=["title", "abstract"]),
    ]

    config = DatasetConfig(
        raw_path="./data/MINDlarge",
        dataset_name="mind_cleaned_tokenized",
        version="large_subset",
        converter_class=MINDImpressionAtomicConverter,
        preprocessors=pipeline,
        splitter=None,
        options={"subfolders": ["train", "dev"]},
    )

    return config, MINDImpressionDataLoader


DATASET_REGISTRY = {
    "ebnerd": get_ebnerd,
    "ebnerd_cleaned": get_ebnerd_cleaned,
    "ebnerd_tokenized": get_ebnerd_tokenized,
    "ebnerd_cleaned_tokenized": get_ebnerd_cleaned_tokenized,
    "mind": get_mind,
    "mind_cleaned": get_mind_cleaned,
    "mind_tokenized": get_mind_tokenized,
    "mind_cleaned_tokenized": get_mind_cleaned_tokenized,
}


def get_available_datasets() -> list[str]:
    """Get list of available dataset configurations."""
    return list(DATASET_REGISTRY.keys())


def get_dataset_description(name: str) -> str:
    """Get description of a dataset configuration."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name].__doc__ or "No description available"
