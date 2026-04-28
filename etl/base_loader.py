from abc import ABC, abstractmethod
from typing import List, Optional, Set, Type, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from etl.converters.base_converter import BaseAtomicConverter
from etl.processing.base_preprocessor import BasePreprocessor, BaseEarlyPreprocessor
from etl.splitters.base_splitter import BaseSplitter


@dataclass
class DatasetConfig:
    raw_path: str
    dataset_name: str
    version: str
    converter_class: Type[BaseAtomicConverter]
    output_dir: str = "./data/atomic_files"
    preprocessors: List[BasePreprocessor] = field(default_factory=list)
    splitter: Optional[BaseSplitter] = None
    spacy_model: str = "en_core_web_sm"
    min_user_history: int = 5
    min_item_frequency: int = 10
    temporal_days: Optional[Union[int, Tuple[int, str]]] = None
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.raw_path:
            raise ValueError("raw_path cannot be empty string")

        if self.min_user_history < 1:
            raise ValueError("min_user_history must be positive")

        if self.min_item_frequency < 1:
            raise ValueError("min_item_frequency must be positive")


class AbstractDataLoader(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self._df_inter: Optional[pd.DataFrame] = None
        self._df_item: Optional[pd.DataFrame] = None

    @property
    def df_inter(self) -> pd.DataFrame:
        assert self._df_inter is not None, "df_inter has not been loaded yet"
        return self._df_inter

    @df_inter.setter
    def df_inter(self, value: pd.DataFrame):
        self._df_inter = value

    @property
    def df_item(self) -> pd.DataFrame:
        assert self._df_item is not None, "df_item has not been loaded yet"
        return self._df_item

    @df_item.setter
    def df_item(self, value: pd.DataFrame):
        self._df_item = value

    def execute_etl_pipeline(self):
        """Run the full ETL pipeline."""
        print(f"[{self.__class__.__name__}] Starting pipeline...")

        # 1. Load raw data
        self._load_raw_data()

        # 2. Run preprocessing pipeline
        self._preprocess_content()

        # 3. Generate splits and atomic Files
        self._export_atomic_files()

        print(f"[{self.__class__.__name__}] Pipeline Complete.")

    @abstractmethod
    def _load_raw_data(self):
        """Load raw data into df_inter and df_item."""
        pass

    def _resolve_early_user_filter(self, all_user_ids: np.ndarray) -> Optional[Set]:
        """Run any BaseEarlyPreprocessor instances in the pipeline."""
        early = [p for p in self.config.preprocessors if isinstance(p, BaseEarlyPreprocessor)]
        if not early:
            return None

        selected = all_user_ids
        for ep in early:
            selected = ep.select_users(selected)

        return set(selected)

    def _preprocess_content(self):
        """Apply preprocessing steps from config.

        BaseEarlyPreprocessor instances are skipped here because
        they have already been applied during raw-data loading.
        """
        regular = [p for p in self.config.preprocessors if not isinstance(p, BaseEarlyPreprocessor)]
        print(f"[{self.__class__.__name__}] Running {len(regular)} preprocessors...")

        for processor in regular:
            self.df_inter, self.df_item = processor.process(self.df_inter, self.df_item)

    def _export_atomic_files(self):
        """Convert data to atomic format and generate train/valid/test splits."""
        assert self.config.converter_class is not None, "converter_class must be set in config"

        print(f"[{self.__class__.__name__}] Initializing converter: {self.config.converter_class.__name__}")

        converter = self.config.converter_class(
            config=self.config, df_inter_loaded=self.df_inter, df_item_loaded=self.df_item
        )

        # 2. Run main conversion (creates dataset.inter and dataset.item)
        converter.convert()

        # Generate day-wise splits for temporal experiments
        if self.config.temporal_days:
            self._generate_day_wise_splits(converter)
            return

        # Generate train/valid/test splits using the configured splitter
        # This is usually not used when running purely temporal experiments, as this does a
        # global split of the whole dataset, and does not produce splits which can be used
        # to build the windows in the sliding windows protocol.
        if self.config.splitter is None:
            print(f"[{self.__class__.__name__}] No splitter defined. Skipping split generation.")
            return

        # 3. Run Split Strategy
        print(f"[{self.__class__.__name__}] Running split strategy: {self.config.splitter.__class__.__name__}")
        train, valid, test = self.config.splitter.split(self.df_inter)

        # 4. Save splits using the converter's logic
        # This ensures the splits have the same headers/format as the main file
        base_name = self.config.dataset_name

        converter.write_interaction_file(train, f"{base_name}.train.inter")
        converter.write_interaction_file(valid, f"{base_name}.valid.inter")
        converter.write_interaction_file(test, f"{base_name}.test.inter")

        print(f"[{self.__class__.__name__}] Export Complete.")

    def _generate_day_wise_splits(self, converter):
        """
        Split interactions by time for temporal stability experiments.
        """
        if isinstance(self.config.temporal_days, tuple):
            # Hour-level granularity
            num_units, granularity = self.config.temporal_days
            if granularity != "hour":
                print(f"Warning: Unknown granularity '{granularity}', expected 'hour'")
                return
            seconds_per_unit = 3600  # 1 hour
            unit_name = "hour"
        else:
            # Day-level granularity
            num_units = self.config.temporal_days
            seconds_per_unit = 86400  # 1 day
            unit_name = "day"

        print(f"[{self.__class__.__name__}] Generating {unit_name}-wise splits (up to {num_units} {unit_name}s)...")

        if "timestamp" not in self.df_inter.columns:
            print(f"Warning: No 'timestamp' column found. Skipping temporal splits.")
            return

        # Convert timestamp to time unit number
        min_timestamp = self.df_inter["timestamp"].min()
        self.df_inter["time_unit"] = ((self.df_inter["timestamp"] - min_timestamp) / seconds_per_unit).astype(int) + 1

        available_units = sorted(self.df_inter["time_unit"].unique())
        max_unit = min(max(available_units), num_units)

        print(f"Dataset spans {len(available_units)} {unit_name}s, generating splits for {unit_name}s 1-{max_unit}")

        base_name = self.config.dataset_name

        for unit in range(1, max_unit + 1):
            unit_df = self.df_inter[self.df_inter["time_unit"] == unit].copy()

            if len(unit_df) == 0:
                print(f"Warning: {unit_name.capitalize()} {unit} has no interactions, skipping")
                continue

            unit_df = unit_df.drop(columns=["time_unit"])

            converter.write_interaction_file(unit_df, f"{base_name}.{unit_name}_{unit}.inter")

        print(f"Generated {max_unit} {unit_name}-wise interaction files")
