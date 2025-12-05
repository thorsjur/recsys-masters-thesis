from abc import ABC, abstractmethod
from typing import List, Optional, Type
import pandas as pd
from dataclasses import dataclass, field

from loaders.converters.base_converter import BaseAtomicConverter
from loaders.processing.base_preprocessor import BasePreprocessor
from loaders.splitters.base_splitter import BaseSplitter

@dataclass
class DatasetConfig:
    raw_path: str
    output_dir: str
    dataset_name: str
    version: str
    converter_class: Type[BaseAtomicConverter]
    preprocessors: List[BasePreprocessor] = field(default_factory=list)
    splitter: Optional[BaseSplitter] = None
    spacy_model: str = "en_core_web_sm"
    min_user_history: int = 5
    min_item_frequency: int = 10

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

    def execute_pipeline(self):
        """Run the full ETL pipeline."""
        print(f"[{self.__class__.__name__}] Starting pipeline...")
        
        # 1. Load Raw Data
        self._load_raw_data()
        
        # 2. Run preprocessing pipeline
        self._preprocess_content()
        
        # 3. Generate Splits and Atomic Files
        self._export_atomic_files()
        
        print(f"[{self.__class__.__name__}] Pipeline Complete.")

    @abstractmethod
    def _load_raw_data(self):
        """Load raw data into df_inter and df_item."""
        pass

    def _preprocess_content(self):
        """Apply preprocessing steps from config."""
        print(f"[{self.__class__.__name__}] Running {len(self.config.preprocessors)} preprocessors...")
        
        for processor in self.config.preprocessors:
            self.df_inter, self.df_item = processor.process(self.df_inter, self.df_item)

    def _export_atomic_files(self):
        """Convert data to atomic format and generate train/valid/test splits."""
        assert self.config.converter_class is not None, "converter_class must be set in config"

        print(f"[{self.__class__.__name__}] Initializing converter: {self.config.converter_class.__name__}")
        
        converter = self.config.converter_class(
            config=self.config,
            df_inter_loaded=self.df_inter,
            df_item_loaded=self.df_item
        )
        
        # 2. Run Main Conversion (creates dataset.inter and dataset.item)
        converter.convert()


        if self.config.splitter is None:
            print(f"[{self.__class__.__name__}] No Splitter defined. Skipping split generation.")
            return
        
        # 3. Run Split Strategy
        print(f"[{self.__class__.__name__}] Running Split Strategy: {self.config.splitter.__class__.__name__}")
        train, valid, test = self.config.splitter.split(self.df_inter)
        
        # 4. Save Splits using the Converter's logic
        # This ensures the splits have the exact same headers/format as the main file
        base_name = self.config.dataset_name
        
        converter.write_interaction_file(train, f"{base_name}.train.inter")
        converter.write_interaction_file(valid, f"{base_name}.valid.inter")
        converter.write_interaction_file(test,  f"{base_name}.test.inter")
        
        print(f"[{self.__class__.__name__}] Export Complete.")
