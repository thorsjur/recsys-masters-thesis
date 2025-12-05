from abc import ABC, abstractmethod
import pandas as pd
import os
from typing import Dict, Optional

class BaseAtomicConverter(ABC):
    def __init__(self, config, 
                 df_inter_loaded: Optional[pd.DataFrame] = None, 
                 df_item_loaded: Optional[pd.DataFrame] = None):
        self.config = config
        self.dataset_name = config.dataset_name
        self.output_path = os.path.join(config.output_dir, self.dataset_name)
        
        self._df_inter = df_inter_loaded
        self._df_item = df_item_loaded
        
        os.makedirs(self.output_path, exist_ok=True)

    @property
    @abstractmethod
    def inter_fields(self) -> Dict[str, str]:
        """Column mapping for interaction data."""
        pass

    @property
    @abstractmethod
    def item_fields(self) -> Dict[str, str]:
        """Column mapping for item data."""
        pass

    @abstractmethod
    def load_inter_df(self) -> pd.DataFrame:
        """Load interaction dataframe."""
        pass

    @abstractmethod
    def load_item_df(self) -> pd.DataFrame:
        """Load item dataframe."""
        pass

    def convert(self):
        """Convert data to atomic file format."""
        print(f"[{self.dataset_name}] Starting conversion...")


        try:
            df_inter = self.load_inter_df()
            self._write_atomic_file(df_inter, self.inter_fields, f"{self.dataset_name}.inter")
        except NotImplementedError:
            print(f"[{self.dataset_name}] No interaction loader defined. Skipping.")

        try:
            df_item = self.load_item_df()
            self._write_atomic_file(df_item, self.item_fields, f"{self.dataset_name}.item")
        except NotImplementedError:
            print(f"[{self.dataset_name}] No item loader defined. Skipping.")
            
        print(f"[{self.dataset_name}] Conversion Complete.")

    def write_interaction_file(self, df: pd.DataFrame, filename: str):
        """
        Public wrapper for the generic writer.
        Used by the Loader to save Train/Valid/Test splits.
        """
        # We reuse the existing interaction field mapping
        self._write_atomic_file(df, self.inter_fields, filename)

    def _write_atomic_file(self, df: pd.DataFrame, field_mapping: Dict[str, str], filename: str):
        """
        Generic writer that replaces the complex RecBole loop.
        """
        if df is None or df.empty:
            print(f"Warning: DataFrame for {filename} is empty. Skipping write.")
            return

        # 1. Select only the columns present in the mapping
        # Keys of field_mapping are the columns in the internal DF
        source_cols = list(field_mapping.keys())
        
        # Check for missing columns
        missing = [c for c in source_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe for {filename}: {missing}")

        # 2. Create a view with only necessary columns
        output_df = df[source_cols].copy()

        # 3. Rename columns to RecBole format (e.g. 'userId' -> 'user_id:token')
        output_df.rename(columns=field_mapping, inplace=True)

        # 4. Write to disk
        full_path = os.path.join(self.output_path, filename)
        output_df.to_csv(full_path, sep='\t', index=False)
        
        print(f"Saved {filename} ({len(output_df)} rows)")