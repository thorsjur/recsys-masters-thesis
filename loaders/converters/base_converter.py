from abc import ABC, abstractmethod
import pandas as pd
import os
from typing import Dict, Optional
from tqdm import tqdm

# Chunk size for writing large files (rows per chunk)
WRITE_CHUNK_SIZE = 500000


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
        Memory-efficient writer that processes data in chunks.
        Avoids copying entire DataFrame to reduce memory usage.
        """
        if df is None or df.empty:
            print(f"Warning: DataFrame for {filename} is empty. Skipping write.")
            return

        # 1. Validate columns exist
        source_cols = list(field_mapping.keys())
        missing = [c for c in source_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe for {filename}: {missing}")

        # 2. Build header with RecBole format names
        header_cols = [field_mapping[c] for c in source_cols]
        
        full_path = os.path.join(self.output_path, filename)
        total_rows = len(df)
        total_chunks = (total_rows + WRITE_CHUNK_SIZE - 1) // WRITE_CHUNK_SIZE
        
        # 3. Write in chunks to avoid memory spikes
        with open(full_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write('\t'.join(header_cols) + '\n')
            
            # Write data in chunks with progress bar
            with tqdm(
                total=total_rows,
                desc=f"Writing {filename}",
                unit="rows",
                unit_scale=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                for chunk_idx, start_idx in enumerate(range(0, total_rows, WRITE_CHUNK_SIZE)):
                    end_idx = min(start_idx + WRITE_CHUNK_SIZE, total_rows)
                    chunk = df.iloc[start_idx:end_idx][source_cols]
                    
                    # Write chunk without header (already written)
                    chunk.to_csv(f, sep='\t', index=False, header=False, mode='a')
                    
                    pbar.update(len(chunk))
                    pbar.set_postfix(chunk=f"{chunk_idx + 1}/{total_chunks}", refresh=False)
                    
                    # Clear chunk reference
                    del chunk
        
        print(f"Saved {filename} ({total_rows:,} rows)")