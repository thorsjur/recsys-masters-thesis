from typing import List, Tuple
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from loaders.base_loader import AbstractDataLoader, DatasetConfig


def _count_file_lines(filepath: str) -> int:
    """Count lines in a file efficiently using buffered reading."""
    with open(filepath, "rb") as f:
        bufsize = 1024 * 1024  # 1MB buffer
        read_f = f.raw.read  # type: ignore[union-attr]
        lines = sum(buf.count(b"\n") for buf in iter(lambda: read_f(bufsize), b""))
    return lines


class MINDDataLoader(AbstractDataLoader):
    """DataLoader for MIND (Microsoft News Dataset)."""

    CHUNK_SIZE = 50000

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def _get_data_paths(self) -> List[str]:
        """Get list of paths to load data from.

        Returns:
            List of absolute paths to directories containing news.tsv and behaviors.tsv.
        """
        subfolders = self.config.options.get("subfolders")

        if subfolders:
            paths = [os.path.join(self.config.raw_path, subfolder) for subfolder in subfolders]
            # Validate all paths exist
            for path in paths:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"Subfolder not found: {path}")
            return paths

        # Default: load directly from raw_path
        return [self.config.raw_path]

    def _load_news_file(self, path: str) -> pd.DataFrame:
        """Load news.tsv from a single directory."""
        news_path = os.path.join(path, "news.tsv")
        # Use integer indices for usecols since header=None (no column names in file)
        # Columns: 0=item_id, 1=category, 2=sub_category, 3=title, 4=abstract, 5=url, 6=t_ents, 7=a_ents
        return pd.read_csv(
            news_path,
            sep="\t",
            header=None,
            names=["item_id", "category", "sub_category", "title", "abstract", "url", "t_ents", "a_ents"],
            usecols=[0, 1, 2, 3, 4],  # item_id, category, sub_category, title, abstract
        )

    def _process_behaviors_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of behaviors data into atomic interactions.
        
        Memory-optimized: drops unnecessary columns early and uses efficient dtypes.
        """
        # Convert time string to unix timestamp (vectorized)
        timestamps = pd.to_datetime(chunk["time_str"], format="%m/%d/%Y %I:%M:%S %p").astype(np.int64) // 10**9

        # Explode impressions (skip history - not needed for conversion, saves significant memory)
        impressions_split = chunk["impressions"].str.split(" ")
        impression_lengths = impressions_split.str.len()
        
        df_exploded = pd.DataFrame(
            {
                "user_id": chunk["user_id"].values.repeat(impression_lengths),
                "timestamp": timestamps.values.repeat(impression_lengths),
                "impression_id": chunk["impression_id"].values.repeat(impression_lengths),
                "impressions": np.concatenate(impressions_split.values),
            }
        )

        # Split "N123-1" into item_id and label (vectorized)
        split_data = df_exploded["impressions"].str.split("-", n=1, expand=True)
        split_data.columns = ["item_id", "label"]
        df_exploded["item_id"] = split_data["item_id"]
        df_exploded["label"] = split_data["label"].astype(np.float32)
        
        # Drop impressions column immediately to free memory
        df_exploded.drop(columns=["impressions"], inplace=True)

        return df_exploded[["user_id", "item_id", "timestamp", "label", "impression_id"]]

    def _load_behaviors_file(self, path: str, impression_id_offset: int = 0) -> Tuple[pd.DataFrame, int]:
        """Load and process behaviors.tsv from a single directory.

        Args:
            path: Directory containing behaviors.tsv.
            impression_id_offset: Offset to add to impression_ids to ensure uniqueness
                when merging multiple files.

        Returns:
            Tuple of (DataFrame with processed atomic interactions, total rows processed).
        """
        behaviors_path = os.path.join(path, "behaviors.tsv")
        folder_name = os.path.basename(path)

        # Count total lines for progress bar
        total_lines = _count_file_lines(behaviors_path)
        total_chunks = (total_lines + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

        chunk_iterator = pd.read_csv(
            behaviors_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time_str", "history", "impressions"],
            chunksize=self.CHUNK_SIZE,
        )

        chunks = []
        rows_processed = 0
        interactions_created = 0

        with tqdm(
            total=total_lines,
            desc=f"{folder_name}",
            unit="rows",
            unit_scale=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for chunk_idx, chunk in enumerate(chunk_iterator):
                chunk_size = len(chunk)
                processed = self._process_behaviors_chunk(chunk)

                if impression_id_offset > 0:
                    processed["impression_id"] = processed["impression_id"] + impression_id_offset

                chunks.append(processed)
                rows_processed += chunk_size
                interactions_created += len(processed)
                pbar.update(chunk_size)
                pbar.set_postfix(
                    chunk=f"{chunk_idx + 1}/{total_chunks}",
                    interactions=f"{interactions_created:,}",
                    refresh=False,
                )

        result = pd.concat(chunks, ignore_index=True)
        
        # Free memory from chunks list
        del chunks
        
        # Optimize memory with efficient dtypes
        result["user_id"] = result["user_id"].astype("category")
        result["item_id"] = result["item_id"].astype("category")
        
        return result, rows_processed

    def _load_raw_data(self):
        """Load raw MIND data from one or more directories."""
        data_paths = self._get_data_paths()

        path_names = [os.path.basename(p) for p in data_paths]
        print(f"Loading MIND ({self.config.version}) from {len(data_paths)} source(s): {path_names}")

        # Load and merge news files (deduplicate by item_id)
        print("Loading news files...")
        news_dfs = [self._load_news_file(path) for path in data_paths]
        self.df_item = pd.concat(news_dfs, ignore_index=True).drop_duplicates(subset=["item_id"], keep="first")
        print(f"  Loaded {len(self.df_item):,} unique news articles")

        # Load and merge behaviors files
        print("Loading behaviors files...")
        interactions_dfs = []
        impression_id_offset = 0
        total_rows = 0

        for path in data_paths:
            df, rows = self._load_behaviors_file(path, impression_id_offset)
            interactions_dfs.append(df)
            total_rows += rows
            # Update offset for next file to ensure unique impression_ids
            impression_id_offset = int(df["impression_id"].max()) + 1

        if len(interactions_dfs) == 1:
            self.df_inter = interactions_dfs[0]
        else:
            self.df_inter = pd.concat(interactions_dfs, ignore_index=True)
            # Re-optimize category dtypes after concat (they get converted back to object)
            self.df_inter["user_id"] = self.df_inter["user_id"].astype("category")
            self.df_inter["item_id"] = self.df_inter["item_id"].astype("category")
        
        # Free memory
        del interactions_dfs
        
        print(f"Loaded {len(self.df_inter):,} atomic interactions from {total_rows:,} behavior rows.")