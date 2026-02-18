from abc import ABC, abstractmethod
from typing import List, Tuple
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from etl.base_loader import AbstractDataLoader, DatasetConfig


class EBNeRDBaseDataLoader(AbstractDataLoader, ABC):
    """
    Common base for EB-NeRD loaders.

    Reads parquet files
    """

    CHUNK_SIZE = 50_000

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def _get_data_paths(self) -> List[str]:
        """Get list of subfolder paths to load data from."""
        subfolders = self.config.options.get("subfolders")

        if subfolders:
            paths = [os.path.join(self.config.raw_path, subfolder) for subfolder in subfolders]
            for path in paths:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"Subfolder not found: {path}")
            return paths

        return [self.config.raw_path]

    def _load_articles_file(self) -> pd.DataFrame:
        """Load articles.parquet from the dataset root directory."""
        articles_path = os.path.join(self.config.raw_path, "articles.parquet")
        return pd.read_parquet(articles_path)

    def _load_history_file(self, path: str) -> pd.DataFrame:
        """Load history.parquet from a single data-split directory.

        If ``max_history_items`` is set in config options, each user's
        article list is truncated to the **last** K items (most recent).
        """
        history_path = os.path.join(path, "history.parquet")
        if not os.path.isfile(history_path):
            return pd.DataFrame(columns=["user_id", "article_id_fixed"])

        df = pd.read_parquet(history_path, columns=["user_id", "article_id_fixed"])

        max_hist = self.config.options.get("max_history_items")
        if max_hist is not None:
            print(f"Truncating user histories to last {max_hist} items...")
            df["article_id_fixed"] = df["article_id_fixed"].apply(
                lambda lst: lst[-max_hist:] if lst is not None and hasattr(lst, '__len__') and len(lst) > max_hist else lst
            )

        return df

    @abstractmethod
    def _process_behaviors_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a behaviors chunk into interactions DataFrame.
        """
        raise NotImplementedError("Subclasses must implement _process_behaviors_chunk().")

    def _finalize_interactions_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce user_id and item_id as category for memory efficiency.
        """
        if "user_id" in df.columns:
            df["user_id"] = df["user_id"].astype("category")
        if "item_id" in df.columns:
            df["item_id"] = df["item_id"].astype("category")
        return df

    def _load_behaviors_file(self, path: str, impression_id_offset: int = 0) -> Tuple[pd.DataFrame, int]:
        """Load and process behaviors.parquet (+ history.parquet) from a single directory."""
        behaviors_path = os.path.join(path, "behaviors.parquet")
        folder_name = os.path.basename(path)

        df_behaviors = pd.read_parquet(behaviors_path)

        # Merge user click history from separate history file
        history_df = self._load_history_file(path)
        if not history_df.empty:
            df_behaviors = df_behaviors.merge(history_df, on="user_id", how="left")

        total_rows = len(df_behaviors)
        total_chunks = (total_rows + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

        chunks: List[pd.DataFrame] = []
        rows_created = 0

        with tqdm(
            total=total_rows,
            desc=f"{folder_name}",
            unit="rows",
            unit_scale=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for chunk_idx, start_idx in enumerate(range(0, total_rows, self.CHUNK_SIZE)):
                end_idx = min(start_idx + self.CHUNK_SIZE, total_rows)
                chunk = df_behaviors.iloc[start_idx:end_idx]

                processed = self._process_behaviors_chunk(chunk)

                required_cols = {"user_id", "item_id", "timestamp", "impression_id"}
                missing = required_cols - set(processed.columns)
                if missing:
                    raise ValueError(
                        f"{self.__class__.__name__}._process_behaviors_chunk() must return columns "
                        f"{sorted(required_cols)}; missing {sorted(missing)}."
                    )

                if impression_id_offset > 0:
                    processed["impression_id"] = processed["impression_id"] + impression_id_offset

                max_negs = self.config.options.get("max_neg_items")
                if max_negs is not None and "neg_item_id_list" in processed.columns:
                    processed["neg_item_id_list"] = processed["neg_item_id_list"].apply(
                        lambda s: " ".join(s.split()[:max_negs]) if s else s
                    )

                chunks.append(processed)
                rows_created += len(processed)

                pbar.update(len(chunk))
                pbar.set_postfix(
                    chunk=f"{chunk_idx + 1}/{total_chunks}",
                    created=f"{rows_created:,}",
                    refresh=False,
                )

        result = pd.concat(chunks, ignore_index=True)
        del chunks

        result = self._finalize_interactions_df(result)
        return result, total_rows

    def _load_raw_data(self):
        """Load raw EB-NeRD data from one or more directories."""
        data_paths = self._get_data_paths()
        path_names = [os.path.basename(p) for p in data_paths]
        print(f"Loading EB-NeRD ({self.config.version}) from {len(data_paths)} source(s): {path_names}")

        # Articles (single file at dataset root)
        print("Loading articles file...")
        self.df_item = self._load_articles_file()
        self.df_item = self.df_item.rename(columns={"article_id": "item_id"})
        print(f"  Loaded {len(self.df_item):,} articles")

        # Behaviors
        print("Loading behaviors files...")
        interactions_dfs: List[pd.DataFrame] = []
        impression_id_offset = 0
        total_rows = 0

        for path in data_paths:
            df, rows = self._load_behaviors_file(path, impression_id_offset)
            interactions_dfs.append(df)
            total_rows += rows
            impression_id_offset = int(df["impression_id"].max()) + 1

        if len(interactions_dfs) == 1:
            self.df_inter = interactions_dfs[0]
        else:
            self.df_inter = pd.concat(interactions_dfs, ignore_index=True)
            self.df_inter = self._finalize_interactions_df(self.df_inter)

        del interactions_dfs
        print(f"Loaded {len(self.df_inter):,} rows from {total_rows:,} behavior rows.")
