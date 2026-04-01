from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from etl.base_loader import AbstractDataLoader, DatasetConfig


def _count_file_lines(filepath: str) -> int:
    """Count lines in a file efficiently using buffered reading."""
    with open(filepath, "rb") as f:
        bufsize = 1024 * 1024  # 1MB buffer
        read_f = f.raw.read  # type: ignore[union-attr]
        lines = sum(buf.count(b"\n") for buf in iter(lambda: read_f(bufsize), b""))
    return lines


class MINDBaseDataLoader(AbstractDataLoader, ABC):
    """
    Common base for MIND loaders.
    """

    CHUNK_SIZE = 50_000

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def _get_data_paths(self) -> List[str]:
        """Get list of paths to load data from."""
        subfolders = self.config.options.get("subfolders")

        if subfolders:
            paths = [os.path.join(self.config.raw_path, subfolder) for subfolder in subfolders]
            for path in paths:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"Subfolder not found: {path}")
            return paths

        return [self.config.raw_path]

    def _load_news_file(self, path: str) -> pd.DataFrame:
        """Load news.tsv from a single directory."""
        news_path = os.path.join(path, "news.tsv")
        return pd.read_csv(
            news_path,
            sep="\t",
            header=None,
            names=["item_id", "category", "sub_category", "title", "abstract", "url", "t_ents", "a_ents"],
            usecols=[0, 1, 2, 3, 4],
        )

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

    def _load_behaviors_file(
        self,
        path: str,
        impression_id_offset: int = 0,
        sampled_users: Optional[Set] = None,
    ) -> Tuple[pd.DataFrame, int]:
        """Load and process behaviors.tsv from a single directory."""
        behaviors_path = os.path.join(path, "behaviors.tsv")
        folder_name = os.path.basename(path)

        total_lines = _count_file_lines(behaviors_path)
        total_chunks = (total_lines + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE

        chunk_iterator = pd.read_csv(
            behaviors_path,
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time_str", "history", "impressions"],
            chunksize=self.CHUNK_SIZE,
        )

        chunks: List[pd.DataFrame] = []
        rows_processed = 0
        rows_created = 0
        rows_filtered = 0

        with tqdm(
            total=total_lines,
            desc=f"{folder_name}",
            unit="rows",
            unit_scale=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} rows [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for chunk_idx, chunk in enumerate(chunk_iterator):
                chunk_size = len(chunk)

                # Early user filtering (before expensive processing)
                if sampled_users is not None:
                    chunk = chunk[chunk["user_id"].isin(sampled_users)]
                    rows_filtered += chunk_size - len(chunk)
                    if chunk.empty:
                        rows_processed += chunk_size
                        pbar.update(chunk_size)
                        continue

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

                chunks.append(processed)
                rows_processed += chunk_size
                rows_created += len(processed)

                pbar.update(chunk_size)
                pbar.set_postfix(
                    chunk=f"{chunk_idx + 1}/{total_chunks}",
                    created=f"{rows_created:,}",
                    refresh=False,
                )

        if not chunks:
            result = pd.DataFrame(columns=["user_id", "item_id", "timestamp", "impression_id"])
        else:
            result = pd.concat(chunks, ignore_index=True)
        del chunks

        if sampled_users is not None:
            print(f"  [{folder_name}] Early user filter: kept {rows_processed - rows_filtered:,} / {rows_processed:,} rows")

        result = self._finalize_interactions_df(result)
        return result, rows_processed

    # ------------------------------------------------------------------
    # Early user filtering (delegates to BaseEarlyPreprocessor pipeline)
    # ------------------------------------------------------------------

    def _gather_all_user_ids(self, data_paths: List[str]) -> np.ndarray:
        """Read only the ``user_id`` column from every behaviours file
        and return the deduplicated union as a numpy array.
        """
        all_users: set = set()
        for path in data_paths:
            bp = os.path.join(path, "behaviors.tsv")
            if os.path.isfile(bp):
                uids = pd.read_csv(
                    bp, sep="\t", header=None, usecols=[1], names=["user_id"],
                )["user_id"].unique()
                all_users.update(uids)
        return np.array(list(all_users))

    def _load_raw_data(self):
        """Load raw MIND data from one or more directories."""
        data_paths = self._get_data_paths()
        path_names = [os.path.basename(p) for p in data_paths]
        print(f"Loading MIND ({self.config.version}) from {len(data_paths)} source(s): {path_names}")

        # News
        print("Loading news files...")
        news_dfs = [self._load_news_file(path) for path in data_paths]
        self.df_item = pd.concat(news_dfs, ignore_index=True).drop_duplicates(subset=["item_id"], keep="first")
        print(f"  Loaded {len(self.df_item):,} unique news articles")

        # Behaviors
        print("Loading behaviors files...")

        # Determine users to keep (None = all) via early preprocessors
        all_user_ids = self._gather_all_user_ids(data_paths)
        sampled_users = self._resolve_early_user_filter(all_user_ids)

        interactions_dfs: List[pd.DataFrame] = []
        impression_id_offset = 0
        total_rows = 0

        for path in data_paths:
            df, rows = self._load_behaviors_file(path, impression_id_offset, sampled_users)
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
