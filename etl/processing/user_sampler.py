from typing import Optional
import numpy as np
import pandas as pd

from etl.processing.base_preprocessor import BaseEarlyPreprocessor


class UserSampler(BaseEarlyPreprocessor):
    """Randomly sample n_users unique users and keep all their interactions."""

    def __init__(self, n_users: int, seed: Optional[int] = 42) -> None:
        if n_users < 1:
            raise ValueError(f"n_users must be >= 1, n_users={n_users}")
        self.n_users = n_users
        self.seed = seed

    def select_users(self, all_user_ids: np.ndarray) -> np.ndarray:
        """Return a randomly sampled subset of all user ids."""
        total = len(all_user_ids)

        if total <= self.n_users:
            print(f"[UserSampler] Only {total:,} users", f"keeping all (requested {self.n_users:,}).")
            return all_user_ids

        rng = np.random.default_rng(self.seed)
        selected = rng.choice(all_user_ids, size=self.n_users, replace=False)
        print(f"[UserSampler] Sampled {self.n_users:,} / " f"{total:,} users (seed={self.seed}).")
        return selected

    def process(self, df_inter: pd.DataFrame, df_item: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_users = df_inter["user_id"].unique()
        total_users = len(unique_users)

        print(f"[UserSampler] Sampling {self.n_users} / {total_users:,} users " f"(seed={self.seed})...")

        if total_users <= self.n_users:
            print(f"Dataset has only {total_users:,} users, keeping all.")
            return df_inter, df_item

        rng = np.random.default_rng(self.seed)
        selected = rng.choice(unique_users, size=self.n_users, replace=False)
        del unique_users

        mask = df_inter["user_id"].isin(selected)
        del selected
        df_inter_out = df_inter.loc[mask]
        del mask

        # Remove items that no longer appear in the sampled interactions
        valid_items = df_inter_out["item_id"].unique()
        df_item_out = df_item.loc[df_item["item_id"].isin(valid_items)]

        print(f"Kept {len(df_inter_out):,} / {len(df_inter):,} interactions " f"({len(valid_items):,} items).")

        return df_inter_out, df_item_out
