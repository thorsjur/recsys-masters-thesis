from abc import ABC, abstractmethod
from typing import Optional, Set
import numpy as np
import pandas as pd

class BasePreprocessor(ABC):
    """
    Interface for any preprocessing step.
    Must return the modified (df_inter, df_item).
    """
    @abstractmethod
    def process(self, df_inter: pd.DataFrame, df_item: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process the dataframes and return modified versions."""
        pass


class BaseEarlyPreprocessor(BasePreprocessor, ABC):
    """Preprocessor that filters users *before* interactions are fully loaded.

    Subclasses implement :meth:`select_users` which receives the full
    set of unique user IDs (gathered from raw files) and returns
    the subset to keep.  The loader calls this **before** reading or
    building interactions, avoiding OOM on large datasets.

    The regular :meth:`process` is still available as a fallback when a
    loader does not support early filtering.
    """

    @abstractmethod
    def select_users(self, all_user_ids: np.ndarray) -> np.ndarray:
        """Return the subset of *all_user_ids* to retain.

        Parameters
        ----------
        all_user_ids : np.ndarray
            Every unique user ID across all raw data sources.

        Returns
        -------
        np.ndarray
            User IDs to keep.
        """
        pass