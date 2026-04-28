from abc import ABC, abstractmethod
from typing import Optional, Set
import numpy as np
import pandas as pd


class BasePreprocessor(ABC):
    """
    Interface for preprocessing steps.
    Returns the modified (df_inter, df_item).
    """

    @abstractmethod
    def process(self, df_inter: pd.DataFrame, df_item: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Process the dataframes and return modified versions."""
        pass


class BaseEarlyPreprocessor(BasePreprocessor, ABC):
    """Preprocessor that filters users before interactions are fully loaded."""

    @abstractmethod
    def select_users(self, all_user_ids: np.ndarray) -> np.ndarray:
        """Return the subset of all user ids to retain."""
        pass
