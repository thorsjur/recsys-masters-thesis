from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple

class BaseSplitter(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split interactions into train, valid, and test sets."""
        pass