from abc import ABC, abstractmethod
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