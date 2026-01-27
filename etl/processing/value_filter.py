from typing import List, Any, Optional, Tuple, TypeAlias, Union
import pandas as pd
from etl.processing.base_preprocessor import BasePreprocessor

Number: TypeAlias = Union[int, float]
Range: TypeAlias = Tuple[Number, Number]


class ValueFilter(BasePreprocessor):
    def __init__(
        self,
        col_name: str,
        valid_values: Optional[List[Any]] = None,
        valid_range: Optional[Range] = None,
    ) -> None:
        self.col_name = col_name
        self.valid_values = valid_values
        self.valid_range = valid_range
        
    def _isNumber(self, value):
        return isinstance(value, (int, float))

    def _normalize(self, value):
        if self._isNumber(value):
            return float(value)
        return value

    def process(self, df_inter: pd.DataFrame, df_item: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert self.valid_values is not None or self.valid_range is not None, "Either valid_values or valid_range must be provided."
        print(f"[{self.__class__.__name__}] Filtering {self.col_name} by valid values/range...")

        assert self.col_name in df_inter.columns, f"Column {self.col_name} not found in interactions DataFrame"

        initial_len = len(df_inter)
        col = df_inter[self.col_name]

        mask = pd.Series([True] * len(df_inter), index=df_inter.index)

        # Categorical case (needs normalization for consistent comparison)
        if self.valid_values is not None:
            normalized_valids = set(self._normalize(v) for v in self.valid_values)
            mask &= col.apply(lambda x: self._normalize(x) in normalized_valids)

        # Value range case
        if self.valid_range is not None:
            min, max = self.valid_range
            assert len(self.valid_range) == 2, "valid_range must be a tuple of (min, max)"
            assert self._isNumber(min) and self._isNumber(max), "valid_range values must be numeric"
            assert min < max, "valid_range min must be less than max"
            
            mask &= col.apply(lambda x: min <= float(x) <= max)

        df_inter_filtered = df_inter[mask].copy()
        print(f"Dropped {initial_len - len(df_inter_filtered)} interactions.")

        return df_inter_filtered, df_item
