from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_analysis.atomic_file import find_item_file as _find_item_file
from data_analysis.atomic_file import load_item_dataframe as _load_item_dataframe


def find_item_file(dataset: str, base_path: str = "data/atomic_files") -> Path:
    return _find_item_file(dataset, base_path)


def load_item_dataframe(item_file: Path) -> pd.DataFrame:
    return _load_item_dataframe(item_file)
