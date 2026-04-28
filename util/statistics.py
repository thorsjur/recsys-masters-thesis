import numpy as np
from typing import Union, List

Numeric = Union[np.ndarray, List[float]]


def _as_array(values: Numeric) -> np.ndarray:
    return np.asarray(values)


def coefficient_of_variation(values: Numeric, percent: bool = True) -> float:
    """Calculate coefficient of variation."""
    arr = _as_array(values)
    if arr.size == 0:
        return 0.0
    m = np.mean(arr)
    if m == 0:
        return 0.0
    cv = np.std(arr, ddof=0) / m
    return float(cv * 100.0) if percent else float(cv)


def range_statistic(values: Numeric) -> float:
    """Calculate range."""
    arr = _as_array(values)
    return 0.0 if arr.size == 0 else float(np.max(arr) - np.min(arr))


def mean(values: Numeric) -> float:
    """Calculate arithmetic mean."""
    arr = _as_array(values)
    return 0.0 if arr.size == 0 else float(np.mean(arr))


def std(values: Numeric, ddof: int = 0) -> float:
    """Calculate standard deviation."""
    arr = _as_array(values)
    return 0.0 if arr.size == 0 else float(np.std(arr, ddof=ddof))


def basic_stats(values: Numeric, ddof: int = 0) -> dict:
    """Calculate mean, std, min, max, range, cv, and n for a dataset."""
    arr = _as_array(values)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "cv": 0.0, "n": 0}

    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": range_statistic(arr),
        "cv": coefficient_of_variation(arr),
        "n": len(arr),
    }
