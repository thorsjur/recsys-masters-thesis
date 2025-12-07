"""
Statistical utility functions for recommendation system evaluation.

This module provides common statistical measures used in stability analysis
and performance evaluation, with a focus on coefficient of variation and
dispersion metrics.
"""

import numpy as np
from typing import Union, List


def coefficient_of_variation(values: Union[np.ndarray, List[float]], 
                             percent: bool = True) -> float:
    """
    Calculate the coefficient of variation (CV) of a dataset.
    
    The coefficient of variation is the ratio of the standard deviation
    to the mean, often expressed as a percentage. It measures relative
    variability and is useful for comparing dispersion across datasets
    with different scales or units.
    
    CV = (σ / μ) * 100%
    
    Args:
        values: Array or list of numerical values
        percent: If True, return as percentage (default). If False, return as ratio.
        
    Returns:
        Coefficient of variation. Returns 0.0 if mean is zero or data is empty.
        
    Examples:
        >>> coefficient_of_variation([1, 2, 3, 4, 5])
        47.14
        >>> coefficient_of_variation([1, 2, 3, 4, 5], percent=False)
        0.4714
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    
    mean = np.mean(values)
    
    if mean == 0:
        return 0.0
    
    std = np.std(values, ddof=0)  # Population std (ddof=0) for consistency
    cv = std / mean
    
    return float(cv * 100.0) if percent else float(cv)


def range_statistic(values: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate the range (max - min) of a dataset.
    
    Args:
        values: Array or list of numerical values
        
    Returns:
        Range (difference between maximum and minimum)
        
    Examples:
        >>> range_statistic([1, 2, 3, 4, 5])
        4.0
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.max(values) - np.min(values))


def mean(values: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate the arithmetic mean.
    
    Args:
        values: Array or list of numerical values
        
    Returns:
        Mean value, or 0.0 if empty
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def std(values: Union[np.ndarray, List[float]], ddof: int = 0) -> float:
    """
    Calculate the standard deviation.
    
    Args:
        values: Array or list of numerical values
        ddof: Delta degrees of freedom (default 0 for population std)
        
    Returns:
        Standard deviation, or 0.0 if empty
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return float(np.std(values, ddof=ddof))


def basic_stats(values: Union[np.ndarray, List[float]], 
               ddof: int = 0) -> dict:
    """
    Calculate basic statistical measures for a dataset.
    
    Args:
        values: Array or list of numerical values
        ddof: Delta degrees of freedom for std calculation
        
    Returns:
        Dictionary containing:
            - mean: Arithmetic mean
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - range: max - min
            - cv: Coefficient of variation (%)
            - n: Number of values
    """
    values = np.asarray(values)
    if values.size == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'cv': 0.0,
            'n': 0
        }
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=ddof)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': range_statistic(values),
        'cv': coefficient_of_variation(values, percent=True),
        'n': len(values)
    }
