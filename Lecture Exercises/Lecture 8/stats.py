"""
Statistical functions module.

This module provides core statistical functions: mean, variance, standard deviation,
and data normalization. All functions operate on sequences of numeric values and 
include comprehensive error checking.

Functions:
    mean: Compute arithmetic mean
    variance: Compute variance with adjustable degrees of freedom
    std: Compute standard deviation with adjustable degrees of freedom
    normalize: Standardize data to zero mean and unit variance

Examples:
    >>> data = [1, 2, 3, 4, 5]
    >>> mean(data)
    3.0
    >>> variance(data)
    2.0
    >>> normalize([1, 2, 3, 4, 5])
    array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])

Notes:
    All functions use NumPy for efficient computation on array-like inputs.
    Input data can be lists, tuples, or NumPy arrays.
"""

import numpy as np


def mean(data):
    """
    Compute the arithmetic mean of data.
    
    The mean is the sum of all values divided by the count. This function
    uses NumPy for fast computation and includes validation for empty or
    invalid data.
    
    Parameters
    ----------
    data : array-like
        Sequence of numeric values (list, tuple, or numpy array).
    
    Returns
    -------
    float
        Arithmetic mean of the data.
    
    Raises
    ------
    ValueError
        If data is empty or contains NaN values.
    TypeError
        If data cannot be converted to a numeric array.
    
    Examples
    --------
    >>> mean([1, 2, 3])
    2.0
    
    >>> mean([10.0, 20.0, 30.0])
    20.0
    
    >>> mean([])  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: data cannot be empty
    
    >>> mean([1, 2, np.nan])  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: data contains NaN values
    """
    data = np.asarray(data)
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    
    if np.any(np.isnan(data)):
        raise ValueError("data contains NaN values")
    
    return float(np.mean(data))


def variance(data, ddof=0):
    """
    Compute the variance of data.
    
    Variance measures the spread of data around the mean. The degrees of
    freedom parameter (ddof) adjusts the denominator: N - ddof. Setting
    ddof=1 computes the unbiased sample variance (Bessel's correction).
    
    Parameters
    ----------
    data : array-like
        Sequence of numeric values (list, tuple, or numpy array).
    ddof : int, optional
        Degrees of freedom adjustment. Default is 0 (population variance).
        Set to 1 for unbiased sample variance.
    
    Returns
    -------
    float
        Variance of the data with specified degrees of freedom.
    
    Raises
    ------
    ValueError
        If data is empty, contains NaN, or if len(data) <= ddof.
    TypeError
        If data cannot be converted to a numeric array.
    
    Examples
    --------
    >>> variance([1, 2, 3])
    0.6666666666666666
    
    >>> variance([1, 2, 3], ddof=1)
    1.0
    
    >>> variance([5, 5, 5])
    0.0
    """
    data = np.asarray(data)
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    
    if np.any(np.isnan(data)):
        raise ValueError("data contains NaN values")
    
    if len(data) <= ddof:
        raise ValueError(f"ddof ({ddof}) must be less than data length ({len(data)})")
    
    return float(np.var(data, ddof=ddof))


def std(data, ddof=0):
    """
    Compute the standard deviation of data.
    
    Standard deviation is the square root of variance. Provides an interpretable
    measure of spread in the same units as the original data.
    
    Parameters
    ----------
    data : array-like
        Sequence of numeric values (list, tuple, or numpy array).
    ddof : int, optional
        Degrees of freedom adjustment. Default is 0 (population).
        Set to 1 for unbiased sample standard deviation.
    
    Returns
    -------
    float
        Standard deviation of the data.
    
    Raises
    ------
    ValueError
        If data is empty, contains NaN, or if len(data) <= ddof.
    TypeError
        If data cannot be converted to a numeric array.
    
    Examples
    --------
    >>> std([1, 2, 3])
    0.816496580927726
    
    >>> std([1, 2, 3], ddof=1)
    1.0
    
    >>> std([10, 10, 10])
    0.0
    """
    data = np.asarray(data)
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    
    if np.any(np.isnan(data)):
        raise ValueError("data contains NaN values")
    
    if len(data) <= ddof:
        raise ValueError(f"ddof ({ddof}) must be less than data length ({len(data)})")
    
    return float(np.std(data, ddof=ddof))


def normalize(data):
    """
    Standardize data to zero mean and unit variance.
    
    Normalization (z-score standardization) transforms data so that it has
    mean 0 and standard deviation 1. Useful for comparing variables on
    different scales or for machine learning preprocessing.
    
    Formula: z = (x - mean(x)) / std(x)
    
    Parameters
    ----------
    data : array-like
        Sequence of numeric values (list, tuple, or numpy array).
    
    Returns
    -------
    ndarray
        Normalized data as a NumPy array with mean ≈ 0 and std ≈ 1.
    
    Raises
    ------
    ValueError
        If data is empty, contains NaN, or if std(data) == 0 (no variance).
    TypeError
        If data cannot be converted to a numeric array.
    
    Examples
    --------
    >>> result = normalize([1, 2, 3])
    >>> np.allclose(result.mean(), 0)
    True
    
    >>> result = normalize([10, 20, 30, 40, 50])
    >>> np.allclose(result.std(), 1.0)
    True
    
    >>> normalize([5, 5, 5])  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: cannot normalize data with zero standard deviation
    """
    data = np.asarray(data, dtype=float)
    
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    
    if np.any(np.isnan(data)):
        raise ValueError("data contains NaN values")
    
    data_std = std(data)
    if data_std == 0:
        raise ValueError("cannot normalize data with zero standard deviation")
    
    return (data - mean(data)) / data_std
