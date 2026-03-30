"""
Profiling exercise: Compare pure Python vs NumPy implementations.

Exercise 8.10: Performance comparison and profiling using:
- %timeit for baseline benchmarking
- cProfile for identifying bottlenecks
- line_profiler for line-by-line analysis
"""

import numpy as np
from stats import mean, variance, std


# ============================================================================
# MEAN IMPLEMENTATIONS: Pure Python vs NumPy
# ============================================================================

def mean_slow(data):
    """
    Pure Python implementation of mean.
    
    Iterates through data element by element with no optimization.
    This is intentionally slow for profiling comparison.
    
    Parameters
    ----------
    data : list
        Sequence of numeric values
    
    Returns
    -------
    float
        Arithmetic mean
    """
    total = 0
    count = 0
    for value in data:
        total = total + value
        count = count + 1
    return total / count


def mean_fast(data):
    """
    NumPy-optimized implementation of mean.
    
    Uses NumPy's vectorized operations for fast computation.
    
    Parameters
    ----------
    data : array-like
        Sequence of numeric values
    
    Returns
    -------
    float
        Arithmetic mean
    """
    return float(np.mean(data))


# ============================================================================
# VARIANCE IMPLEMENTATIONS: Pure Python vs NumPy
# ============================================================================

def variance_slow(data, ddof=0):
    """
    Pure Python implementation of variance.
    
    Computes variance using explicit loops and accumulation.
    This is intentionally slow for profiling comparison.
    
    Parameters
    ----------
    data : list
        Sequence of numeric values
    ddof : int, optional
        Degrees of freedom adjustment
    
    Returns
    -------
    float
        Variance of the data
    """
    # First pass: compute mean
    m = mean_slow(data)
    
    # Second pass: compute sum of squared deviations
    sum_sq_dev = 0
    count = 0
    for value in data:
        deviation = value - m
        sum_sq_dev = sum_sq_dev + (deviation * deviation)
        count = count + 1
    
    # Adjust by degrees of freedom
    return sum_sq_dev / (count - ddof)


def variance_fast(data, ddof=0):
    """
    NumPy-optimized implementation of variance.
    
    Uses NumPy's vectorized operations for fast computation.
    
    Parameters
    ----------
    data : array-like
        Sequence of numeric values
    ddof : int, optional
        Degrees of freedom adjustment
    
    Returns
    -------
    float
        Variance of the data
    """
    return float(np.var(data, ddof=ddof))


# ============================================================================
# PROFILING UTILITIES
# ============================================================================

def benchmark_implementations(n_values):
    """
    Create test data and benchmark all implementations.
    
    Parameters
    ----------
    n_values : int
        Size of test data
    
    Returns
    -------
    data : list
        Generated test data
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking with {n_values:,} values")
    print(f"{'='*70}")
    
    # Generate test data
    rng = np.random.default_rng(42)
    data = rng.normal(100, 15, n_values).tolist()
    
    return data


def compare_mean_implementations(data):
    """
    Compare mean_slow vs mean_fast.
    
    Parameters
    ----------
    data : list
        Input data
    """
    print("\nMean Comparison:")
    print("-" * 70)
    
    # Verify results match
    result_slow = mean_slow(data)
    result_fast = mean_fast(data)
    
    print(f"  mean_slow: {result_slow:.6f}")
    print(f"  mean_fast: {result_fast:.6f}")
    print(f"  Match: {np.isclose(result_slow, result_fast)}")


def compare_variance_implementations(data):
    """
    Compare variance_slow vs variance_fast.
    
    Parameters
    ----------
    data : list
        Input data
    """
    print("\nVariance Comparison:")
    print("-" * 70)
    
    # Verify results match
    result_slow = variance_slow(data, ddof=0)
    result_fast = variance_fast(data, ddof=0)
    
    print(f"  variance_slow (ddof=0): {result_slow:.6f}")
    print(f"  variance_fast (ddof=0): {result_fast:.6f}")
    print(f"  Match: {np.isclose(result_slow, result_fast)}")
    
    # Also test with ddof=1
    result_slow_1 = variance_slow(data, ddof=1)
    result_fast_1 = variance_fast(data, ddof=1)
    
    print(f"  variance_slow (ddof=1): {result_slow_1:.6f}")
    print(f"  variance_fast (ddof=1): {result_fast_1:.6f}")
    print(f"  Match: {np.isclose(result_slow_1, result_fast_1)}")


if __name__ == '__main__':
    # Benchmark with increasing data sizes
    for n in [1000, 10000, 100000]:
        data = benchmark_implementations(n)
        compare_mean_implementations(data)
        compare_variance_implementations(data)
    
    print("\n" + "="*70)
    print("To run profiling:")
    print("="*70)
    print("1. Use %timeit in Jupyter:")
    print("   %timeit mean_slow(data)")
    print("   %timeit mean_fast(data)")
    print("")
    print("2. Use cProfile:")
    print("   python -m cProfile -s cumulative profiling.py")
    print("")
    print("3. Use line_profiler:")
    print("   kernprof -l -v profiling.py")
    print("")
