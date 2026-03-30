"""
Pytest test suite for stats module.

Exercise 8.5: Rewritten using pytest with:
- Plain assert statements
- pytest.approx for float comparisons
- @pytest.mark.parametrize for multiple datasets
- pytest.raises for error testing
"""

import pytest
import numpy as np
from stats import mean, variance, std, normalize


# ============================================================================
# EXERCISE 8.5: Mean Tests with Parametrization
# ============================================================================

@pytest.mark.parametrize("data,expected", [
    ([1, 2, 3], 2.0),
    ([10.0, 20.0, 30.0], 20.0),
    ([5, 5, 5], 5.0),
    ([-1, -2, -3], -2.0),
    ([0], 0.0),
    ([-5, 0, 5], 0.0),
    ([1], 1.0),
    ([100, 200], 150.0),
])
def test_mean_parametrized(data, expected):
    """Test mean across multiple datasets using parametrization."""
    assert mean(data) == pytest.approx(expected)


def test_mean_empty_raises():
    """Test mean raises ValueError for empty data."""
    with pytest.raises(ValueError, match="empty"):
        mean([])


def test_mean_nan_raises():
    """Test mean raises ValueError for NaN data."""
    with pytest.raises(ValueError, match="NaN"):
        mean([1, 2, np.nan])


def test_mean_numpy_array():
    """Test mean works with NumPy arrays."""
    data = np.array([1, 2, 3, 4, 5])
    assert mean(data) == pytest.approx(3.0)


# ============================================================================
# EXERCISE 8.5: Variance Tests
# ============================================================================

@pytest.mark.parametrize("data,ddof,expected", [
    ([1, 2, 3], 0, 2/3),
    ([1, 2, 3], 1, 1.0),
    ([1, 2, 3, 4, 5], 0, 2.0),
    ([1, 2, 3, 4, 5], 1, 2.5),
    ([5, 5, 5], 0, 0.0),
    ([-1, -2, -3], 0, 2/3),
])
def test_variance_parametrized(data, ddof, expected):
    """Test variance with various ddof values."""
    assert variance(data, ddof=ddof) == pytest.approx(expected)


def test_variance_empty_raises():
    """Test variance raises ValueError for empty data."""
    with pytest.raises(ValueError):
        variance([])


def test_variance_ddof_exceeds_len_raises():
    """Test variance raises ValueError when ddof >= len(data)."""
    with pytest.raises(ValueError, match="ddof"):
        variance([1, 2], ddof=2)


def test_variance_nan_raises():
    """Test variance raises ValueError for NaN data."""
    with pytest.raises(ValueError):
        variance([1, 2, np.nan])


# ============================================================================
# EXERCISE 8.5: Std Tests
# ============================================================================

def test_std_squared_equals_variance():
    """Verify that std(data)**2 ≈ variance(data)."""
    for data in [[1, 2, 3], [1, 2, 3, 4, 5], [-5, 0, 5]]:
        std_val = std(data)
        var_val = variance(data)
        assert (std_val ** 2) == pytest.approx(var_val, rel=1e-10)


def test_std_empty_raises():
    """Test std raises ValueError for empty data."""
    with pytest.raises(ValueError):
        std([])


def test_std_ddof_exceeds_len_raises():
    """Test std raises ValueError when ddof >= len(data)."""
    with pytest.raises(ValueError, match="ddof"):
        std([1, 2], ddof=2)


def test_std_nan_raises():
    """Test std raises ValueError for NaN data."""
    with pytest.raises(ValueError):
        std([1, np.nan])


# ============================================================================
# EXERCISE 8.5: Normalize Tests
# ============================================================================

@pytest.mark.parametrize("data", [
    [1, 2, 3, 4, 5],
    [0, 100, 200, 300],
    [-10, -5, 0, 5, 10],
    [0.1, 0.2, 0.3],
    [-100, 100],
])
def test_normalize_mean_zero(data):
    """Test that normalized data has mean ≈ 0."""
    normalized = normalize(data)
    assert np.mean(normalized) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("data", [
    [1, 2, 3, 4, 5],
    [0, 100, 200, 300],
    [-10, -5, 0, 5, 10],
])
def test_normalize_std_one(data):
    """Test that normalized data has std ≈ 1."""
    normalized = normalize(data)
    assert np.std(normalized) == pytest.approx(1.0, abs=1e-12)


def test_normalize_identical_raises():
    """Test that normalizing identical values raises ValueError."""
    with pytest.raises(ValueError, match="zero"):
        normalize([5, 5, 5])


def test_normalize_empty_raises():
    """Test normalize raises ValueError for empty data."""
    with pytest.raises(ValueError):
        normalize([])


def test_normalize_nan_raises():
    """Test normalize raises ValueError for NaN data."""
    with pytest.raises(ValueError):
        normalize([1, 2, np.nan])


def test_normalize_returns_ndarray():
    """Test that normalize returns a NumPy array."""
    result = normalize([1, 2, 3])
    assert isinstance(result, np.ndarray)
