"""
Property-based tests using Hypothesis.

Exercise 8.6: Verify statistical properties hold for any data:
- Shift invariance of mean
- Scale invariance of variance
- Normalization properties
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from stats import mean, variance, std, normalize


# Define strategies for generated data
reasonable_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False
)

lists_with_variance = st.lists(
    reasonable_floats,
    min_size=2,
    max_size=1000,
    unique=False
)

lists_with_different_elements = st.lists(
    reasonable_floats,
    min_size=2,
    max_size=1000,
    unique=True
).filter(lambda x: len(set(x)) > 1)


# ============================================================================
# PROPERTY 1: Shift Invariance of Mean
# ============================================================================

@given(data=lists_with_variance, shift=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False))
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
def test_mean_shift_invariant(data, shift):
    """
    Property: mean([x + c for x in data]) == mean(data) + c
    
    Shifting all values by a constant shifts the mean by that constant.
    """
    if len(data) == 0 or any(np.isnan(x) for x in data):
        return
    
    original_mean = mean(data)
    shifted_data = [x + shift for x in data]
    shifted_mean = mean(shifted_data)
    
    assert shifted_mean == pytest.approx(original_mean + shift, rel=1e-10, abs=1e-12)


# ============================================================================
# PROPERTY 2: Scale of Variance
# ============================================================================

@given(
    data=lists_with_different_elements,
    scale=st.floats(0.1, 1000, allow_nan=False, allow_infinity=False)
)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
def test_variance_scale_property(data, scale):
    """
    Property: variance([c*x for x in data]) == c² × variance(data)
    
    Scaling all values by a factor scales the variance by that factor squared.
    """
    if len(data) < 2 or any(np.isnan(x) for x in data):
        return
    
    try:
        original_var = variance(data)
    except ValueError:
        return
    
    scaled_data = [x * scale for x in data]
    scaled_var = variance(scaled_data)
    expected_var = scale ** 2 * original_var
    
    # Use relative tolerance for larger variances
    assert scaled_var == pytest.approx(expected_var, rel=1e-10, abs=1e-12)


# ============================================================================
# PROPERTY 3: Normalization Properties
# ============================================================================

@given(data=lists_with_different_elements)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
def test_normalize_mean_zero(data):
    """
    Property: After normalize(data), the result always has mean ≈ 0.
    """
    if len(data) < 2 or any(np.isnan(x) for x in data):
        return
    
    try:
        normalized = normalize(data)
    except ValueError:
        return
    
    result_mean = np.mean(normalized)
    assert result_mean == pytest.approx(0.0, abs=1e-12)


@given(data=lists_with_different_elements)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
def test_normalize_std_one(data):
    """
    Property: After normalize(data), the result always has std ≈ 1.
    """
    if len(data) < 2 or any(np.isnan(x) for x in data):
        return
    
    try:
        normalized = normalize(data)
    except ValueError:
        return
    
    result_std = np.std(normalized, ddof=0)
    assert result_std == pytest.approx(1.0, abs=1e-12)


# ============================================================================
# PROPERTY 4: Combined Shift and Scale
# ============================================================================

@given(
    data=lists_with_variance,
    shift=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    scale=st.floats(0.1, 100, allow_nan=False, allow_infinity=False)
)
@settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=50)
def test_mean_variance_combined_transformation(data, shift, scale):
    """
    Property: Verify mean and variance transform correctly under combined shift and scale.
    
    If Y = c*X + b, then:
    - E[Y] = c*E[X] + b
    - Var[Y] = c²*Var[X]
    """
    if len(data) < 2 or any(np.isnan(x) for x in data):
        return
    
    original_mean = mean(data)
    original_var = variance(data)
    
    transformed = [scale * x + shift for x in data]
    transformed_mean = mean(transformed)
    transformed_var = variance(transformed)
    
    expected_mean = scale * original_mean + shift
    expected_var = scale ** 2 * original_var
    
    assert transformed_mean == pytest.approx(expected_mean, rel=1e-10, abs=1e-12)
    assert transformed_var == pytest.approx(expected_var, rel=1e-10, abs=1e-12)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
