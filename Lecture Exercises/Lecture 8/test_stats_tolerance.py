"""
Tolerance-based testing using numpy.testing.

Exercise 8.7: Use numpy.testing.assert_allclose for numerical tests
with allowable floating-point deviations.
"""

import numpy as np
import pytest
from stats import mean, variance, std, normalize


class TestToleranceBased:
    """Tolerance-based tests for statistical functions."""
    
    def test_mean_exact_result(self):
        """
        Verify mean([1, 2, 3, 4, 5]) == 3.0 with rtol=1e-12.
        """
        result = mean([1, 2, 3, 4, 5])
        np.testing.assert_allclose(result, 3.0, rtol=1e-12, atol=0)
    
    def test_mean_with_floats(self):
        """Verify mean with floating point inputs."""
        data = [1.1, 2.2, 3.3, 4.4]
        result = mean(data)
        expected = 2.75
        np.testing.assert_allclose(result, expected, rtol=1e-12)
    
    def test_bessel_correction_sample_variance(self):
        """
        Verify Bessel correction: generate random normal data with σ=2,
        compute sample variance (ddof=1), verify ≈ 4.0 with rtol=0.05.
        
        This tests that sample variance is an unbiased estimator of population variance.
        """
        rng = np.random.default_rng(42)
        data = rng.normal(0, 2, 10000)
        
        sample_var = variance(data, ddof=1)
        expected_var = 4.0
        
        # Allow 5% relative tolerance for random data
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.05)
    
    def test_normalization_mean_tolerance(self):
        """
        Verify that after normalize(data), mean is within 1e-12 of zero.
        """
        data = [1, 2, 3, 4, 5, 100, 200]
        normalized = normalize(data)
        result_mean = np.mean(normalized)
        
        np.testing.assert_allclose(result_mean, 0.0, atol=1e-12)
    
    def test_normalization_std_tolerance(self):
        """
        Verify that after normalize(data), std is within 1e-12 of 1.0.
        """
        data = np.linspace(0, 1000, 100)
        normalized = normalize(data)
        result_std = np.std(normalized)
        
        np.testing.assert_allclose(result_std, 1.0, atol=1e-12)
    
    def test_large_range_normalization(self):
        """Test normalization on data with very large range."""
        data = [-1e6, 0, 1e6]
        normalized = normalize(data)
        
        np.testing.assert_allclose(np.mean(normalized), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(normalized), 1.0, atol=1e-12)
    
    def test_small_scale_normalization(self):
        """Test normalization on data with very small values."""
        data = [1e-8, 2e-8, 3e-8]
        normalized = normalize(data)
        
        np.testing.assert_allclose(np.mean(normalized), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.std(normalized), 1.0, atol=1e-12)
    
    def test_std_squared_equals_variance_tolerance(self):
        """Verify std²(data) ≈ variance(data) using tolerance testing."""
        data = np.random.default_rng(123).normal(100, 15, 1000)
        
        std_val = std(data)
        var_val = variance(data)
        
        np.testing.assert_allclose(std_val ** 2, var_val, rtol=1e-12)
    
    def test_variance_ddof_relationship(self):
        """
        Verify relationship: variance(data, ddof=1) / variance(data, ddof=0) = n/(n-1)
        """
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        n = len(data)
        
        var_0 = variance(data, ddof=0)
        var_1 = variance(data, ddof=1)
        
        expected_ratio = n / (n - 1)
        actual_ratio = var_1 / var_0
        
        np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-12)
    
    def test_mean_invariance_precision(self):
        """
        Test that mean is stable for array with problematic floating point values.
        """
        data = [1/3, 1/3, 1/3]  # Problematic floating point values
        result = mean(data)
        
        # Should be very close to 1/3 but not exact due to float precision
        np.testing.assert_allclose(result, 1/3, rtol=1e-15)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
