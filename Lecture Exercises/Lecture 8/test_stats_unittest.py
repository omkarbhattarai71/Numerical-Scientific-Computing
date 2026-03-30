"""
Unit tests for stats module using unittest framework.

Exercises 8.4: Complete test suite covering:
- mean: empty data, identical values, negative numbers, NaN values
- variance: ddof=0, ddof=1, ddof >= len(data) error
- std: relationship to variance
- normalize: mean ≈ 0, std ≈ 1 after normalization
"""

import unittest
import numpy as np
from stats import mean, variance, std, normalize


class TestMean(unittest.TestCase):
    """Test suite for the mean function."""
    
    def test_simple_mean(self):
        """Test mean of simple integer list."""
        self.assertEqual(mean([1, 2, 3]), 2.0)
    
    def test_floating_point(self):
        """Test mean with floating point values."""
        result = mean([1.5, 2.5, 3.5])
        self.assertAlmostEqual(result, 2.5)
    
    def test_identical_values(self):
        """Test mean when all values are identical."""
        self.assertEqual(mean([5, 5, 5, 5]), 5.0)
    
    def test_negative_numbers(self):
        """Test mean with negative numbers."""
        self.assertEqual(mean([-1, -2, -3]), -2.0)
    
    def test_mixed_positive_negative(self):
        """Test mean with mixed positive and negative values."""
        result = mean([-5, 0, 5])
        self.assertAlmostEqual(result, 0.0)
    
    def test_single_value(self):
        """Test mean of a single value."""
        self.assertEqual(mean([42]), 42.0)
    
    def test_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        with self.assertRaises(ValueError) as context:
            mean([])
        self.assertIn("empty", str(context.exception))
    
    def test_nan_raises(self):
        """Test that NaN data raises ValueError."""
        with self.assertRaises(ValueError) as context:
            mean([1, 2, np.nan])
        self.assertIn("NaN", str(context.exception))
    
    def test_numpy_array(self):
        """Test mean works with NumPy arrays."""
        data = np.array([1, 2, 3, 4, 5])
        self.assertEqual(mean(data), 3.0)


class TestVariance(unittest.TestCase):
    """Test suite for the variance function."""
    
    def test_population_variance(self):
        """Test population variance (ddof=0)."""
        result = variance([1, 2, 3])
        expected = np.var([1, 2, 3], ddof=0)
        self.assertAlmostEqual(result, expected)
    
    def test_sample_variance(self):
        """Test sample variance (ddof=1, Bessel's correction)."""
        result = variance([1, 2, 3], ddof=1)
        expected = np.var([1, 2, 3], ddof=1)
        self.assertAlmostEqual(result, expected)
    
    def test_identical_values_zero_variance(self):
        """Test that identical values have zero variance."""
        self.assertEqual(variance([5, 5, 5, 5]), 0.0)
    
    def test_negative_numbers(self):
        """Test variance with negative numbers."""
        result = variance([-5, -3, -1])
        self.assertGreater(result, 0)
    
    def test_ddof_exceeds_data_raises(self):
        """Test that ddof >= len(data) raises ValueError."""
        with self.assertRaises(ValueError) as context:
            variance([1, 2], ddof=2)
        self.assertIn("ddof", str(context.exception))
    
    def test_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        with self.assertRaises(ValueError):
            variance([])
    
    def test_nan_raises(self):
        """Test that NaN data raises ValueError."""
        with self.assertRaises(ValueError):
            variance([1, 2, np.nan])
    
    def test_numpy_array(self):
        """Test variance works with NumPy arrays."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = variance(data)
        expected = np.var(data)
        self.assertAlmostEqual(result, expected)


class TestStd(unittest.TestCase):
    """Test suite for the std function."""
    
    def test_simple_std(self):
        """Test standard deviation of simple data."""
        result = std([1, 2, 3])
        expected = np.std([1, 2, 3])
        self.assertAlmostEqual(result, expected)
    
    def test_std_squared_equals_variance(self):
        """Verify that std(data)**2 ≈ variance(data)."""
        data = [1, 2, 3, 4, 5]
        std_val = std(data)
        var_val = variance(data)
        self.assertAlmostEqual(std_val ** 2, var_val, places=10)
    
    def test_std_squared_equals_variance_ddof1(self):
        """Verify that std(data, ddof=1)**2 ≈ variance(data, ddof=1)."""
        data = [1, 2, 3, 4, 5]
        std_val = std(data, ddof=1)
        var_val = variance(data, ddof=1)
        self.assertAlmostEqual(std_val ** 2, var_val, places=10)
    
    def test_identical_values_zero_std(self):
        """Test that identical values have zero standard deviation."""
        self.assertEqual(std([10, 10, 10]), 0.0)
    
    def test_negative_numbers(self):
        """Test std with negative numbers."""
        result = std([-5, -3, -1])
        self.assertGreater(result, 0)
    
    def test_ddof_exceeds_data_raises(self):
        """Test that ddof >= len(data) raises ValueError."""
        with self.assertRaises(ValueError):
            std([1, 2], ddof=2)
    
    def test_empty_data_raises(self):
        """Test that empty data raises ValueError."""
        with self.assertRaises(ValueError):
            std([])


class TestNormalize(unittest.TestCase):
    """Test suite for the normalize function."""
    
    def test_normalize_mean_close_to_zero(self):
        """Verify that normalized data has mean ≈ 0."""
        data = [1, 2, 3, 4, 5]
        normalized = normalize(data)
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=10)
    
    def test_normalize_std_close_to_one(self):
        """Verify that normalized data has std ≈ 1."""
        data = [1, 2, 3, 4, 5]
        normalized = normalize(data)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=10)
    
    def test_normalize_returns_ndarray(self):
        """Test that normalize returns a NumPy array."""
        result = normalize([1, 2, 3])
        self.assertIsInstance(result, np.ndarray)
    
    def test_normalize_large_range(self):
        """Test normalization with large range of values."""
        data = [0, 100, 200, 300, 400]
        normalized = normalize(data)
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=10)
    
    def test_normalize_negative_values(self):
        """Test normalization with negative values."""
        data = [-10, -5, 0, 5, 10]
        normalized = normalize(data)
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=10)
    
    def test_normalize_identical_values_raises(self):
        """Test that normalizing identical values raises ValueError."""
        with self.assertRaises(ValueError) as context:
            normalize([5, 5, 5])
        self.assertIn("zero", str(context.exception))
    
    def test_normalize_empty_raises(self):
        """Test that empty data raises ValueError."""
        with self.assertRaises(ValueError):
            normalize([])
    
    def test_normalize_nan_raises(self):
        """Test that NaN data raises ValueError."""
        with self.assertRaises(ValueError):
            normalize([1, 2, np.nan])


if __name__ == '__main__':
    unittest.main()
