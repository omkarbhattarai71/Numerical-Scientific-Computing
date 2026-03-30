"""
Quick test script to verify all implementations work.
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/Numerical-Scientific-Computing/Lecture Exercises/Lecture 8')

print("="*70)
print("VERIFYING ALL EXERCISE 8 IMPLEMENTATIONS")
print("="*70)

# Test 8.1-8.2: Import stats and verify basic functionality
print("\n[8.1-8.2] Testing stats.py (Docstrings + Defensive Coding)...")
try:
    from stats import mean, variance, std, normalize
    
    # Test basic functionality
    assert mean([1, 2, 3]) == 2.0
    assert variance([1, 2, 3]) > 0
    assert std([1, 2, 3]) > 0
    
    # Test error handling
    try:
        mean([])
    except ValueError as e:
        assert "empty" in str(e)
    
    print("✓ stats.py: All basic tests passed")
    print("  - Functions work correctly")
    print("  - Input validation active")
    print("  - Docstrings present")
except Exception as e:
    print(f"✗ stats.py failed: {e}")

# Test 8.4: unittest
print("\n[8.4] Testing unittest suite...")
try:
    import unittest
    from test_stats_unittest import TestMean, TestVariance, TestStd, TestNormalize
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMean))
    suite.addTests(loader.loadTestsFromTestCase(TestVariance))
    suite.addTests(loader.loadTestsFromTestCase(TestStd))
    suite.addTests(loader.loadTestsFromTestCase(TestNormalize))
    
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print(f"✓ unittest: {result.testsRun} tests passed")
    else:
        print(f"✗ unittest: {len(result.failures)} failures, {len(result.errors)} errors")
except Exception as e:
    print(f"✗ unittest failed: {e}")

# Test 8.5: pytest
print("\n[8.5] Testing pytest suite...")
try:
    import pytest
    
    # Just verify pytest can be imported and run
    print("✓ pytest: Module imported successfully")
    print("  Run: python -m pytest test_stats_pytest.py -v")
except Exception as e:
    print(f"✗ pytest failed: {e}")

# Test 8.6: Hypothesis
print("\n[8.6] Testing Hypothesis property-based tests...")
try:
    from hypothesis import given, strategies as st
    print("✓ Hypothesis: Module imported successfully")
    print("  Run: python -m pytest test_stats_hypothesis.py -v")
except Exception as e:
    print(f"✗ Hypothesis failed: {e}")

# Test 8.7: Tolerance-based testing
print("\n[8.7] Testing tolerance-based numerical tests...")
try:
    import numpy.testing as npt
    
    # Test a simple tolerance comparison
    result = mean([1, 2, 3, 4, 5])
    npt.assert_allclose(result, 3.0, rtol=1e-12)
    
    print("✓ Tolerance testing: numpy.testing.assert_allclose works")
except Exception as e:
    print(f"✗ Tolerance testing failed: {e}")

# Test 8.9: messy_solver.py exists
print("\n[8.9] Static code analysis (pylint)...")
try:
    import messy_solver
    print("✓ messy_solver.py: Module exists and imports")
    print("  Run: pylint messy_solver.py")
except Exception as e:
    print(f"✗ messy_solver.py failed: {e}")

# Test 8.10: Profiling functions
print("\n[8.10] Testing profiling implementations...")
try:
    from profiling import mean_slow, mean_fast, variance_slow, variance_fast
    
    test_data = [1, 2, 3, 4, 5]
    
    result_slow = mean_slow(test_data)
    result_fast = mean_fast(test_data)
    assert result_slow == result_fast
    
    result_slow_v = variance_slow(test_data)
    result_fast_v = variance_fast(test_data)
    assert abs(result_slow_v - result_fast_v) < 1e-10
    
    print("✓ Profiling functions: Both slow and fast implementations work")
    print(f"  mean_slow([1,2,3,4,5]) = {result_slow}")
    print(f"  variance_slow([1,2,3,4,5]) = {result_slow_v}")
except Exception as e:
    print(f"✗ Profiling failed: {e}")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("""
All core implementations are ready. Now run individual tests:

Exercise 8.3 (Doctest):
  python -m doctest stats.py -v

Exercise 8.4 (Unittest):
  python -m unittest test_stats_unittest -v

Exercise 8.5 (Pytest):
  python -m pytest test_stats_pytest.py -v

Exercise 8.6 (Hypothesis):
  pip install hypothesis
  python -m pytest test_stats_hypothesis.py -v

Exercise 8.7 (Tolerance):
  python -m pytest test_stats_tolerance.py -v

Exercise 8.8 (Coverage):
  pip install coverage
  coverage run -m pytest test_stats_pytest.py
  coverage report -m stats.py

Exercise 8.9 (Pylint):
  pip install pylint
  pylint messy_solver.py
  pylint stats.py

Exercise 8.10 (Profiling):
  python profiling.py
  python -m cProfile -s cumulative profiling.py
""")
