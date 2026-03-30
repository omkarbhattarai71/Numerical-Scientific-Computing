"""
Lecture 8: Complete Guide to Software Engineering Best Practices

This script demonstrates all exercises 8.1-8.10 and provides commands to run each.
"""

import subprocess
import sys
import os

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0


def main():
    """Run demonstrations for all exercises."""
    
    # Change to script directory
    os.chdir(SCRIPT_DIR)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  Lecture 8: Complete Exercise Solutions                     ║
║                                                                              ║
║  Exercises 8.1-8.10: Professional Scientific Python Testing & Profiling     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    exercises = [
        ("EXERCISE 8.3: DOCTEST", 
         f"{sys.executable} -m doctest stats.py -v",
         "Running doctest on stats.py"),
        
        ("EXERCISE 8.4: UNITTEST",
         f"{sys.executable} -m unittest test_stats_unittest -v",
         "Running unittest test suite"),
        
        ("EXERCISE 8.5: PYTEST",
         f"{sys.executable} -m pytest test_stats_pytest.py -v",
         "Running pytest test suite"),
        
        ("EXERCISE 8.6: HYPOTHESIS",
         f"{sys.executable} -m pytest test_stats_hypothesis.py -v --tb=short",
         "Running hypothesis property-based tests"),
        
        ("EXERCISE 8.7: TOLERANCE-BASED TESTING",
         f"{sys.executable} -m pytest test_stats_tolerance.py -v",
         "Running tolerance-based tests"),
        
        ("EXERCISE 8.8: CODE COVERAGE",
         f"{sys.executable} -m coverage run -m pytest test_stats_pytest.py && {sys.executable} -m coverage report -m stats.py",
         "Running coverage analysis"),
        
        ("EXERCISE 8.9: PYLINT - MESSY SOLVER",
         f"{sys.executable} -m pylint messy_solver.py --disable=all --enable=E,W",
         "Analyzing messy_solver.py with pylint"),
        
        ("EXERCISE 8.9: PYLINT - STATS.PY",
         f"{sys.executable} -m pylint stats.py --disable=all --enable=E",
         "Analyzing stats.py with pylint"),
        
        ("EXERCISE 8.10: PROFILING",
         f"{sys.executable} profiling.py",
         "Running profiling comparison"),
    ]
    
    results = {}
    
    for exercise_name, command, description in exercises:
        try:
            success = run_command(command, description)
            results[exercise_name] = "✓ PASS" if success else "✗ FAIL"
        except Exception as e:
            print(f"Error running {exercise_name}: {e}")
            results[exercise_name] = "✗ ERROR"
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*70}")
    for exercise, status in results.items():
        print(f"{status}  {exercise}")
    
    print(f"\n{'='*70}")
    print("INDIVIDUAL EXERCISE COMMANDS")
    print(f"{'='*70}")
    print("""
Exercise 8.1 - Docstrings:
    Review stats.py for NumPy-style docstrings

Exercise 8.2 - Defensive Coding:
    Review stats.py for input validation in mean(), variance(), etc.

Exercise 8.3 - Doctest:
    python -m doctest stats.py -v

Exercise 8.4 - Unittest:
    python -m unittest test_stats_unittest -v

Exercise 8.5 - Pytest:
    python -m pytest test_stats_pytest.py -v

Exercise 8.6 - Hypothesis:
    python -m pytest test_stats_hypothesis.py -v

Exercise 8.7 - Tolerance Testing:
    python -m pytest test_stats_tolerance.py -v

Exercise 8.8 - Coverage:
    coverage run -m pytest test_stats_pytest.py
    coverage report -m stats.py
    coverage html  # For detailed HTML report

Exercise 8.9 - Pylint:
    pylint messy_solver.py  # Deliberately poor code
    pylint stats.py         # Your implementation

Exercise 8.10 - Profiling:
    python profiling.py
    python -m cProfile -s cumulative profiling.py
    kernprof -l -v profiling.py
    """)


if __name__ == '__main__':
    main()
