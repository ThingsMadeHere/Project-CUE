#!/usr/bin/env python3
"""
Test runner for Universal Compositional Embedder tests
"""
import pytest
import sys
import os

def run_tests():
    """Run all tests in the test suite"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Add project root to Python path
    sys.path.insert(0, project_root)
    
    # Change to the tests directory to run tests
    os.chdir(script_dir)
    
    # Run pytest with coverage and detailed output
    exit_code = pytest.main([
        '.',  # Run all tests in current directory
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--cov=src',  # Coverage for src module
        '--cov-report=html:coverage_report',  # HTML coverage report
        '--cov-report=term-missing',  # Terminal output with missing lines
        '-x',  # Stop after first failure
        '--disable-warnings'  # Suppress warnings
    ])
    
    return exit_code

if __name__ == "__main__":
    print("Running Universal Compositional Embedder tests...")
    print("=" * 50)
    
    exit_code = run_tests()
    
    print("=" * 50)
    if exit_code == 0:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    
    sys.exit(exit_code)