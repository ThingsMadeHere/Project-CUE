# Tests for Universal Compositional Embedder

This directory contains the comprehensive test suite for the Universal Compositional Embedder with Cone-Based Semantics.

## Test Structure

The test suite is organized into the following categories:

- `test_models.py` - Tests for the core model components including encoders, projection heads, compositional layer, and cone embeddings
- `test_data.py` - Tests for data loading and preprocessing functionality
- `test_training.py` - Tests for training losses and the trainer class
- `test_utils.py` - Tests for utility functions and cone operations
- `test_integration.py` - Integration tests for the full system
- `conftest.py` - Pytest configuration and fixtures
- `run_tests.py` - Test runner script

## Running Tests

### Prerequisites

Install the required test dependencies:

```bash
pip install -r tests/requirements.txt
```

### Running All Tests

To run all tests:

```bash
python -m pytest tests/ -v
```

Or use the test runner script:

```bash
python tests/run_tests.py
```

### Running Specific Tests

To run tests for a specific module:

```bash
python -m pytest tests/test_models.py -v
```

To run tests with coverage:

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Markers

Some tests are marked with special markers:

- `slow`: Tests that take a long time to run
- `gpu`: Tests that require GPU resources

To skip slow tests:

```bash
python -m pytest tests/ -m "not slow"
```

## Test Coverage

The test suite aims to provide comprehensive coverage of:

1. **Model Components**: All encoder types, projection heads, compositional layer, and cone operations
2. **Data Pipeline**: Loading, preprocessing, and handling of multi-modal data
3. **Training Components**: Loss functions and training procedures
4. **Utility Functions**: Cone operations and helper functions
5. **Integration**: Full system functionality and cross-module interactions

## Adding New Tests

When adding new functionality to the project, please ensure to add corresponding tests in the appropriate test file:

- Model components → `test_models.py`
- Data functionality → `test_data.py`
- Training features → `test_training.py`
- Utility functions → `test_utils.py`
- Cross-module functionality → `test_integration.py`

Each test should follow the existing patterns and include appropriate assertions to verify the expected behavior.