# BiasX Testing Guide

This directory contains the test suite for the BiasX framework. The tests are organized into three main categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **System-level Tests**: Test the entire system as a whole

## Test Structure

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── test_analyzer.py  # Tests for Analyzer component
│   ├── test_calculator.py# Tests for Calculator component
│   ├── test_config.py    # Tests for configuration management
│   ├── test_dataset.py   # Tests for Dataset component
│   ├── test_explainer.py # Tests for Explainer component
│   ├── test_model.py     # Tests for Model component
│   └── test_types.py     # Tests for type definitions
│
├── integration/          # Tests for component interactions
│   ├── test_dataset_model.py     # Dataset to Model interaction
│   ├── test_end_to_end.py        # Complete pipeline testing
│   ├── test_explainer_calculator.py  # Explainer to Calculator interaction
│   └── test_model_explainer.py   # Model to Explainer interaction
│
├── system-level/         # System-level tests
│   ├── test_error_handling.py    # Error handling scenarios
│   ├── test_output_validation.py # Output validation tests
│   ├── test_performance.py       # Performance testing
│   └── test_system.py            # End-to-end system tests
│
├── data/                 # Test data and fixtures
│   ├── sample_images/    # Sample images for testing
│   └── sample_models/    # Sample models for testing
│
├── conftest.py           # Shared pytest fixtures
└── test_infrastructure.py # Test infrastructure and utility functions
```

## Prerequisites

Before running the tests, ensure you have:

1. Installed BiasX and its dependencies
2. Installed pytest and related testing packages

```bash
# Install BiasX in development mode
pip install -e .

# Install test dependencies
pip install pytest pytest-cov
```

## Running Tests

### Running All Tests

To run all tests with coverage reporting:

```bash
# From the project root directory
pytest --cov=biasx --cov-report=term --cov-report=html

# Or using the Makefile
make test
```

### Running Specific Test Categories

Run only unit tests:

```bash
pytest tests/unit/
```

Run only integration tests:

```bash
pytest tests/integration/
```

Run only system-level tests:

```bash
pytest tests/system-level/
```

### Running Specific Test Files

```bash
# Run a specific test file
pytest tests/unit/test_model.py

# Run a specific test case
pytest tests/unit/test_model.py::test_model_initialization
```

### Running Tests by Markers

Tests are marked with categories to allow selective execution:

```bash
# Run only integration tests using markers
pytest -m integration

# Run only model-related tests
pytest -m model

# Run only performance tests
pytest -m slow

# Run tests with multiple markers (AND)
pytest -m "integration and model"

# Run tests with either marker (OR)
pytest -m "unit or integration"
```

Available markers include:
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Tests that take longer to run
- `model`, `dataset`, `explainer`, `calculator`, `analyzer`: Component-specific tests
- `mocked`: Tests that use mocked components

## Configuration

The test suite uses fixtures defined in `conftest.py` to provide:
- Mock components for isolated testing
- Test data and models
- Utility functions

You can customize test behavior using environment variables:

```bash
# Use a specific model for testing
export BIASX_TEST_MODEL_PATH=/path/to/test_model.h5

# Run with increased verbosity
pytest -v

# Show output from tests (including print statements)
pytest -s
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. Place unit tests in the appropriate file under `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Place system-level tests in `tests/system-level/`
4. Add appropriate markers to categorize tests
5. Use fixtures from `conftest.py` when possible
6. Follow the naming convention: `test_<functionality>.py` for files and `test_<specific_case>` for functions

Example:

```python
import pytest

@pytest.mark.unit
@pytest.mark.model
def test_new_model_feature():
    # Test code here
    assert result == expected
```

## Troubleshooting

If you encounter issues running tests:

1. Ensure all dependencies are installed
2. Check for missing test data
3. Verify test environment configuration
4. Examine the test output for specific errors

For integration or system-level tests that require external resources, ensure those resources are available or properly mocked.
