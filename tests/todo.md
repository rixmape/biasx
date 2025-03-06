# BiasX Testing Implementation Roadmap

This document outlines a three-phase approach to implementing tests for the BiasX package using pytest. Each phase builds upon the previous one, starting with basic setup and working toward comprehensive testing of complex components.

## Phase 1: Foundation Setup and Basic Testing

The initial phase focuses on establishing the testing infrastructure and implementing basic unit tests for simpler components.

### 1.1 Directory Structure and Configuration

- Create a `tests/` directory at the root level of the BiasX package
- Set up the following structure:

  ```plaintext
  tests/
  ├── conftest.py             # Global fixtures
  ├── test_utils.py           # Tests for utility functions
  ├── test_types.py           # Tests for type definitions
  ├── test_config.py          # Tests for configuration handling
  ├── data/                   # Test data directory
  │   ├── sample_images/      # Sample images for testing
  │   ├── sample_models/      # Minimal TensorFlow models
  │   └── test_configs/       # Test configuration files
  └── fixtures/               # Module-specific fixtures
      ├── model_fixtures.py
      ├── dataset_fixtures.py
      └── explainer_fixtures.py
  ```

- Create a `pytest.ini` file at the project root with initial settings:

  ```ini
  [pytest]
  testpaths = tests
  python_files = test_*.py
  python_classes = Test*
  python_functions = test_*
  markers =
      unit: mark a test as a unit test
      integration: mark a test as an integration test
      slow: mark a test as slow (useful for CI settings)
  ```

### 1.2 Core Fixtures and Test Utilities

- In `conftest.py`, implement:
  - Fixtures for configuration options
  - Simple mock classes for external dependencies
  - Path handling for test resources
- Create small test data samples:
  - Simplified JSON configuration files
  - Minimal image files (could be single-color 10x10 images)
  - Basic test data structures

### 1.3 Initial Unit Tests

- Test utility functions from `utils.py`:
  - `get_json_config()` with mocked file system
  - `get_cache_dir()` functionality
  - Path resolution functions
- Test type definitions from `types.py`:
  - Test enum conversions
  - Test box calculations and properties
  - Validate data class functionality

### 1.4 Configuration System Tests

- Implement tests for the `Config` class:
  - Test loading from dictionaries
  - Test loading from JSON files (using temporary files)
  - Test default value application
  - Test enum conversion in config values
  - Test validation of required fields

## Phase 2: Component Testing with Mocks

The second phase focuses on testing individual components with mock dependencies to ensure isolation.

### 2.1 Enhanced Test Infrastructure

- Create more sophisticated fixtures:
  - Mock TensorFlow model that returns predictable outputs
  - Mock dataset that yields predefined batches
  - Mock landmarker that returns predefined facial features
- Implement parametrization utilities for test reuse across different scenarios
- Set up test coverage reporting with pytest-cov

### 2.2 Dataset Processing Tests

- Test the `Dataset` class:
  - Test batch generation with mocked file reading
  - Test preprocessing functions with small sample images
  - Test metadata extraction from predefined dataframes
  - Verify correct application of transformations (resize, color conversion)
- Use monkeypatch to intercept file operations and dataset loading

### 2.3 Model Handling Tests

- Test the `Model` class:
  - Mock TensorFlow load_model to return a simple model
  - Test prediction functionality with synthetic inputs
  - Test probability processing and class conversion
  - Verify batch handling for different input shapes
- Create a minimal TensorFlow model for testing:

  ```python
  def create_test_model():
      """Create a minimal gender classification model for testing."""
      inputs = tf.keras.layers.Input(shape=(10, 10, 1))
      x = tf.keras.layers.Flatten()(inputs)
      x = tf.keras.layers.Dense(10, activation='relu')(x)
      outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
      return tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

### 2.4 Explainer Component Tests

- Create isolated tests for the `ClassActivationMapper`:
  - Test heatmap generation with a mock model
  - Test heatmap processing and thresholding
  - Verify bounding box creation
- Test the `FacialLandmarker` class:
  - Mock MediaPipe outputs
  - Test landmark to bounding box conversion
  - Test feature mapping
- Test the main `Explainer` class with mocked sub-components

## Phase 3: Integration Testing and Advanced Components

The final phase targets the most complex components and their integration points.

### 3.1 Calculator Testing Strategy

- Implement detailed tests for the `Calculator` class:
  - Test feature bias calculation with synthetic explanations
  - Test disparity score calculations
  - Verify correct handling of gender-based statistics
  - Validate mathematical formulas with hard-coded expected values
- Use parameterized tests to evaluate edge cases:
  - No misclassifications
  - All misclassifications
  - Extreme bias scenarios

### 3.2 Analyzer Integration Tests

- Test the `BiasAnalyzer` class with mocked components:
  - Test batch processing
  - Test result aggregation
  - Verify correct component configuration and initialization
- Implement end-to-end workflow tests using simplified inputs:
  - Create a test that processes a small batch through the entire pipeline
  - Validate the structure of the final `AnalysisResult`
  - Check for correct handling of empty inputs and edge cases

### 3.3 Coverage Analysis and Enhancement

- Generate coverage reports for the entire package
- Identify uncovered code paths and high-complexity areas
- Add targeted tests for areas with insufficient coverage
- Document intentionally untested code (if any)
- Set up test parameterization for comprehensive feature testing:

  ```python
  @pytest.mark.parametrize("feature", list(FacialFeature))
  @pytest.mark.parametrize("gender", list(Gender))
  def test_feature_bias_calculation(feature, gender, calculator):
      # Test bias calculation for every feature and gender combination
      ...
  ```

### 3.4 Advanced Mock Strategies

- Implement custom mock classes for complex external dependencies:
  - Create a `MockMediaPipe` class that simulates face landmark detection
  - Develop a `MockClassActivationMap` that generates synthetic heatmaps
  - Build a `MockModelPredictor` that provides controlled classification outputs
- Create factory functions for generating test data with controlled properties:

  ```python
  def create_test_explanation(
      gender=Gender.MALE,
      prediction=Gender.FEMALE,
      confidence=0.8,
      activated_features=None
  ):
      """Create a test explanation with specified properties."""
      # Implementation that creates a synthetic Explanation object
      ...
  ```

## Implementation Notes

1. **Test Isolation**: Ensure each test can run independently by avoiding shared state and using proper fixture scoping
2. **Mock External APIs**: All external library calls should be mocked to prevent network requests and ensure reproducibility
3. **Fixture Reuse**: Design fixtures for reusability across test modules and phases
4. **Parameterization**: Use pytest's parameterization features extensively to test multiple scenarios with minimal code duplication
5. **Documentation**: Document test purpose, fixtures, and assertions clearly
6. **Artifact Generation**: Add capability to save test artifacts (images, reports) for manual inspection when necessary

## Success Criteria

- **Coverage Target**: Achieve at least 70% statement coverage for initial testing
- **Test Independence**: Each test should pass when run in isolation or as part of the full suite
- **Fast Execution**: Core tests should execute quickly (under 5 minutes) for developer feedback
- **Clear Failures**: Test failures should provide clear, actionable information about what failed and why
