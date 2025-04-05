"""Tests for the model handling functionality in BiasX."""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

from biasx.models import Model
from biasx.types import Gender
from unittest.mock import patch, MagicMock


def create_test_model(input_shape=(48, 48, 1), num_classes=2):
    """
    Create a minimal gender classification model for testing.
    
    Args:
        input_shape (tuple): Shape of input images
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: A simple test model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


@pytest.fixture
def sample_model_path():
    """
    Create a temporary model file for testing.
    
    Returns:
        str: Path to the temporary model file
    """
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        # Create and save a test model
        model = create_test_model()
        model.save(temp_file.name)
        
        # Yield the filename
        yield temp_file.name
    
    # Clean up the file after the test
    try:
        os.unlink(temp_file.name)
    except Exception:
        pass


def test_model_initialization(sample_model_path):
    """
    Test Model class initialization.
    
    Verifies:
    - Model can be loaded correctly
    - Initialization parameters are set
    """
    # Test with default parameters
    biasx_model = Model(
        path=sample_model_path, 
        inverted_classes=False, 
        batch_size=32
    )
    
    # Verify model attributes
    assert hasattr(biasx_model, 'model')
    assert biasx_model.inverted_classes == False
    assert biasx_model.batch_size == 32


def test_model_input_preparation(sample_model_path):
    """
    Test input preparation methods.

    Verifies:
    - Correct handling of different input shapes
    - Proper dimension expansion
    """
    # Create a test model
    biasx_model = Model(
        path=sample_model_path,
        inverted_classes=False,
        batch_size=32
    )

    # Test cases for different input scenarios
    test_cases = [
        # Single 2D image (grayscale)
        np.random.rand(48, 48),

        # Single 3D image (color)
        np.random.rand(48, 48, 1),

        # Multiple images
        np.random.rand(5, 48, 48, 1)
    ]

    for input_array in test_cases:
        # Wrap the input in a list to match the method's expectation
        prepared_input = biasx_model._prepare_input(
            [input_array] if input_array.ndim < 4 else input_array
        )

        # Verify input shape
        assert prepared_input.ndim == 4  # Always 4D
        assert prepared_input.shape[1:] == (48, 48, 1)  # Match model input shape


def test_model_probability_processing(sample_model_path):
    """
    Test probability processing and class conversion.
    
    Verifies:
    - Correct probability calculation
    - Proper gender prediction
    - Confidence score generation
    """
    # Create a test model
    biasx_model = Model(
        path=sample_model_path, 
        inverted_classes=False, 
        batch_size=32
    )
    
    # Create sample inputs
    inputs = [
        np.random.rand(48, 48, 1),  # First input
        np.random.rand(48, 48, 1)   # Second input
    ]
    
    # Predict
    predictions = biasx_model.predict(inputs)
    
    # Verify predictions
    assert len(predictions) == len(inputs)
    
    for pred in predictions:
        # Verify prediction type
        assert isinstance(pred[0], Gender)
        assert isinstance(pred[1], float)
        
        # Verify confidence range
        assert 0 <= pred[1] <= 1


def test_model_inverted_classes(sample_model_path):
    """
    Test inverted classes functionality.
    
    Verifies:
    - Class inversion works correctly
    """
    # Test with inverted classes
    biasx_model_inverted = Model(
        path=sample_model_path, 
        inverted_classes=True, 
        batch_size=32
    )
    
    # Test with non-inverted classes
    biasx_model_normal = Model(
        path=sample_model_path, 
        inverted_classes=False, 
        batch_size=32
    )
    
    # Create sample input
    inputs = [np.random.rand(48, 48, 1)]
    
    # Predict with inverted and non-inverted
    pred_inverted = biasx_model_inverted.predict(inputs)[0]
    pred_normal = biasx_model_normal.predict(inputs)[0]
    
    # Verify opposite genders
    assert pred_inverted[0] != pred_normal[0]


def test_batch_prediction(sample_model_path):
    """
    Test batch prediction functionality.
    
    Verifies:
    - Correct handling of different batch sizes
    - Consistent output for batches
    """
    biasx_model = Model(
        path=sample_model_path, 
        inverted_classes=False, 
        batch_size=16
    )
    
    # Test with different batch sizes
    batch_sizes = [1, 5, 10, 32]
    
    for batch_size in batch_sizes:
        # Generate random inputs
        inputs = [np.random.rand(48, 48, 1) for _ in range(batch_size)]
        
        # Predict
        predictions = biasx_model.predict(inputs)
        
        # Verify batch prediction
        assert len(predictions) == batch_size
        
        for pred in predictions:
            # Check prediction structure
            assert isinstance(pred[0], Gender)
            assert isinstance(pred[1], float)
            assert 0 <= pred[1] <= 1


def test_empty_input_handling(sample_model_path):
    """
    Test handling of empty input.
    
    Verifies:
    - Graceful handling of empty input list
    """
    biasx_model = Model(
        path=sample_model_path, 
        inverted_classes=False, 
        batch_size=32
    )
    
    # Test empty input
    predictions = biasx_model.predict([])
    
    # Verify empty output
    assert len(predictions) == 0


def test_get_probabilities_non_softmax_output(sample_model_path):
    """
    Test handling of model output that doesn't need softmax application.
    
    Verifies:
    - Correct handling of output that already has probabilities
    """
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=32)
    
    # Create a small batch
    test_batch = np.random.rand(2, 48, 48, 1)
    
    # Mock the model's predict method to return a simple output
    # that will trigger the condition where softmax is not applied
    # We need to create a single-column output as this will hit the condition
    # where len(output.shape) > 1 and output.shape[1] > 1 is False
    mock_output = np.array([[0.7], [0.4]])
    
    with patch.object(model.model, 'predict', return_value=mock_output):
        # Call the method
        probs = model._get_probabilities(test_batch)
        
        # Verify the output is returned as-is without softmax
        assert np.array_equal(probs, mock_output)


def test_prepare_input_direct_ndarray(sample_model_path):
    """
    Test preparing input when given a direct ndarray instead of a list.
    
    Verifies:
    - Correct handling of direct ndarray input
    """
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=32)
    
    # Test with a direct ndarray input (not in a list)
    # This tests line 47 where batch = preprocessed_images is assigned
    test_img = np.random.rand(48, 48, 1)  # Single 3D image with channel
    
    # Call the method directly with the ndarray
    result = model._prepare_input(test_img)
    
    # Should expand to batch dimension
    assert result.shape == (1, 48, 48, 1)
    
    # Try with a 2D image (no channel)
    test_img_2d = np.random.rand(48, 48)  # Single 2D image without channel
    
    # This should hit line 38 for handling a direct ndarray
    result_2d = model._prepare_input(test_img_2d)
    
    # Should add both batch and channel dimensions
    assert result_2d.shape == (1, 48, 48, 1)


def test_get_probabilities_with_empty_batch(sample_model_path):
    """
    Test that _get_probabilities correctly handles empty batches.
    
    Verifies:
    - Proper handling of zero-length inputs
    """
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=32)
    
    # Create an empty batch
    empty_batch = np.empty((0, 48, 48, 1))
    
    # Call _get_probabilities with the empty batch
    result = model._get_probabilities(empty_batch)
    
    # Verify result is an empty array with the correct shape
    assert result.shape == (0, 2)