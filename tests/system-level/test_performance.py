"""Performance and Scalability Tests for BiasX."""

import pytest
import tempfile
import os
import tensorflow as tf
import numpy as np
import tracemalloc

from biasx import BiasAnalyzer


@pytest.fixture(scope="module")
def sample_model():
    """Create a temporary model for performance testing with proper conv layers."""
    # Create a model with convolutional layers
    model = tf.keras.Sequential([
        # Add convolutional layers to ensure Grad-CAM works
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='last_conv_layer'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
        model.save(temp_model_file.name)
        temp_model_path = temp_model_file.name
    
    yield temp_model_path
    
    # Cleanup
    os.unlink(temp_model_path)


@pytest.mark.performance
@pytest.mark.parametrize("dataset_size", [100, 500, 1000, 5000])
def test_performance_scalability(sample_model, dataset_size, benchmark):
    """
    Test the performance and scalability of BiasX analyzer with varying dataset sizes.
    
    Args:
        sample_model (str): Path to the temporary model file
        dataset_size (int): Number of samples to test
        benchmark: pytest-benchmark fixture for performance measurement
    """
    # Prepare configuration for the test
    config = {
        "model": {"path": sample_model},
        "dataset": {
            "source": "utkface",
            "max_samples": dataset_size,
            "image_width": 48,
            "image_height": 48,
            "color_mode": "L",
            "single_channel": True,
        },
        "explainer": {
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
        }
    }
    
    def run_analysis():
        """Wrapper function to run BiasAnalyzer."""
        analyzer = BiasAnalyzer(config)
        return analyzer.analyze()
    
    # Use pytest-benchmark to measure performance
    result = benchmark(run_analysis)
    
    # Performance assertions
    assert result is not None, "Analysis result should not be None"
    
    # Performance metrics
    print(f"\nDataset Size: {dataset_size}")
    print(f"Number of Explanations: {len(result.explanations)}")
    print(f"Number of Feature Analyses: {len(result.feature_analyses)}")
    print(f"BiasX Score: {result.disparity_scores.biasx}")
    print(f"Equalized Odds: {result.disparity_scores.equalized_odds}")
    
    # Validate basic result structure
    assert len(result.explanations) > 0, "Should have at least some explanations"
    assert len(result.feature_analyses) > 0, "Feature analyses should be generated"
    assert hasattr(result.disparity_scores, 'biasx'), "Disparity scores should have biasx attribute"
    assert hasattr(result.disparity_scores, 'equalized_odds'), "Disparity scores should have equalized_odds attribute"


def test_memory_usage(sample_model):
    """
    Conduct a basic memory usage test for larger dataset sizes.
    
    This test is more of a sanity check to ensure that memory usage 
    doesn't grow exponentially with dataset size.
    """
    config = {
        "model": {"path": sample_model},
        "dataset": {
            "source": "utkface",
            "max_samples": 5000,
            "image_width": 48,
            "image_height": 48,
            "color_mode": "L",
            "single_channel": True,
        },
        "explainer": {
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
        }
    }
    
    # Start memory tracking
    tracemalloc.start()
    
    # Run analysis
    analyzer = BiasAnalyzer(config)
    result = analyzer.analyze()
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Basic memory usage assertions
    # These values are approximate and may need adjustment based on actual implementation
    assert current > 0, "Memory usage should be greater than 0"
    assert peak < 1000 * 1024 * 1024, "Peak memory usage should be less than 1 GB"  # 1 GB limit
    
    # Verify result is not None and has basic structure
    assert result is not None, "Analysis result should not be None"
    assert len(result.explanations) > 0, "Should have at least some explanations"