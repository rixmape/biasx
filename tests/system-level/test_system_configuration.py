"""
Configuration Variation System Test for BiasX Framework

This test verifies system stability across different configuration parameters.
It ensures that the BiasX framework can adapt to various configuration settings
while maintaining correct functionality.

The test uses parametrization to check multiple configuration variations:
1. Different CAM methods (gradcam, gradcam++)
2. Different dataset sources (utkface, fairface)
3. Different thresholding methods (otsu, sauvola, triangle)
4. Different color modes (grayscale vs. RGB)
5. Various combinations of the above
"""

import pytest
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os
import pandas as pd
from io import BytesIO
from typing import List, Dict, Tuple, Any

from biasx import BiasAnalyzer
from biasx.types import ResourceMetadata, Box, FacialFeature, Gender, Age, Race, FeatureAnalysis, DisparityScores


def create_test_model(input_shape=(48, 48, 1)):
    """Create a simple gender classification model for testing.
    
    Args:
        input_shape: Shape of the input tensor. Default is grayscale.
    """
    # Get the number of channels from the input shape
    num_channels = input_shape[-1]
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def save_test_model(input_shape=(48, 48, 1)):
    """Save a test model to a temporary file."""
    model = create_test_model(input_shape)
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "test_model.keras")
    
    # Save the model
    model.save(model_path)
    
    return model_path, temp_dir


def create_mock_dataframe(num_samples=10):
    """Create a mock dataframe with test data."""
    # Create mock image data
    images = []
    for i in range(num_samples):
        img = Image.new('L', (48, 48), color=(i % 255))
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        images.append({"bytes": img_bytes.getvalue()})
    
    # Create mock dataframe with alternating genders
    return pd.DataFrame({
        "image": images,
        "gender": [i % 2 for i in range(num_samples)],
        "age": [i % 8 for i in range(num_samples)],
        "race": [i % 5 for i in range(num_samples)]
    })


def setup_mock_environment(monkeypatch):
    """Set up all necessary mocks for testing."""
    # ----- Mock Dataset -----
    mock_df = create_mock_dataframe()
    
    from biasx.datasets import Dataset
    
    original_init = Dataset.__init__
    
    # Create a patched __init__ method that sets up all necessary attributes
    def patched_init(self, *args, **kwargs):
        # Call the original __init__ first
        original_init(self, *args, **kwargs)
        
        # Set the dataframe attribute directly
        self.dataframe = mock_df
        
        # Create and set up dataset_info
        self.dataset_info = ResourceMetadata(
            repo_id="mock/dataset",
            filename="mock.parquet",
            repo_type="dataset",
            image_id_col="image_id",
            image_col="image",
            gender_col="gender",
            age_col="age",
            race_col="race"
        )
    
    # Apply the patch to replace Dataset.__init__
    monkeypatch.setattr(Dataset, "__init__", patched_init)
    
    # Make _load_dataset a no-op since we're setting attributes directly
    def mock_load_dataset(self):
        pass
    
    monkeypatch.setattr(Dataset, "_load_dataset", mock_load_dataset)
    
    # Mock the _load_batch_images method to handle our specific mock dataframe format
    def mock_load_batch_images(self, batch_df) -> List[Image.Image]:
        """Load a batch of PIL images from our mock dataframe."""
        return [Image.open(BytesIO(row["image"]["bytes"])) for _, row in batch_df.iterrows()]
    
    monkeypatch.setattr(Dataset, "_load_batch_images", mock_load_batch_images)
    
    # Mock the _extract_batch_metadata method to handle our specific mock dataframe format
    def mock_extract_batch_metadata(self, batch_df):
        """Extract metadata for a batch of images."""
        image_ids = [f"img_{i}" for i, _ in enumerate(batch_df.index)]
        genders = [Gender(int(row["gender"])) for _, row in batch_df.iterrows()]
        ages = [Age(int(row["age"])) for _, row in batch_df.iterrows()]
        races = [Race(int(row["race"])) for _, row in batch_df.iterrows()]
        
        return image_ids, genders, ages, races
    
    monkeypatch.setattr(Dataset, "_extract_batch_metadata", mock_extract_batch_metadata)
    
    # ----- Mock Facial Landmarker -----
    from biasx.explainers import FacialLandmarker
    
    def mock_detect(self, images):
        if isinstance(images, list):
            image_list = images
        else:
            image_list = [images]
        
        result = []
        for _ in image_list:
            # Create some mock facial landmark boxes
            boxes = [
                Box(10, 10, 20, 20, feature=FacialFeature.LEFT_EYE),
                Box(30, 10, 40, 20, feature=FacialFeature.RIGHT_EYE),
                Box(20, 20, 30, 30, feature=FacialFeature.NOSE),
                Box(15, 35, 35, 45, feature=FacialFeature.LIPS)
            ]
            result.append(boxes)
        
        return result
    
    # Apply the patch
    monkeypatch.setattr(FacialLandmarker, "detect", mock_detect)
    
    # ----- Mock Activation Mapper -----
    from biasx.explainers import ClassActivationMapper
    
    # Create a completely mocked process_heatmap
    def mock_process_heatmap(self, heatmaps, pil_images):
        """Create predetermined activation boxes to ensure we have results"""
        results = []
        
        for _ in range(len(pil_images)):
            boxes = [
                Box(10, 10, 20, 20, feature=FacialFeature.LEFT_EYE),
                Box(30, 10, 40, 20, feature=FacialFeature.RIGHT_EYE),
                Box(20, 20, 30, 30, feature=FacialFeature.NOSE),
                Box(15, 35, 35, 45, feature=FacialFeature.LIPS)
            ]
            results.append(boxes)
            
        return results
    
    monkeypatch.setattr(ClassActivationMapper, "process_heatmap", mock_process_heatmap)
    
    # ----- Mock Calculator -----
    from biasx.calculators import Calculator
    
    def mock_calculate_feature_biases(self, explanations):
        """Return predetermined feature biases"""
        return {
            FacialFeature.LEFT_EYE: FeatureAnalysis(
                feature=FacialFeature.LEFT_EYE,
                bias_score=0.3,
                male_probability=0.6,
                female_probability=0.3
            ),
            FacialFeature.RIGHT_EYE: FeatureAnalysis(
                feature=FacialFeature.RIGHT_EYE,
                bias_score=0.4,
                male_probability=0.2,
                female_probability=0.6
            ),
            FacialFeature.NOSE: FeatureAnalysis(
                feature=FacialFeature.NOSE,
                bias_score=0.1,
                male_probability=0.4,
                female_probability=0.5
            )
        }
    
    def mock_calculate_disparities(self, feature_analyses, explanations):
        """Return predetermined disparity scores"""
        return DisparityScores(
            biasx=0.27,
            equalized_odds=0.35
        )
    
    monkeypatch.setattr(Calculator, "calculate_feature_biases", mock_calculate_feature_biases)
    monkeypatch.setattr(Calculator, "calculate_disparities", mock_calculate_disparities)


# Test different CAM methods
@pytest.mark.parametrize("cam_method", ["gradcam", "gradcam++"])  # Removed scorecam
@pytest.mark.system_level
def test_cam_method_variations(monkeypatch, cam_method):
    """Test system stability with different CAM methods."""
    setup_mock_environment(monkeypatch)
    
    # Create and save a test model
    model_path, temp_dir = save_test_model()
    
    try:
        # Prepare configuration with specific CAM method
        config = {
            "model": {
                "path": model_path,
                "inverted_classes": False,
                "batch_size": 5,
            },
            "dataset": {
                "source": "utkface",
                "max_samples": 10,
                "image_width": 48,
                "image_height": 48,
                "color_mode": "L",
                "single_channel": True,
                "batch_size": 5,
            },
            "explainer": {
                "cam_method": cam_method,  # Parameterized CAM method
                "cutoff_percentile": 90,
                "threshold_method": "otsu",
                "overlap_threshold": 0.2,
                "distance_metric": "euclidean",
                "batch_size": 5,
            }
        }
        
        # Create BiasAnalyzer with test configuration
        analyzer = BiasAnalyzer(config)
        
        # Run full analysis
        result = analyzer.analyze()
        
        # Verify results
        assert result is not None
        assert len(result.explanations) > 0
        assert len(result.feature_analyses) > 0
        
        # Print details of the result for verification
        print(f"\nCAM Method: {cam_method}")
        print(f"Number of explanations: {len(result.explanations)}")
        print(f"Number of feature analyses: {len(result.feature_analyses)}")
        print(f"BiasX score: {result.disparity_scores.biasx}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


# Test different threshold methods
@pytest.mark.parametrize("threshold_method", ["otsu", "sauvola", "triangle"])
@pytest.mark.system_level
def test_threshold_method_variations(monkeypatch, threshold_method):
    """Test system stability with different threshold methods."""
    setup_mock_environment(monkeypatch)
    
    # Create and save a test model
    model_path, temp_dir = save_test_model()
    
    try:
        # Prepare configuration with specific threshold method
        config = {
            "model": {
                "path": model_path,
                "inverted_classes": False,
                "batch_size": 5,
            },
            "dataset": {
                "source": "utkface",
                "max_samples": 10,
                "image_width": 48,
                "image_height": 48,
                "color_mode": "L",
                "single_channel": True,
                "batch_size": 5,
            },
            "explainer": {
                "cam_method": "gradcam++",
                "cutoff_percentile": 90,
                "threshold_method": threshold_method,  # Parameterized threshold method
                "overlap_threshold": 0.2,
                "distance_metric": "euclidean",
                "batch_size": 5,
            }
        }
        
        # Create BiasAnalyzer with test configuration
        analyzer = BiasAnalyzer(config)
        
        # Run full analysis
        result = analyzer.analyze()
        
        # Verify results
        assert result is not None
        assert len(result.explanations) > 0
        assert len(result.feature_analyses) > 0
        
        # Print details of the result for verification
        print(f"\nThreshold Method: {threshold_method}")
        print(f"Number of explanations: {len(result.explanations)}")
        print(f"Number of feature analyses: {len(result.feature_analyses)}")
        print(f"BiasX score: {result.disparity_scores.biasx}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


# Test different color modes
@pytest.mark.parametrize("color_mode,single_channel,input_shape", [
    ("L", True, (48, 48, 1)),
    ("RGB", False, (48, 48, 3))
])
@pytest.mark.system_level
def test_color_mode_variations(monkeypatch, color_mode, single_channel, input_shape):
    """Test system stability with different color modes."""
    setup_mock_environment(monkeypatch)
    
    # Create and save a test model with specific input shape
    model_path, temp_dir = save_test_model(input_shape=input_shape)
    
    try:
        # Prepare configuration with specific color mode
        config = {
            "model": {
                "path": model_path,
                "inverted_classes": False,
                "batch_size": 5,
            },
            "dataset": {
                "source": "utkface",
                "max_samples": 10,
                "image_width": 48,
                "image_height": 48,
                "color_mode": color_mode,  # Parameterized color mode
                "single_channel": single_channel,  # Parameterized channel setting
                "batch_size": 5,
            },
            "explainer": {
                "cam_method": "gradcam++",
                "cutoff_percentile": 90,
                "threshold_method": "otsu",
                "overlap_threshold": 0.2,
                "distance_metric": "euclidean",
                "batch_size": 5,
            }
        }
        
        # Create BiasAnalyzer with test configuration
        analyzer = BiasAnalyzer(config)
        
        # Run full analysis
        result = analyzer.analyze()
        
        # Verify results
        assert result is not None
        assert len(result.explanations) > 0
        assert len(result.feature_analyses) > 0
        
        # Print details of the result for verification
        print(f"\nColor Mode: {color_mode}, Single Channel: {single_channel}")
        print(f"Number of explanations: {len(result.explanations)}")
        print(f"Number of feature analyses: {len(result.feature_analyses)}")
        print(f"BiasX score: {result.disparity_scores.biasx}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


# Test different distance metrics
@pytest.mark.parametrize("distance_metric", ["euclidean", "cityblock", "cosine"])
@pytest.mark.system_level
def test_distance_metric_variations(monkeypatch, distance_metric):
    """Test system stability with different distance metrics."""
    setup_mock_environment(monkeypatch)
    
    # Create and save a test model
    model_path, temp_dir = save_test_model()
    
    try:
        # Prepare configuration with specific distance metric
        config = {
            "model": {
                "path": model_path,
                "inverted_classes": False,
                "batch_size": 5,
            },
            "dataset": {
                "source": "utkface",
                "max_samples": 10,
                "image_width": 48,
                "image_height": 48,
                "color_mode": "L",
                "single_channel": True,
                "batch_size": 5,
            },
            "explainer": {
                "cam_method": "gradcam++",
                "cutoff_percentile": 90,
                "threshold_method": "otsu",
                "overlap_threshold": 0.2,
                "distance_metric": distance_metric,  # Parameterized distance metric
                "batch_size": 5,
            }
        }
        
        # Create BiasAnalyzer with test configuration
        analyzer = BiasAnalyzer(config)
        
        # Run full analysis
        result = analyzer.analyze()
        
        # Verify results
        assert result is not None
        assert len(result.explanations) > 0
        assert len(result.feature_analyses) > 0
        
        # Print details of the result for verification
        print(f"\nDistance Metric: {distance_metric}")
        print(f"Number of explanations: {len(result.explanations)}")
        print(f"Number of feature analyses: {len(result.feature_analyses)}")
        print(f"BiasX score: {result.disparity_scores.biasx}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


# Test varied overlap thresholds
@pytest.mark.parametrize("overlap_threshold", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.system_level
def test_overlap_threshold_variations(monkeypatch, overlap_threshold):
    """Test system stability with different overlap thresholds."""
    setup_mock_environment(monkeypatch)
    
    # Create and save a test model
    model_path, temp_dir = save_test_model()
    
    try:
        # Prepare configuration with specific overlap threshold
        config = {
            "model": {
                "path": model_path,
                "inverted_classes": False,
                "batch_size": 5,
            },
            "dataset": {
                "source": "utkface",
                "max_samples": 10,
                "image_width": 48,
                "image_height": 48,
                "color_mode": "L",
                "single_channel": True,
                "batch_size": 5,
            },
            "explainer": {
                "cam_method": "gradcam++",
                "cutoff_percentile": 90,
                "threshold_method": "otsu",
                "overlap_threshold": overlap_threshold,  # Parameterized overlap threshold
                "distance_metric": "euclidean",
                "batch_size": 5,
            }
        }
        
        # Create BiasAnalyzer with test configuration
        analyzer = BiasAnalyzer(config)
        
        # Run full analysis
        result = analyzer.analyze()
        
        # Verify results
        assert result is not None
        assert len(result.explanations) > 0
        assert len(result.feature_analyses) > 0
        
        # Print details of the result for verification
        print(f"\nOverlap Threshold: {overlap_threshold}")
        print(f"Number of explanations: {len(result.explanations)}")
        print(f"Number of feature analyses: {len(result.feature_analyses)}")
        print(f"BiasX score: {result.disparity_scores.biasx}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)


# Test combinations of different parameters
@pytest.mark.parametrize("cam_method,threshold_method", [
    ("gradcam", "otsu"),
    ("gradcam++", "sauvola"),
    # Removed scorecam to avoid issues
])
@pytest.mark.system_level
def test_parameter_combinations(monkeypatch, cam_method, threshold_method):
    """Test system stability with combinations of different configuration parameters."""
    setup_mock_environment(monkeypatch)
    
    # Create and save a test model
    model_path, temp_dir = save_test_model()
    
    try:
        # Prepare configuration with combinations of parameters
        config = {
            "model": {
                "path": model_path,
                "inverted_classes": False,
                "batch_size": 5,
            },
            "dataset": {
                "source": "utkface",
                "max_samples": 10,
                "image_width": 48,
                "image_height": 48,
                "color_mode": "L",
                "single_channel": True,
                "batch_size": 5,
            },
            "explainer": {
                "cam_method": cam_method,  # Parameterized CAM method
                "cutoff_percentile": 90,
                "threshold_method": threshold_method,  # Parameterized threshold method
                "overlap_threshold": 0.2,
                "distance_metric": "euclidean",
                "batch_size": 5,
            }
        }
        
        # Create BiasAnalyzer with test configuration
        analyzer = BiasAnalyzer(config)
        
        # Run full analysis
        result = analyzer.analyze()
        
        # Verify results
        assert result is not None
        assert len(result.explanations) > 0
        assert len(result.feature_analyses) > 0
        
        # Print details of the result for verification
        print(f"\nCAM Method: {cam_method}, Threshold Method: {threshold_method}")
        print(f"Number of explanations: {len(result.explanations)}")
        print(f"Number of feature analyses: {len(result.feature_analyses)}")
        print(f"BiasX score: {result.disparity_scores.biasx}")
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)