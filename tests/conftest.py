# Global fixtures
import os
import json
import tempfile
import pathlib
from typing import Dict, Any, Optional, Tuple

import pytest
import numpy as np


from PIL import Image
from unittest.mock import MagicMock, patch

from biasx.config import Config
from biasx.types import Age, Box, FacialFeature, Gender, ImageData, Race


import tensorflow as tf

from biasx.types import (
    Age, Box, CAMMethod, ColorMode, DatasetSource, 
    DistanceMetric, FacialFeature, Gender, ImageData, 
    LandmarkerSource, Race, ThresholdMethod
)

# ===== Path Handling Fixtures =====

@pytest.fixture(scope="session")
def test_dir() -> pathlib.Path:
    """Return the path to the tests directory."""
    return pathlib.Path(__file__).parent


@pytest.fixture(scope="session")
def test_data_dir(test_dir) -> pathlib.Path:
    """Return the path to the test data directory."""
    data_dir = test_dir / "data"
    assert data_dir.exists(), f"Test data directory does not exist: {data_dir}"
    return data_dir


@pytest.fixture(scope="session")
def sample_images_dir(test_data_dir) -> pathlib.Path:
    """Return the path to sample images for testing."""
    return test_data_dir / "sample_images"


@pytest.fixture(scope="session")
def sample_models_dir(test_data_dir) -> pathlib.Path:
    """Return the path to sample models for testing."""
    return test_data_dir / "sample_models"


@pytest.fixture(scope="session")
def test_configs_dir(test_data_dir) -> pathlib.Path:
    """Return the path to test configuration files."""
    return test_data_dir / "test_configs"


# ===== Configuration Fixtures =====

@pytest.fixture
def minimal_config() -> Dict[str, Any]:
    """Return a minimal valid configuration dictionary."""
    return {
        "model_path": "/path/to/model.h5",
        "dataset_path": "/path/to/dataset",
        "output_dir": "/path/to/output",
        "batch_size": 32,
        "image_size": [224, 224],
        "explainer_type": "gradcam"
    }


@pytest.fixture
def full_config() -> Dict[str, Any]:
    """Return a complete configuration with all possible options."""
    return {
        "model_path": "/path/to/model.h5",
        "dataset_path": "/path/to/dataset",
        "output_dir": "/path/to/output",
        "batch_size": 32,
        "image_size": [224, 224],
        "channels": 3,
        "explainer_type": "gradcam",
        "threshold": 0.75,
        "cache_dir": "/path/to/cache",
        "preprocess_fn": "default",
        "augmentation": True,
        "normalize": True,
        "gender_classes": ["male", "female"],
        "confidence_threshold": 0.6,
        "features_to_analyze": ["eyes", "nose", "mouth", "jaw", "forehead"],
        "save_visualizations": True,
        "visualization_format": "png",
        "verbose": True,
        "num_workers": 4,
        "seed": 42
    }


@pytest.fixture
def temp_config_file(full_config):
    """Create a temporary JSON configuration file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(full_config, f, indent=2)
        temp_filename = f.name
    
    yield temp_filename
    
    # Cleanup the temporary file after the test
    os.unlink(temp_filename)


# ===== Mock Classes for External Dependencies =====

class MockTensorFlowModel:
    """Mock TensorFlow model for testing."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def predict(self, x, **kwargs):
        """Return fake predictions for the input."""
        # Ensure input has the correct shape
        if isinstance(x, list):
            batch_size = len(x)
        else:
            batch_size = x.shape[0] if len(x.shape) > 3 else 1
        
        # Generate fake predictions with slight randomness
        # but predictable based on the input sum
        if isinstance(x, list) or isinstance(x, np.ndarray):
            # For simplicity, base the prediction on the sum of the input
            if isinstance(x, list):
                sums = [np.sum(item) for item in x]
            else:
                sums = [np.sum(x[i]) for i in range(batch_size)]
            
            # Create probabilities that sum to 1
            predictions = []
            for val in sums:
                # Use the sum to bias the result
                # Scale to be between 0.3 and 0.7 for the first class
                prob = (((val % 100) / 100) * 0.4) + 0.3
                predictions.append([prob, 1 - prob])
            
            return np.array(predictions)
        
        # Default random predictions if input is not as expected
        return np.random.random((batch_size, self.num_classes))


@pytest.fixture
def mock_tf_model():
    """Fixture to provide a mock TensorFlow model."""
    return MockTensorFlowModel()


class MockMediaPipeFaceDetector:
    """Mock MediaPipe face detector for testing."""
    
    def __init__(self, detection_confidence=0.5, max_num_faces=1):
        self.detection_confidence = detection_confidence
        self.max_num_faces = max_num_faces
        # Predefined landmarks for consistent testing
        self.predefined_landmarks = {
            # Simplified facial landmarks (normally would have many more points)
            "face_oval": [(100, 100), (150, 100), (150, 150), (100, 150)],
            "left_eye": [(110, 110), (120, 110), (120, 120), (110, 120)],
            "right_eye": [(130, 110), (140, 110), (140, 120), (130, 120)],
            "nose": [(125, 125), (125, 135)],
            "mouth": [(115, 140), (135, 140)],
            "left_eyebrow": [(110, 105), (120, 105)],
            "right_eyebrow": [(130, 105), (140, 105)]
        }
    
    def process(self, image):
        """Mock processing of an image to detect faces and landmarks."""
        # Always return the predefined landmarks for testing consistency
        return MockMediaPipeResults(self.predefined_landmarks)


class MockMediaPipeResults:
    """Mock results from MediaPipe face detection."""
    
    def __init__(self, landmarks_dict):
        self.landmarks_dict = landmarks_dict
        self.multi_face_landmarks = [MockFaceLandmarks(landmarks_dict)]


class MockFaceLandmarks:
    """Mock face landmarks data structure."""
    
    def __init__(self, landmarks_dict):
        self.landmarks_dict = landmarks_dict
        # Create a flat list of landmarks for compatibility with MediaPipe API
        self.landmark = []
        for feature, points in landmarks_dict.items():
            for x, y in points:
                # MediaPipe landmarks are normalized to [0,1]
                # Convert pixel coordinates to normalized values
                # Assuming 200x200 image for simplicity
                normalized_x = x / 200
                normalized_y = y / 200
                self.landmark.append(MockLandmark(normalized_x, normalized_y))


class MockLandmark:
    """Mock individual landmark point."""
    
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


@pytest.fixture
def mock_mediapipe_detector():
    """Fixture to provide a mock MediaPipe face detector."""
    return MockMediaPipeFaceDetector()


# ===== Test Data Generation =====

@pytest.fixture
def sample_image_batch(batch_size=4):
    """Generate a batch of simple test images."""
    # Create a batch of simple 10x10 single-channel images
    images = np.zeros((batch_size, 10, 10, 3), dtype=np.uint8)
    
    # Add some patterns to differentiate the images
    for i in range(batch_size):
        # Create a unique pattern for each image
        # Add a diagonal line with increasing intensity
        for j in range(10):
            images[i, j, j, 0] = 100 + i * 10  # Red channel
            if i % 2 == 0:
                images[i, j, 9-j, 1] = 150 + i * 10  # Green channel
            else:
                images[i, 9-j, j, 2] = 200 + i * 10  # Blue channel
    
    return images


@pytest.fixture
def sample_image_paths(tmpdir, sample_image_batch):
    """Create temporary image files and return their paths."""
    image_paths = []
    batch_size = sample_image_batch.shape[0]
    
    # Save each image to a temporary file
    for i in range(batch_size):
        img_path = tmpdir.join(f"test_image_{i}.png")
        # We'd normally use cv2 or PIL to save, but for the fixture
        # we'll just pretend we saved the file and record its path
        np.save(str(img_path), sample_image_batch[i])
        image_paths.append(str(img_path))
    
    return image_paths


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "image_0001.jpg": {"gender": "male", "age": 25, "ethnicity": "caucasian"},
        "image_0002.jpg": {"gender": "female", "age": 30, "ethnicity": "asian"},
        "image_0003.jpg": {"gender": "male", "age": 45, "ethnicity": "black"},
        "image_0004.jpg": {"gender": "female", "age": 35, "ethnicity": "hispanic"},
        "image_0005.jpg": {"gender": "non-binary", "age": 28, "ethnicity": "south_asian"}
    }


# ===== Type checking and validation helpers =====

def assert_valid_bbox(bbox):
    """Validate that a bounding box has the correct format."""
    assert isinstance(bbox, (list, tuple)), "Bounding box must be a list or tuple"
    assert len(bbox) == 4, "Bounding box must have 4 elements"
    assert all(isinstance(coord, (int, float)) for coord in bbox), "Bounding box coordinates must be numbers"
    x1, y1, x2, y2 = bbox
    assert x1 <= x2, "Bounding box x1 must be <= x2"
    assert y1 <= y2, "Bounding box y1 must be <= y2"


def assert_valid_heatmap(heatmap, original_image_shape=None):
    """Validate that a heatmap has the correct format."""
    assert isinstance(heatmap, np.ndarray), "Heatmap must be a numpy array"
    assert len(heatmap.shape) in (2, 3), "Heatmap must be 2D or 3D"
    
    if len(heatmap.shape) == 3:
        # If it's a 3D array, the last dimension should be 1 or 3 (grayscale or RGB)
        assert heatmap.shape[2] in (1, 3), "3D heatmap must have 1 or 3 channels"
    
    if original_image_shape is not None:
        # If original shape is provided, check that dimensions match or are proportional
        orig_h, orig_w = original_image_shape[:2]
        heat_h, heat_w = heatmap.shape[:2]
        
        # Heatmaps may be smaller than original image due to pooling layers
        # Check if dimensions are proportional
        assert heat_h / heat_w == pytest.approx(orig_h / orig_w, rel=0.1), \
            "Heatmap aspect ratio should match original image"


# ===== Utility Functions =====

def create_test_config(tmpdir, **kwargs):
    """Create a test configuration file with specified parameters."""
    # Start with minimal valid configuration
    config = {
        "model_path": str(tmpdir.join("model.h5")),
        "dataset_path": str(tmpdir.join("dataset")),
        "output_dir": str(tmpdir.join("output")),
        "batch_size": 16,
        "image_size": [224, 224],
        "explainer_type": "gradcam"
    }
    
    # Update with provided kwargs
    config.update(kwargs)
    
    # Write to a JSON file
    config_path = tmpdir.join("config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(config_path)

@pytest.fixture
def temp_file():
    """Fixture to provide a temporary file path that will be cleaned up after test."""
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp_path = temp.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_config():
    """Fixture to provide a basic mock configuration."""
    return Config({
        "model": {
            "path": "test_model.h5",
            "inverted_classes": False,
            "batch_size": 32,
        },
        "explainer": {
            "landmarker_source": "mediapipe",
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
            "overlap_threshold": 0.2,
            "distance_metric": "euclidean",
            "batch_size": 16,
        },
        "dataset": {
            "source": "utkface",
            "max_samples": 100,
            "shuffle": True,
            "seed": 42,
            "batch_size": 32,
        }
    })


@pytest.fixture
def sample_image():
    """Fixture to provide a sample PIL image."""
    img = Image.new('L', (48, 48), color=128)
    return img


@pytest.fixture
def sample_image_array():
    """Fixture to provide a sample image as numpy array."""
    return np.ones((48, 48, 1), dtype=np.float32) * 0.5


@pytest.fixture
def mock_image_data():
    """Fixture to provide a mock ImageData object."""
    img = Image.new('L', (48, 48), color=128)
    img_array = np.ones((48, 48, 1), dtype=np.float32) * 0.5
    
    return ImageData(
        image_id="test_image_001",
        pil_image=img,
        preprocessed_image=img_array,
        width=48,
        height=48,
        gender=Gender.MALE,
        age=Age.RANGE_20_29,
        race=Race.WHITE
    )


@pytest.fixture
def mock_boxes():
    """Fixture to provide a list of facial feature boxes."""
    return [
        Box(10, 15, 30, 25, feature=FacialFeature.LEFT_EYE),
        Box(35, 15, 55, 25, feature=FacialFeature.RIGHT_EYE),
        Box(30, 30, 40, 45, feature=FacialFeature.NOSE),
        Box(25, 50, 45, 60, feature=FacialFeature.LIPS)
    ]


@pytest.fixture
def mock_activation_map():
    """Fixture to provide a mock activation heatmap."""
    # Create a heatmap with a hotspot
    heatmap = np.zeros((48, 48), dtype=np.float32)
    heatmap[15:30, 20:35] = 0.8  # Hotspot area
    return heatmap


@pytest.fixture
def mock_tf_model():
    """Fixture to provide a mock TensorFlow model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.3, 0.7]])  # Predict female with 70% confidence
    return mock_model


@pytest.fixture
def mock_landmarker():
    """Fixture to provide a mock facial landmarker."""
    with patch('biasx.explainers.FacialLandmarker') as mock:
        landmarker = MagicMock()
        landmarker.detect.return_value = [[
            Box(10, 15, 30, 25, feature=FacialFeature.LEFT_EYE),
            Box(35, 15, 55, 25, feature=FacialFeature.RIGHT_EYE),
            Box(30, 30, 40, 45, feature=FacialFeature.NOSE),
            Box(25, 50, 45, 60, feature=FacialFeature.LIPS)
        ]]
        mock.return_value = landmarker
        yield mock


@pytest.fixture
def mock_cam():
    """Fixture to provide a mock class activation mapper."""
    with patch('biasx.explainers.ClassActivationMapper') as mock:
        mapper = MagicMock()
        
        # Create a heatmap with a hotspot
        heatmap = np.zeros((48, 48), dtype=np.float32)
        heatmap[15:30, 20:35] = 0.8  # Hotspot area
        
        mapper.generate_heatmap.return_value = [heatmap]
        mapper.process_heatmap.return_value = [[
            Box(15, 20, 30, 35)
        ]]
        
        mock.return_value = mapper
        yield mock

@pytest.fixture
def integration_config():
    """Return a configuration suitable for integration testing."""
    return {
        "model": {
            "path": "mock_model.h5",  # Will be replaced with actual path in tests
            "inverted_classes": False,
            "batch_size": 8,  # Small batch size for faster testing
        },
        "explainer": {
            "landmarker_source": "mediapipe",
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
            "overlap_threshold": 0.2,
            "distance_metric": "euclidean",
            "batch_size": 8,
        },
        "dataset": {
            "source": "utkface",
            "image_width": 48,  # Small size for testing
            "image_height": 48,
            "color_mode": "L",
            "single_channel": True,
            "max_samples": 20,  # Limited sample size
            "shuffle": True,
            "seed": 42,
            "batch_size": 8,
        }
    }

@pytest.fixture
def sample_model_path():
    """Create a temporary model file for testing."""
    # Create a minimal test model
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        model_path = temp_file.name
        model.save(model_path)
        
    yield model_path
    
    # Clean up after test
    if os.path.exists(model_path):
        os.unlink(model_path)

@pytest.fixture
def controlled_test_images(num_images=4):
    """Generate a controlled set of test images with known properties."""
    images = []
    for i in range(num_images):
        # Create a base image with a simple pattern
        img_array = np.zeros((48, 48), dtype=np.uint8)
        
        # Add some distinguishing features
        if i % 2 == 0:  # 'Male-like' features
            # Add eyes in the upper part
            img_array[10:15, 15:20] = 200  # Left eye
            img_array[10:15, 28:33] = 200  # Right eye
            # Add a larger jaw area
            img_array[30:40, 15:33] = 150
        else:  # 'Female-like' features
            # Add eyes in the upper part
            img_array[12:16, 16:21] = 200  # Left eye
            img_array[12:16, 27:32] = 200  # Right eye
            # Add lips
            img_array[30:35, 20:28] = 180
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_array)
        images.append(pil_image)
    
    return images

@pytest.fixture
def preprocessed_test_images(controlled_test_images):
    """Create preprocessed versions of the test images."""
    preprocessed = []
    for img in controlled_test_images:
        # Convert to numpy array, normalize, and add channel dimension
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        preprocessed.append(img_array)
    return preprocessed

@pytest.fixture
def test_image_data(controlled_test_images, preprocessed_test_images):
    """Create test ImageData objects with controlled characteristics."""
    image_data_list = []
    for i, (pil_img, preprocessed_img) in enumerate(zip(controlled_test_images, preprocessed_test_images)):
        # Alternate gender for testing different scenarios
        gender = Gender.MALE if i % 2 == 0 else Gender.FEMALE
        
        image_data = ImageData(
            image_id=f"test_image_{i}",
            pil_image=pil_img,
            preprocessed_image=preprocessed_img,
            width=48,
            height=48,
            gender=gender,
            age=Age.RANGE_20_29,  # Fixed age for simplicity
            race=Race.WHITE      # Fixed race for simplicity
        )
        image_data_list.append(image_data)
    
    return image_data_list

@pytest.fixture
def mock_activation_maps(num_maps=4):
    """Generate mock activation maps for testing."""
    maps = []
    for i in range(num_maps):
        # Create base heatmap of zeros
        heatmap = np.zeros((48, 48))
        
        # Add activation patterns based on index
        if i % 2 == 0:  # 'Male predicted' activation pattern
            # Activate jaw area
            heatmap[30:40, 15:33] = 0.8
        else:  # 'Female predicted' activation pattern
            # Activate eyes and lips
            heatmap[12:16, 16:21] = 0.7  # Left eye
            heatmap[12:16, 27:32] = 0.7  # Right eye
            heatmap[30:35, 20:28] = 0.9  # Lips
        
        maps.append(heatmap)
    
    return maps

@pytest.fixture
def mock_landmark_boxes(num_sets=4):
    """Generate mock landmark boxes for testing."""
    landmarks_sets = []
    for _ in range(num_sets):
        # Create a standard set of facial landmarks for each image
        landmarks = [
            Box(15, 12, 20, 16, feature=FacialFeature.LEFT_EYE),
            Box(28, 12, 33, 16, feature=FacialFeature.RIGHT_EYE),
            Box(20, 18, 28, 25, feature=FacialFeature.NOSE),
            Box(20, 30, 28, 35, feature=FacialFeature.LIPS),
            Box(10, 20, 15, 30, feature=FacialFeature.LEFT_CHEEK),
            Box(33, 20, 38, 30, feature=FacialFeature.RIGHT_CHEEK),
            Box(18, 35, 30, 42, feature=FacialFeature.CHIN),
            Box(15, 5, 33, 10, feature=FacialFeature.FOREHEAD),
            Box(15, 10, 20, 12, feature=FacialFeature.LEFT_EYEBROW),
            Box(28, 10, 33, 12, feature=FacialFeature.RIGHT_EYEBROW),
        ]
        landmarks_sets.append(landmarks)
    
    return landmarks_sets

@pytest.fixture
def create_test_explanation():
    """Fixture that provides a function to create test explanations."""
    def _create_explanation(
        image_id="test_image",
        true_gender=Gender.MALE,
        pred_gender=Gender.FEMALE,
        confidence=0.85,
        activation_boxes=None
    ):
        """Create a synthetic explanation with specified properties."""
        # Create basic image data
        pil_image = Image.new('L', (48, 48), color=128)
        img_array = np.ones((48, 48, 1), dtype=np.float32) * 0.5
        
        image_data = ImageData(
            image_id=image_id,
            pil_image=pil_image,
            preprocessed_image=img_array,
            width=48,
            height=48,
            gender=true_gender,
            age=Age.RANGE_20_29,
            race=Race.WHITE
        )
        
        # Default activation boxes if none provided
        if activation_boxes is None:
            activation_boxes = [
                Box(15, 12, 20, 16, feature=FacialFeature.LEFT_EYE),
                Box(28, 12, 33, 16, feature=FacialFeature.RIGHT_EYE)
            ]
        
        # Create activation map with simple pattern
        activation_map = np.zeros((48, 48))
        for box in activation_boxes:
            activation_map[box.min_y:box.max_y, box.min_x:box.max_x] = 0.8
        
        # Create standard landmark boxes
        landmark_boxes = [
            Box(15, 12, 20, 16, feature=FacialFeature.LEFT_EYE),
            Box(28, 12, 33, 16, feature=FacialFeature.RIGHT_EYE),
            Box(20, 18, 28, 25, feature=FacialFeature.NOSE),
            Box(20, 30, 28, 35, feature=FacialFeature.LIPS)
        ]
        
        from biasx.types import Explanation
        return Explanation(
            image_data=image_data,
            predicted_gender=pred_gender,
            prediction_confidence=confidence,
            activation_map=activation_map,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes
        )
    
    return _create_explanation

@pytest.fixture
def check_activation_regions():
    """Fixture providing a function to verify activation map regions."""
    def _check_regions(activation_map, predicted_gender):
        """Verify that activation map regions align with the predicted gender."""
        # Simple validation based on predicted gender
        if predicted_gender == Gender.MALE:
            # For male predictions, expect more activation in jaw/chin area
            lower_region = activation_map[30:, :]
            upper_region = activation_map[:20, :]
            # Lower face should have more activation for male predictions
            assert np.mean(lower_region) > np.mean(upper_region)
        else:
            # For female predictions, expect more activation in eyes/lips
            eye_region = activation_map[10:20, :]
            lip_region = activation_map[30:35, 20:28]
            rest_region = np.delete(activation_map, np.s_[10:20, :], axis=0)
            rest_region = np.delete(rest_region, np.s_[30:35, 20:28], axis=0)
            # Eyes and lips should have more activation for female predictions
            assert (np.mean(eye_region) + np.mean(lip_region))/2 > np.mean(rest_region)
        
        return True
    
    return _check_regions