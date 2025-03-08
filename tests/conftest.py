# Global fixtures
import os
import json
import tempfile
import pathlib
from typing import Dict, Any, Optional

import pytest
import numpy as np


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