# tests/test_infrastructure.py
import io
import os
from typing import List, Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from biasx.types import (
    Age, Box, FacialFeature, Gender, ImageData, 
    LandmarkerSource, Race
)


class MockTensorFlowModel:
    """
    Mock TensorFlow model for controlled testing scenarios.
    
    Allows precise control over model predictions and behavior.
    """
    def __init__(
        self, 
        input_shape: Tuple[int, int, int] = (48, 48, 1), 
        num_classes: int = 2,
        prediction_strategy: str = 'balanced'
    ):
        """
        Initialize a mock TensorFlow model with configurable behavior.
        
        Args:
            input_shape (Tuple[int, int, int]): Shape of input images
            num_classes (int): Number of output classes
            prediction_strategy (str): Strategy for generating predictions
                - 'balanced': Roughly equal predictions
                - 'biased': Skewed predictions
                - 'random': Completely random predictions
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.prediction_strategy = prediction_strategy
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Generate predictions with controlled behavior.
        
        Args:
            images (np.ndarray): Input images for prediction
        
        Returns:
            np.ndarray: Predicted probabilities
        """
        batch_size = images.shape[0]
        
        if self.prediction_strategy == 'balanced':
            # Roughly balanced predictions
            predictions = np.array([
                [0.45, 0.55] if i % 2 == 0 else [0.55, 0.45] 
                for i in range(batch_size)
            ])
        elif self.prediction_strategy == 'biased':
            # Biased towards one class
            predictions = np.array([
                [0.2, 0.8] if i % 2 == 0 else [0.8, 0.2] 
                for i in range(batch_size)
            ])
        else:  # 'random'
            predictions = np.random.rand(batch_size, self.num_classes)
            predictions /= predictions.sum(axis=1)[:, np.newaxis]
        
        return predictions
    
    def summary(self):
        """
        Provide a mock model summary.
        """
        print(f"Mock TensorFlow Model")
        print(f"Input Shape: {self.input_shape}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Prediction Strategy: {self.prediction_strategy}")


class MockDataset:
    """
    Mock Dataset for controlled testing scenarios.
    
    Generates predefined batches with controllable characteristics.
    """
    def __init__(
        self, 
        batch_size: int = 32, 
        num_batches: int = 5,
        gender_distribution: Optional[List[float]] = None,
        age_distribution: Optional[List[float]] = None
    ):
        """
        Initialize a mock dataset with configurable parameters.
        
        Args:
            batch_size (int): Number of images per batch
            num_batches (int): Total number of batches
            gender_distribution (List[float], optional): Probability of male/female
            age_distribution (List[float], optional): Probability distribution for age groups
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        
        # Default distributions if not provided
        self.gender_distribution = gender_distribution or [0.5, 0.5]
        self.age_distribution = age_distribution or [0.125] * 8
        
        self._current_batch = 0
    
    def __iter__(self):
        """Make the dataset iterable."""
        return self
    
    def __next__(self) -> List[ImageData]:
        """
        Generate a batch of mock image data.
        
        Returns:
            List[ImageData]: A batch of simulated image data
        """
        if self._current_batch >= self.num_batches:
            raise StopIteration
        
        batch_data = []
        for _ in range(self.batch_size):
            # Randomly select gender based on distribution
            gender = np.random.choice([Gender.MALE, Gender.FEMALE], p=self.gender_distribution)
            
            # Randomly select age group based on distribution
            age = np.random.choice(list(Age), p=self.age_distribution)
            
            # Create mock PIL image (48x48 grayscale)
            mock_image = Image.fromarray(
                np.random.randint(0, 256, (48, 48), dtype=np.uint8)
            )
            
            # Preprocess the image
            preprocessed_image = np.array(
                mock_image.convert('L').resize((48, 48)), 
                dtype=np.float32
            ) / 255.0
            preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)
            
            image_data = ImageData(
                image_id=f"mock_image_{self._current_batch}_{_}",
                pil_image=mock_image,
                preprocessed_image=preprocessed_image,
                width=48,
                height=48,
                gender=gender,
                age=age
            )
            
            batch_data.append(image_data)
        
        self._current_batch += 1
        return batch_data
    
    def __len__(self):
        """Return total number of batches."""
        return self.num_batches


class MockLandmarker:
    """
    Mock Facial Landmarker for controlled testing scenarios.
    
    Generates predefined facial landmarks with controllable characteristics.
    """
    def __init__(
        self, 
        source: LandmarkerSource = LandmarkerSource.MEDIAPIPE,
        landmark_strategy: str = 'consistent'
    ):
        """
        Initialize a mock landmarker.
        
        Args:
            source (LandmarkerSource): Source of landmarker
            landmark_strategy (str): Strategy for generating landmarks
                - 'consistent': Generates similar landmarks for similar images
                - 'random': Completely random landmark generation
        """
        self.source = source
        self.landmark_strategy = landmark_strategy
    
    def detect(self, images: List[Image.Image]) -> List[List[Box]]:
        """
        Detect facial landmarks for a list of images.
        
        Args:
            images (List[Image.Image]): Input images
        
        Returns:
            List[List[Box]]: Detected facial landmark boxes
        """
        results = []
        for idx, image in enumerate(images):
            width, height = image.size
            
            if self.landmark_strategy == 'consistent':
                # Generate consistent landmarks based on image index
                features = {
                    FacialFeature.LEFT_EYE: (width * 0.3, height * 0.4, width * 0.4, height * 0.5),
                    FacialFeature.RIGHT_EYE: (width * 0.6, height * 0.4, width * 0.7, height * 0.5),
                    FacialFeature.NOSE: (width * 0.45, height * 0.5, width * 0.55, height * 0.6),
                    FacialFeature.LIPS: (width * 0.3, height * 0.7, width * 0.7, height * 0.8),
                    FacialFeature.LEFT_CHEEK: (width * 0.2, height * 0.5, width * 0.3, height * 0.6),
                    FacialFeature.RIGHT_CHEEK: (width * 0.7, height * 0.5, width * 0.8, height * 0.6),
                }
            else:  # 'random'
                features = {
                    feature: (
                        np.random.randint(0, width // 2),
                        np.random.randint(0, height // 2),
                        np.random.randint(width // 2, width),
                        np.random.randint(height // 2, height)
                    )
                    for feature in FacialFeature
                }
            
            landmark_boxes = [
                Box(
                    min_x=int(x1), 
                    min_y=int(y1), 
                    max_x=int(x2), 
                    max_y=int(y2), 
                    feature=feature
                )
                for feature, (x1, y1, x2, y2) in features.items()
            ]
            
            results.append(landmark_boxes)
        
        return results


@pytest.fixture(scope='session')
def mock_tensorflow_model():
    """
    Pytest fixture for creating a mock TensorFlow model.
    
    Yields:
        MockTensorFlowModel: A mock TensorFlow model
    """
    return MockTensorFlowModel()


@pytest.fixture(scope='session')
def mock_dataset():
    """
    Pytest fixture for creating a mock dataset.
    
    Yields:
        MockDataset: A mock dataset with configurable parameters
    """
    return MockDataset()


@pytest.fixture(scope='session')
def mock_landmarker():
    """
    Pytest fixture for creating a mock facial landmarker.
    
    Yields:
        MockLandmarker: A mock facial landmarker
    """
    return MockLandmarker()


def generate_test_coverage_config():
    """
    Generate a configuration for test coverage reporting.
    
    Returns:
        dict: Configuration for pytest-cov
    """
    return {
        'pytest_args': [
            '--cov=biasx',
            '--cov-report=html',
            '--cov-report=term',
            '--cov-config=.coveragerc'
        ],
        'coverage_config': {
            'run': {
                'source': ['biasx'],
                'omit': [
                    '*/__init__.py',
                    '*/tests/*',
                    '*/_version.py'
                ]
            },
            'report': {
                'exclude_lines': [
                    'pragma: no cover',
                    'def __repr__',
                    'raise NotImplementedError',
                    'if __name__ == .__main__.:',
                    'pass'
                ]
            }
        }
    }