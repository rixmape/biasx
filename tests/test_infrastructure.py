# tests/test_infrastructure.py
import io
import os
from typing import List, Optional, Tuple, Union, Dict, Any

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

"""Mock classes and utilities for BiasX integration testing."""

class MockTensorFlowModel:
    """Mock TensorFlow model for controlled testing scenarios."""
    
    def __init__(self, prediction_strategy='balanced', input_shape=(48, 48, 1)):
        """
        Initialize with different prediction strategies for testing.
        
        Args:
            prediction_strategy: Controls prediction pattern:
                - 'balanced': Alternates male/female predictions
                - 'all_male': Always predicts male
                - 'all_female': Always predicts female
                - 'random': Random predictions
            input_shape: Expected input shape
        """
        self.prediction_strategy = prediction_strategy
        self.input_shape = input_shape
        self.layers = []  # Mock layers for compatibility
        
    def predict(self, images, verbose=0, batch_size=None):
        """
        Generate predictions with controlled behavior.
        
        Args:
            images: Input images (list or numpy array)
            verbose: Ignored, for compatibility
            batch_size: Ignored, for compatibility
        
        Returns:
            List of (Gender, confidence) tuples based on strategy
        """
        # Determine batch size from input
        if isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = images.shape[0] if images.ndim > 3 else 1
        
        # Generate predictions based on strategy
        if self.prediction_strategy == 'balanced':
            # Return alternating gender predictions
            return [(Gender.MALE, 0.8) if i % 2 == 0 else (Gender.FEMALE, 0.8) 
                    for i in range(batch_size)]
        elif self.prediction_strategy == 'all_male':
            # Always predict male
            return [(Gender.MALE, 0.9) for _ in range(batch_size)]
        elif self.prediction_strategy == 'all_female':
            # Always predict female
            return [(Gender.FEMALE, 0.9) for _ in range(batch_size)]
        elif self.prediction_strategy == 'based_on_input':
            # Use input image sum to determine prediction
            # Higher values predict male, lower values predict female
            if isinstance(images, list):
                sums = [np.sum(img) for img in images]
            else:
                sums = [np.sum(images[i]) for i in range(batch_size)]
            
            return [(Gender.MALE, 0.8) if s > 0.5 else (Gender.FEMALE, 0.8) 
                    for s in sums]
        else:  # 'random'
            # Random predictions
            return [(Gender.MALE if np.random.random() > 0.5 else Gender.FEMALE, 
                    0.7 + 0.3 * np.random.random()) for _ in range(batch_size)]
    
    # Mocked methods for compatibility with activation mapping
    def get_layer(self, name):
        """Mock get_layer method for testing."""
        return MockLayer(name)
    
    def get_config(self):
        """Mock get_config method for testing."""
        return {"name": "mock_model", "input_shape": self.input_shape}
    
    def save(self, filepath, **kwargs):
        """Mock save method that creates an empty file."""
        with open(filepath, 'w') as f:
            f.write("Mock model file")


class MockLayer:
    """Mock TensorFlow layer for testing."""
    
    def __init__(self, name):
        """Initialize with layer name."""
        self.name = name
        self.activation = None
        self.output = None
        
    def get_config(self):
        """Mock get_config method."""
        return {"name": self.name}


class MockDataset:
    """Mock Dataset for controlled testing scenarios."""
    
    def __init__(
        self, 
        batch_size: int = 4, 
        num_batches: int = 3,
        gender_distribution: Optional[List[float]] = None,
        include_errors: bool = False
    ):
        """
        Initialize a mock dataset with configurable parameters.
        
        Args:
            batch_size: Number of images per batch
            num_batches: Total number of batches to generate
            gender_distribution: Probability [male, female] (default: [0.5, 0.5])
            include_errors: Whether to include error cases (e.g., missing images)
        """
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.gender_distribution = gender_distribution or [0.5, 0.5]
        self.include_errors = include_errors
        self._current_batch = 0
    
    def __iter__(self):
        """Make the dataset iterable."""
        self._current_batch = 0
        return self
    
    def __next__(self) -> List[ImageData]:
        """
        Generate a batch of mock image data.
        
        Returns:
            List[ImageData]: A batch of simulated image data
        
        Raises:
            StopIteration: When all batches have been generated
        """
        if self._current_batch >= self.num_batches:
            raise StopIteration
        
        batch_data = []
        for i in range(self.batch_size):
            # Generate gender based on distribution
            gender = Gender.MALE if np.random.random() < self.gender_distribution[0] else Gender.FEMALE
            
            # Generate age (weighted towards middle age groups)
            age_weights = [0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]
            age = np.random.choice(list(Age), p=age_weights)
            
            # Create mock image
            img_size = (48, 48)
            mock_image = Image.new('L', img_size, color=128)
            
            # Add gender-specific patterns
            if gender == Gender.MALE:
                # Darker region in lower face for male
                img_array = np.ones(img_size) * 128
                img_array[30:40, 15:33] = 180  # Jaw area
                mock_image = Image.fromarray(img_array.astype(np.uint8))
            else:
                # Darker regions for eyes and lips for female
                img_array = np.ones(img_size) * 128
                img_array[12:16, 16:21] = 180  # Left eye
                img_array[12:16, 27:32] = 180  # Right eye
                img_array[30:35, 20:28] = 190  # Lips
                mock_image = Image.fromarray(img_array.astype(np.uint8))
            
            # Preprocess the image
            img_array = np.array(mock_image, dtype=np.float32) / 255.0
            preprocessed_image = np.expand_dims(img_array, axis=-1)
            
            # Create ImageData object
            image_data = ImageData(
                image_id=f"mock_image_batch{self._current_batch}_{i}",
                pil_image=mock_image,
                preprocessed_image=preprocessed_image,
                width=48,
                height=48,
                gender=gender,
                age=age,
                race=Race.WHITE  # Fixed for simplicity
            )
            
            # Include some error cases if specified
            if self.include_errors and np.random.random() < 0.1:
                # 10% chance of introducing an error
                error_type = np.random.choice(['missing_image', 'bad_preprocess'])
                
                if error_type == 'missing_image':
                    image_data.pil_image = None
                elif error_type == 'bad_preprocess':
                    # Invalid shape or values
                    image_data.preprocessed_image = np.zeros((10, 10, 1))
            
            batch_data.append(image_data)
        
        self._current_batch += 1
        return batch_data
    
    def __len__(self):
        """Return total number of batches."""
        return self.num_batches


class MockExplainer:
    """Mock Explainer for controlled testing scenarios."""
    
    def __init__(
        self,
        feature_focus: Optional[List[FacialFeature]] = None,
        error_rate: float = 0.0
    ):
        """
        Initialize a mock explainer.
        
        Args:
            feature_focus: Features to emphasize in activation maps
            error_rate: Probability of simulating an error
        """
        self.feature_focus = feature_focus or [
            FacialFeature.LEFT_EYE, 
            FacialFeature.RIGHT_EYE,
            FacialFeature.NOSE,
            FacialFeature.LIPS
        ]
        self.error_rate = error_rate
    
    def explain_batch(
        self, 
        pil_images: List[Image.Image], 
        preprocessed_images: List[np.ndarray], 
        model: Any, 
        target_classes: List[Gender]
    ) -> Tuple[List[np.ndarray], List[List[Box]], List[List[Box]]]:
        """
        Generate mock explanations for a batch of images.
        
        Args:
            pil_images: List of PIL images
            preprocessed_images: List of preprocessed image arrays
            model: Model object (unused in mock)
            target_classes: List of target gender classes
            
        Returns:
            Tuple containing:
            - List of activation maps
            - List of activation boxes
            - List of landmark boxes
        """
        # Simulate an error if needed
        if np.random.random() < self.error_rate:
            raise RuntimeError("Mock explainer error")
        
        batch_size = len(pil_images)
        activation_maps = []
        activation_boxes = []
        landmark_boxes = []
        
        for i, (image, target) in enumerate(zip(pil_images, target_classes)):
            # Get image dimensions
            width, height = (48, 48) if image is None else image.size
            
            # Generate activation map with gender-specific patterns
            activation_map = np.zeros((height, width))
            
            if target == Gender.MALE:
                # Activate jaw/chin area for male predictions
                activation_map[30:40, 15:33] = 0.8
            else:
                # Activate eyes and lips for female predictions
                activation_map[12:16, 16:21] = 0.7  # Left eye
                activation_map[12:16, 27:32] = 0.7  # Right eye
                activation_map[30:35, 20:28] = 0.9  # Lips
            
            activation_maps.append(activation_map)
            
            # Generate activation boxes based on the heatmap
            boxes = []
            if target == Gender.MALE:
                boxes.append(Box(15, 30, 33, 40, feature=FacialFeature.CHIN))
            else:
                boxes.append(Box(16, 12, 21, 16, feature=FacialFeature.LEFT_EYE))
                boxes.append(Box(27, 12, 32, 16, feature=FacialFeature.RIGHT_EYE))
                boxes.append(Box(20, 30, 28, 35, feature=FacialFeature.LIPS))
            
            activation_boxes.append(boxes)
            
            # Generate standard landmark boxes for all facial features
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
            landmark_boxes.append(landmarks)
        
        return activation_maps, activation_boxes, landmark_boxes


class MockCalculator:
    """Mock Calculator for controlled testing scenarios."""
    
    def __init__(self, bias_level: float = 0.3):
        """
        Initialize a mock calculator.
        
        Args:
            bias_level: Base bias level to simulate (0.0-1.0)
        """
        self.bias_level = bias_level
    
    def calculate_feature_biases(self, explanations: List[Any]) -> Dict[FacialFeature, Any]:
        """
        Calculate mock feature biases.
        
        Args:
            explanations: List of explanation objects
            
        Returns:
            Dictionary mapping features to mock analysis objects
        """
        from biasx.types import FeatureAnalysis
        
        # Count misclassifications by gender
        male_misclassified = sum(1 for e in explanations 
                                if e.image_data.gender == Gender.MALE and e.predicted_gender != Gender.MALE)
        female_misclassified = sum(1 for e in explanations 
                                  if e.image_data.gender == Gender.FEMALE and e.predicted_gender != Gender.FEMALE)
        
        # Calculate probabilities based on activated features in misclassifications
        feature_counts = {feature: {"male": 0, "female": 0} for feature in FacialFeature}
        
        for exp in explanations:
            if exp.image_data.gender != exp.predicted_gender:  # Misclassified
                activated_features = set(box.feature for box in exp.activation_boxes if box.feature)
                for feature in activated_features:
                    if feature:
                        if exp.image_data.gender == Gender.MALE:
                            feature_counts[feature]["male"] += 1
                        else:
                            feature_counts[feature]["female"] += 1
        
        # Calculate feature analyses
        feature_analyses = {}
        for feature in FacialFeature:
            male_prob = (feature_counts[feature]["male"] / male_misclassified 
                         if male_misclassified > 0 else 0.0)
            female_prob = (feature_counts[feature]["female"] / female_misclassified 
                          if female_misclassified > 0 else 0.0)
            
            # Add some controlled randomness to probabilities
            male_prob = min(1.0, max(0.0, male_prob + np.random.uniform(-0.1, 0.1)))
            female_prob = min(1.0, max(0.0, female_prob + np.random.uniform(-0.1, 0.1)))
            
            # Calculate bias score as absolute difference with some randomness
            bias_score = abs(male_prob - female_prob) + np.random.uniform(-0.05, 0.05)
            bias_score = min(1.0, max(0.0, bias_score))
            
            # Only include features with some activation
            if male_prob > 0 or female_prob > 0:
                feature_analyses[feature] = FeatureAnalysis(
                    feature=feature,
                    bias_score=round(bias_score, 3),
                    male_probability=round(male_prob, 3),
                    female_probability=round(female_prob, 3)
                )
        
        return feature_analyses
    
    def calculate_disparities(self, feature_analyses, explanations):
        """
        Calculate mock disparity scores.
        
        Args:
            feature_analyses: Dictionary of feature analyses
            explanations: List of explanation objects
            
        Returns:
            DisparityScores object with mock values
        """
        from biasx.types import DisparityScores
        
        if not feature_analyses:
            return DisparityScores()
            
        # Calculate BiasX score (average of feature bias scores)
        bias_scores = [analysis.bias_score for analysis in feature_analyses.values()]
        biasx_score = sum(bias_scores) / len(bias_scores)
        
        # Add some randomness to the score
        biasx_score = min(1.0, max(0.0, biasx_score + np.random.uniform(-0.05, 0.05)))
        
        # Calculate mock equalized odds score
        # Base it on how balanced the misclassifications are between genders
        male_misclassified = sum(1 for e in explanations 
                                if e.image_data.gender == Gender.MALE and e.predicted_gender != Gender.MALE)
        female_misclassified = sum(1 for e in explanations 
                                  if e.image_data.gender == Gender.FEMALE and e.predicted_gender != Gender.FEMALE)
        
        male_total = sum(1 for e in explanations if e.image_data.gender == Gender.MALE)
        female_total = sum(1 for e in explanations if e.image_data.gender == Gender.FEMALE)
        
        male_error_rate = male_misclassified / male_total if male_total > 0 else 0
        female_error_rate = female_misclassified / female_total if female_total > 0 else 0
        
        equalized_odds = abs(male_error_rate - female_error_rate)
        
        # Add some randomness
        equalized_odds = min(1.0, max(0.0, equalized_odds + np.random.uniform(-0.05, 0.05)))
        
        return DisparityScores(
            biasx=round(biasx_score, 3),
            equalized_odds=round(equalized_odds, 3)
        )


# Utility function to create test explanations
def create_test_explanation(
    image_id="test_image",
    true_gender=Gender.MALE,
    pred_gender=Gender.FEMALE,
    confidence=0.85,
    activation_boxes=None
):
    """
    Create a synthetic explanation with specified properties.
    
    Args:
        image_id: Image identifier
        true_gender: Actual gender
        pred_gender: Predicted gender
        confidence: Prediction confidence
        activation_boxes: Optional list of activation boxes
    
    Returns:
        Explanation object with the specified properties
    """
    from biasx.types import Explanation
    
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
    
    return Explanation(
        image_data=image_data,
        predicted_gender=pred_gender,
        prediction_confidence=confidence,
        activation_map=activation_map,
        activation_boxes=activation_boxes,
        landmark_boxes=landmark_boxes
    )


# Utility function to create test image data
def create_test_image_data(image_id, gender=None):
    """
    Create test ImageData with specified properties.
    
    Args:
        image_id: Image identifier
        gender: Optional gender (randomly chosen if None)
    
    Returns:
        ImageData object for testing
    """
    if gender is None:
        gender = Gender.MALE if np.random.random() > 0.5 else Gender.FEMALE
    
    # Create PIL image with gender-specific features
    img_array = np.ones((48, 48), dtype=np.uint8) * 128
    
    if gender == Gender.MALE:
        img_array[30:40, 15:33] = 180  # Jaw area
    else:
        img_array[12:16, 16:21] = 180  # Left eye
        img_array[12:16, 27:32] = 180  # Right eye
        img_array[30:35, 20:28] = 190  # Lips
    
    pil_image = Image.fromarray(img_array)
    
    # Preprocess the image
    preprocessed = np.array(pil_image, dtype=np.float32) / 255.0
    preprocessed = np.expand_dims(preprocessed, axis=-1)
    
    return ImageData(
        image_id=image_id,
        pil_image=pil_image,
        preprocessed_image=preprocessed,
        width=48,
        height=48,
        gender=gender,
        age=Age.RANGE_20_29,
        race=Race.WHITE
    )