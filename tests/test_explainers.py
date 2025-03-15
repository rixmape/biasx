# tests/test_explainers.py
import io
import json
import tempfile
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
import mediapipe as mp
import unittest.mock

from biasx.types import (
    CAMMethod, 
    Gender, 
    LandmarkerSource, 
    ThresholdMethod, 
    DistanceMetric,
    Box,
    FacialFeature
)
from biasx.explainers import ClassActivationMapper, FacialLandmarker, Explainer
from biasx.models import Model


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
    
    # Add more layers to support CAM visualization
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


@pytest.fixture
def sample_images():
    """
    Generate sample test images.
    
    Returns:
        List[Image.Image]: List of test images
    """
    images = []
    for _ in range(3):
        # Create a random grayscale image
        img = Image.fromarray(
            np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        )
        images.append(img)
    return images


@pytest.fixture
def mock_tensorflow_model():
    """
    Create a mock TensorFlow model for testing CAM.
    
    Returns:
        tf.keras.Model: A simple test model
    """
    return create_test_model()


def test_class_activation_mapper(mock_tensorflow_model):
    """
    Test ClassActivationMapper functionality.
    
    Verifies:
    - Heatmap generation
    - Different CAM methods
    - Thresholding
    """
    # Only test GRADCAM and GRADCAM++ due to ScoreCAM complexity
    cam_methods = [CAMMethod.GRADCAM, CAMMethod.GRADCAM_PLUS_PLUS]
    threshold_methods = [ThresholdMethod.OTSU, ThresholdMethod.SAUVOLA]
    
    for cam_method in cam_methods:
        for threshold_method in threshold_methods:
            # Create ClassActivationMapper
            cam_mapper = ClassActivationMapper(
                cam_method=cam_method, 
                cutoff_percentile=90, 
                threshold_method=threshold_method
            )
            
            # Generate random input
            input_images = np.random.rand(3, 48, 48, 1)
            target_classes = [0, 1, 0]  # Use integer classes
            
            # Generate heatmaps
            heatmaps = cam_mapper.generate_heatmap(
                mock_tensorflow_model,
                input_images,
                target_classes
            )
            
            # Verify heatmap generation
            assert len(heatmaps) == len(input_images)
            for heatmap in heatmaps:
                assert isinstance(heatmap, np.ndarray)
                assert heatmap.ndim == 2


def test_class_activation_mapper_processing(sample_images):
    """
    Test heatmap processing and bounding box creation.
    
    Verifies:
    - Heatmap thresholding
    - Bounding box generation
    """
    # Create ClassActivationMapper
    cam_mapper = ClassActivationMapper(
        cam_method=CAMMethod.GRADCAM, 
        cutoff_percentile=90, 
        threshold_method=ThresholdMethod.OTSU
    )
    
    # Generate mock heatmaps
    heatmaps = [
        np.random.rand(48, 48) for _ in sample_images
    ]
    
    # Process heatmaps
    activation_boxes = cam_mapper.process_heatmap(heatmaps, sample_images)
    
    # Verify bounding box generation
    assert len(activation_boxes) == len(sample_images)
    for boxes in activation_boxes:
        assert isinstance(boxes, list)
        for box in boxes:
            assert isinstance(box, Box)
            assert box.min_x < box.max_x
            assert box.min_y < box.max_y


@pytest.fixture
def mock_landmark_mapping():
    """
    Create a mock landmark mapping for testing.
    
    Returns:
        dict: Mock landmark mapping
    """
    return {
        FacialFeature.LEFT_EYE: [0, 1, 2],
        FacialFeature.RIGHT_EYE: [3, 4, 5],
        FacialFeature.NOSE: [6, 7, 8],
    }


def test_facial_landmarker(mock_landmark_mapping):
    """
    Test FacialLandmarker functionality.

    Verifies:
    - Landmark detection
    - Feature mapping
    """
    # Import necessary modules to avoid circular imports
    import unittest
    from biasx.explainers import FacialLandmarker, LandmarkerSource, FacialFeature
    from biasx.types import Box
    
    # Find the largest index in the mapping to determine how many landmarks we need
    max_index = 0
    for indices in mock_landmark_mapping.values():
        if indices and max(indices) > max_index:
            max_index = max(indices)
    
    # Need to create at least max_index + 1 landmarks
    num_landmarks = max_index + 1
    
    # Create a mock detector with a custom detect method
    class MockDetector:
        def detect(self, mp_image):
            class MockResult:
                # Create enough landmarks to cover all indices in the mapping
                face_landmarks = [[
                    type('MockLandmark', (), {'x': 0.5, 'y': 0.5})()
                    for _ in range(num_landmarks)
                ]]
            return MockResult()
    
    # Mock create_from_options to return our mock detector
    def mock_create_from_options(options):
        return MockDetector()
    
    # Use patch to replace the problematic methods
    with unittest.mock.patch('mediapipe.tasks.python.vision.face_landmarker.FaceLandmarker.create_from_options', 
                            mock_create_from_options):
        # Mock the resource loading to avoid file access
        def mock_load_resources(self):
            self.landmarker_info = type('ResourceMetadata', (), {
                'repo_id': 'test/repo',
                'filename': 'test_model.task',
                'repo_type': 'model'
            })()
            self.model_path = '/mock/path/model.task'
            self.landmark_mapping = mock_landmark_mapping
        
        # Apply the mock to _load_resources
        original_load_resources = FacialLandmarker._load_resources
        FacialLandmarker._load_resources = mock_load_resources
        
        try:
            # Create the FacialLandmarker instance
            landmarker = FacialLandmarker(source=LandmarkerSource.MEDIAPIPE)
            
            # Create a test image
            test_image = Image.new('RGB', (100, 100))
            
            # Test the detect method
            result = landmarker.detect(test_image)
            
            # Assertions
            assert isinstance(result, list)
            assert len(result) == 1  # One result for one image
            
            boxes = result[0]
            assert isinstance(boxes, list)
            assert len(boxes) == len(mock_landmark_mapping)
            
            # Check that boxes were created for each feature in the mapping
            features_in_boxes = {box.feature for box in boxes}
            for feature in mock_landmark_mapping:
                assert feature in features_in_boxes
                
        finally:
            # Restore the original method
            FacialLandmarker._load_resources = original_load_resources

def test_explainer_integration(mock_tensorflow_model, sample_images):
    """
    Test Explainer class integration.
    
    Verifies:
    - Batch explanation generation
    - Interaction of sub-components
    """
    # Create a temporary file for the model
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        model_path = temp_file.name
        mock_tensorflow_model.save(model_path)

    try:
        # Create Explainer with test configurations
        explainer = Explainer(
            landmarker_source=LandmarkerSource.MEDIAPIPE,
            cam_method=CAMMethod.GRADCAM,
            cutoff_percentile=90,
            threshold_method=ThresholdMethod.OTSU,
            overlap_threshold=0.2,
            distance_metric=DistanceMetric.EUCLIDEAN,
            batch_size=16
        )

        # Preprocess images
        preprocessed_images = [
            np.array(img.convert('L').resize((48, 48)), dtype=np.float32) / 255.0
            for img in sample_images
        ]
        preprocessed_images = [np.expand_dims(img, axis=-1) for img in preprocessed_images]

        # Create a simple mock model
        model = Model(
            path=model_path,
            inverted_classes=False,
            batch_size=16
        )

        # Explain batch
        activation_maps, activation_boxes, landmark_boxes = explainer.explain_batch(
            pil_images=sample_images,
            preprocessed_images=preprocessed_images,
            model=model,
            target_classes=[Gender.MALE] * len(sample_images)
        )

        # Verify outputs
        assert len(activation_maps) == len(sample_images)
        assert len(activation_boxes) == len(sample_images)
        assert len(landmark_boxes) == len(sample_images)

        # Check individual outputs
        for maps, act_boxes, land_boxes in zip(activation_maps, activation_boxes, landmark_boxes):
            assert isinstance(maps, np.ndarray)
            assert isinstance(act_boxes, list)
            assert isinstance(land_boxes, list)
    finally:
        # Clean up the temporary model file
        import os
        if os.path.exists(model_path):
            os.unlink(model_path)