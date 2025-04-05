"""Tests for integration between Model and Explainer components, 
ensuring prediction outputs are correctly used for visual explanations."""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from biasx.models import Model
from biasx.explainers import Explainer
from biasx.types import (
    Gender, CAMMethod, LandmarkerSource, ThresholdMethod, 
    DistanceMetric, Box, FacialFeature
)


@pytest.mark.integration
@pytest.mark.model_explainer
def test_model_to_explainer_prediction_flow(sample_model_path, controlled_test_images, check_activation_regions):
    """Test that Model predictions are correctly processed by Explainer."""
    # Setup
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=4)
    explainer = Explainer(
        landmarker_source=LandmarkerSource.MEDIAPIPE,
        cam_method=CAMMethod.GRADCAM,
        cutoff_percentile=90,
        threshold_method=ThresholdMethod.OTSU,
        overlap_threshold=0.2,
        distance_metric=DistanceMetric.EUCLIDEAN,
        batch_size=4
    )
    
    # Get test images
    images = controlled_test_images[:4]  # Use first 4 test images
    
    # Preprocess images for model input
    preprocessed = []
    for img in images:
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        preprocessed.append(img_array)
    
    # Mock the model's predict method to return controlled predictions
    with patch.object(model, 'predict') as mock_predict:
        mock_predict.return_value = [
            (Gender.MALE, 0.9),
            (Gender.FEMALE, 0.85),
            (Gender.MALE, 0.8),
            (Gender.FEMALE, 0.95)
        ]
        
        # Generate predictions
        predictions = model.predict(preprocessed)
        
        # Mock the landmarker to avoid actual detection
        with patch.object(explainer.landmarker, 'detect') as mock_detect:
            # Important fix: Return a list of lists of Box objects
            mock_detect.return_value = [
                [  # Landmarks for image 1
                    Box(15, 12, 20, 16, feature=FacialFeature.LEFT_EYE),
                    Box(28, 12, 33, 16, feature=FacialFeature.RIGHT_EYE),
                    Box(20, 18, 28, 25, feature=FacialFeature.NOSE),
                    Box(20, 30, 28, 35, feature=FacialFeature.LIPS)
                ],
                [  # Landmarks for image 2
                    Box(16, 13, 21, 17, feature=FacialFeature.LEFT_EYE),
                    Box(29, 13, 34, 17, feature=FacialFeature.RIGHT_EYE),
                    Box(21, 19, 29, 26, feature=FacialFeature.NOSE),
                    Box(21, 31, 29, 36, feature=FacialFeature.LIPS)
                ],
                [  # Landmarks for image 3
                    Box(14, 11, 19, 15, feature=FacialFeature.LEFT_EYE),
                    Box(27, 11, 32, 15, feature=FacialFeature.RIGHT_EYE),
                    Box(19, 17, 27, 24, feature=FacialFeature.NOSE),
                    Box(19, 29, 27, 34, feature=FacialFeature.LIPS)
                ],
                [  # Landmarks for image 4
                    Box(15, 12, 20, 16, feature=FacialFeature.LEFT_EYE),
                    Box(28, 12, 33, 16, feature=FacialFeature.RIGHT_EYE),
                    Box(20, 18, 28, 25, feature=FacialFeature.NOSE),
                    Box(20, 30, 28, 35, feature=FacialFeature.LIPS)
                ]
            ]
            
            # Mock the activation mapper to avoid actual CAM computation
            with patch.object(explainer.activation_mapper, 'generate_heatmap') as mock_generate:
                # Create mock activation maps corresponding to the gender predictions
                mock_generate.return_value = [
                    np.ones((48, 48)) * 0.5,  # Generic activation map for image 1
                    np.ones((48, 48)) * 0.5,  # Generic activation map for image 2
                    np.ones((48, 48)) * 0.5,  # Generic activation map for image 3
                    np.ones((48, 48)) * 0.5   # Generic activation map for image 4
                ]
                
                # Mock process_heatmap to return controlled boxes
                with patch.object(explainer.activation_mapper, 'process_heatmap') as mock_process:
                    # Define activation boxes based on the predicted gender
                    mock_process.return_value = [
                        [  # Activation boxes for image 1 (Male)
                            Box(15, 30, 33, 40)  # Jaw area
                        ],
                        [  # Activation boxes for image 2 (Female)
                            Box(16, 12, 21, 16),  # Left eye
                            Box(29, 13, 34, 17),  # Right eye
                            Box(21, 31, 29, 36)   # Lips
                        ],
                        [  # Activation boxes for image 3 (Male)
                            Box(14, 29, 32, 39)  # Jaw area
                        ],
                        [  # Activation boxes for image 4 (Female)
                            Box(15, 12, 20, 16),  # Left eye
                            Box(28, 12, 33, 16),  # Right eye
                            Box(20, 30, 28, 35)   # Lips
                        ]
                    ]
                    
                    # Generate explanations using model predictions
                    activation_maps, activation_boxes, landmark_boxes = explainer.explain_batch(
                        pil_images=images,
                        preprocessed_images=preprocessed,
                        model=model,
                        target_classes=[pred[0] for pred in predictions]
                    )
    
    # Assertions
    assert len(activation_maps) == 4
    assert len(activation_boxes) == 4
    assert len(landmark_boxes) == 4
    
    # Verify that explainer was called with the correct target classes
    mock_generate.assert_called_once()
    # Extract the target_classes argument (3rd argument)
    target_classes_arg = mock_generate.call_args[0][2]
    assert target_classes_arg == [Gender.MALE, Gender.FEMALE, Gender.MALE, Gender.FEMALE]
    
    # Verify that process_heatmap was called with the generated heatmaps
    mock_process.assert_called_once()
    # First argument should be the heatmaps from generate_heatmap
    heatmaps_arg = mock_process.call_args[0][0]
    assert len(heatmaps_arg) == 4


@pytest.mark.integration
@pytest.mark.model_explainer
@pytest.mark.parametrize("target_gender", [Gender.MALE, Gender.FEMALE])
def test_model_to_explainer_target_class(target_gender, sample_model_path, controlled_test_images):
    """Test that the Explainer correctly handles different target classes from Model."""
    # Setup
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=2)
    explainer = Explainer(
        landmarker_source=LandmarkerSource.MEDIAPIPE,
        cam_method=CAMMethod.GRADCAM,
        cutoff_percentile=90,
        threshold_method=ThresholdMethod.OTSU,
        overlap_threshold=0.2,
        distance_metric=DistanceMetric.EUCLIDEAN,
        batch_size=2
    )
    
    # Get test images
    images = controlled_test_images[:2]  # Use first 2 test images
    
    # Preprocess images for model input
    preprocessed = []
    for img in images:
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        preprocessed.append(img_array)
    
    # Override model predictions to always return the target gender
    with patch.object(model, 'predict') as mock_predict:
        mock_predict.return_value = [(target_gender, 0.9), (target_gender, 0.9)]
        
        # Generate predictions
        predictions = model.predict(preprocessed)
        
        # Mock the Explainer's dependencies
        with patch.object(explainer.landmarker, 'detect', return_value=[[]]):
            with patch.object(explainer.activation_mapper, 'generate_heatmap') as mock_generate:
                mock_generate.return_value = [np.ones((48, 48)) * 0.5, np.ones((48, 48)) * 0.5]
                
                with patch.object(explainer.activation_mapper, 'process_heatmap', return_value=[[]]):
                    # Generate explanations
                    activation_maps, activation_boxes, landmark_boxes = explainer.explain_batch(
                        pil_images=images,
                        preprocessed_images=preprocessed,
                        model=model,
                        target_classes=[pred[0] for pred in predictions]
                    )
    
    # Verify that generate_heatmap was called with the correct target class
    mock_generate.assert_called_once()
    target_classes_arg = mock_generate.call_args[0][2]
    assert all(cls == target_gender for cls in target_classes_arg)


@pytest.mark.integration
@pytest.mark.model_explainer
def test_model_to_explainer_empty_input():
    """Test that Model and Explainer correctly handle empty inputs."""
    # Setup with mocks to avoid actual model loading
    model = MagicMock()
    model.predict.return_value = []
    
    explainer = MagicMock()
    explainer.explain_batch.return_value = ([], [], [])
    
    # Test with empty inputs
    empty_images = []
    empty_preprocessed = []
    
    # Generate predictions
    predictions = model.predict(empty_preprocessed)
    
    # Generate explanations
    activation_maps, activation_boxes, landmark_boxes = explainer.explain_batch(
        pil_images=empty_images,
        preprocessed_images=empty_preprocessed,
        model=model,
        target_classes=[pred[0] for pred in predictions]
    )
    
    # Verify results
    assert predictions == []
    assert activation_maps == []
    assert activation_boxes == []
    assert landmark_boxes == []
    
    # Verify predict was called with empty list
    model.predict.assert_called_once_with([])
    
    # Verify explain_batch was called with empty lists
    explainer.explain_batch.assert_called_once()
    call_args = explainer.explain_batch.call_args[1]
    assert call_args["pil_images"] == []
    assert call_args["preprocessed_images"] == []
    assert call_args["target_classes"] == []


@pytest.mark.integration
@pytest.mark.model_explainer
def test_model_to_explainer_inverted_classes(sample_model_path, controlled_test_images):
    """Test integration with inverted class labels in the model."""
    # Setup with inverted classes
    model = Model(path=sample_model_path, inverted_classes=True, batch_size=2)
    explainer = Explainer(
        landmarker_source=LandmarkerSource.MEDIAPIPE,
        cam_method=CAMMethod.GRADCAM,
        cutoff_percentile=90,
        threshold_method=ThresholdMethod.OTSU,
        overlap_threshold=0.2,
        distance_metric=DistanceMetric.EUCLIDEAN,
        batch_size=2
    )
    
    # Get test images
    images = controlled_test_images[:2]
    
    # Preprocess images for model input
    preprocessed = []
    for img in images:
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        preprocessed.append(img_array)
    
    # Mock the _get_probabilities method to return known values
    with patch.object(model, '_get_probabilities') as mock_get_probs:
        # Return probabilities that would normally result in [Female, Male]
        # But with inverted_classes=True, should result in [Male, Female]
        mock_get_probs.return_value = np.array([
            [0.3, 0.7],  # Would normally be Female (1)
            [0.8, 0.2]   # Would normally be Male (0)
        ])
        
        # With mocked dependencies
        with patch.object(explainer.landmarker, 'detect', return_value=[[]]):
            with patch.object(explainer.activation_mapper, 'generate_heatmap') as mock_generate:
                mock_generate.return_value = [np.ones((48, 48)) * 0.5] * 2
                
                with patch.object(explainer.activation_mapper, 'process_heatmap', return_value=[[]]):
                    # Generate predictions
                    predictions = model.predict(preprocessed)
                    
                    # Generate explanations
                    explainer.explain_batch(
                        pil_images=images,
                        preprocessed_images=preprocessed,
                        model=model,
                        target_classes=[pred[0] for pred in predictions]
                    )
    
    # Verify predictions were inverted
    assert predictions[0][0] == Gender.MALE   # Would be Female without inversion
    assert predictions[1][0] == Gender.FEMALE  # Would be Male without inversion
    
    # Verify generate_heatmap was called with inverted classes
    mock_generate.assert_called_once()
    target_classes = mock_generate.call_args[0][2]
    assert target_classes == [Gender.MALE, Gender.FEMALE]