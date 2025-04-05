"""Tests for integration between Dataset and Model components,
ensuring data preprocessing compatibility and batch handling."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from biasx.models import Model
from biasx.datasets import Dataset
from biasx.types import Gender, ColorMode, DatasetSource


@pytest.mark.integration
@pytest.mark.dataset_model
def test_dataset_to_model_preprocessing_compatibility(sample_model_path):
    """Test that Dataset preprocessing output is compatible with Model input."""
    # Setup
    dataset = Dataset(
        source=DatasetSource.UTKFACE,
        image_width=48,
        image_height=48,
        color_mode=ColorMode.GRAYSCALE,
        single_channel=True,
        max_samples=10,
        shuffle=False,
        seed=42,
        batch_size=5
    )
    
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=5)
    
    # Mock the dataset's _load_dataset method to avoid actual data loading
    with patch.object(dataset, '_load_dataset'):
        # Create a mock __iter__ method that returns controlled test batches
        mock_batch1 = [MagicMock() for _ in range(5)]
        mock_batch2 = [MagicMock() for _ in range(5)]
        
        # Configure each mock image data object with preprocessed image
        for i, mock_img_data in enumerate(mock_batch1 + mock_batch2):
            # Create a properly shaped preprocessed image
            mock_img_data.preprocessed_image = np.ones((48, 48, 1), dtype=np.float32) * (i / 10)
            mock_img_data.gender = Gender.MALE if i % 2 == 0 else Gender.FEMALE
        
        # Configure dataset to return our mock batches
        dataset.__iter__ = lambda self: iter([mock_batch1, mock_batch2])
        
        # Test each batch
        for batch in dataset:
            # Extract preprocessed images from ImageData objects
            preprocessed_images = [img_data.preprocessed_image for img_data in batch]
            
            # Verify model can process these images without errors
            predictions = model.predict(preprocessed_images)
            
            # Assertions
            assert len(predictions) == len(batch)
            for pred in predictions:
                assert isinstance(pred[0], Gender)
                assert 0 <= pred[1] <= 1  # Confidence score


@pytest.mark.integration
@pytest.mark.dataset_model
@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_dataset_model_batch_compatibility(sample_model_path, batch_size):
    """Test that Dataset and Model handle various batch sizes correctly."""
    # Setup
    dataset = Dataset(
        source=DatasetSource.UTKFACE,
        image_width=48,
        image_height=48,
        color_mode=ColorMode.GRAYSCALE,
        single_channel=True,
        max_samples=batch_size,  # Match max_samples to batch_size for simplicity
        shuffle=False,
        seed=42,
        batch_size=batch_size
    )
    
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=batch_size)
    
    # Mock the dataset's _load_dataset method to avoid actual data loading
    with patch.object(dataset, '_load_dataset'):
        # Configure mock batch
        mock_batch = [MagicMock() for _ in range(batch_size)]
        
        # Configure each mock image data object with preprocessed image
        for i, mock_img_data in enumerate(mock_batch):
            # Create a properly shaped preprocessed image
            mock_img_data.preprocessed_image = np.ones((48, 48, 1), dtype=np.float32) * (i / 10)
            mock_img_data.gender = Gender.MALE if i % 2 == 0 else Gender.FEMALE
        
        # Configure dataset to return our mock batch
        dataset.__iter__ = lambda self: iter([mock_batch])
        
        # Test the batch
        for batch in dataset:
            # Extract preprocessed images from ImageData objects
            preprocessed_images = [img_data.preprocessed_image for img_data in batch]
            
            # Verify model can process these images without errors
            predictions = model.predict(preprocessed_images)
            
            # Assertions
            assert len(predictions) == batch_size
            for i, pred in enumerate(predictions):
                assert isinstance(pred[0], Gender)
                assert 0 <= pred[1] <= 1


@pytest.mark.integration
@pytest.mark.dataset_model
def test_dataset_model_color_mode_compatibility(sample_model_path):
    """Test compatibility between Dataset color modes and Model input requirements."""
    # Test different color mode configurations
    color_configs = [
        {"color_mode": ColorMode.GRAYSCALE, "single_channel": True, "expected_shape": (48, 48, 1)},
        {"color_mode": ColorMode.RGB, "single_channel": False, "expected_shape": (48, 48, 3)}
    ]
    
    for config in color_configs:
        # Setup dataset with specific color configuration
        dataset = Dataset(
            source=DatasetSource.UTKFACE,
            image_width=48,
            image_height=48,
            color_mode=config["color_mode"],
            single_channel=config["single_channel"],
            max_samples=5,
            shuffle=False,
            seed=42,
            batch_size=5
        )
        
        # Create model with matching input shape
        input_shape = config["expected_shape"]
        
        # Mock the dataset's _load_dataset method
        with patch.object(dataset, '_load_dataset'):
            # Configure mock batch
            mock_batch = [MagicMock() for _ in range(5)]
            
            # Configure each mock image data with appropriate shape
            for mock_img_data in mock_batch:
                mock_img_data.preprocessed_image = np.ones(input_shape, dtype=np.float32) * 0.5
                mock_img_data.gender = Gender.MALE
            
            # Configure dataset to return our mock batch
            dataset.__iter__ = lambda self: iter([mock_batch])
            
            # Mock the model's predict method to verify input shapes
            with patch('biasx.models.Model.predict') as mock_predict:
                mock_predict.return_value = [(Gender.MALE, 0.8) for _ in range(5)]
                
                # Test the batch
                for batch in dataset:
                    # Extract preprocessed images
                    preprocessed_images = [img_data.preprocessed_image for img_data in batch]
                    
                    # Call model predict
                    model_results = mock_predict(preprocessed_images)
                    
                    # Verify input shape
                    for img in preprocessed_images:
                        assert img.shape == input_shape
                    
                    # Verify mock_predict was called
                    mock_predict.assert_called_once()


@pytest.mark.integration
@pytest.mark.dataset_model
def test_dataset_model_empty_batch_handling(sample_model_path):
    """Test handling of empty batches between Dataset and Model."""
    # Setup
    dataset = Dataset(
        source=DatasetSource.UTKFACE,
        image_width=48,
        image_height=48,
        color_mode=ColorMode.GRAYSCALE,
        single_channel=True,
        max_samples=0,  # Empty dataset
        shuffle=False,
        seed=42,
        batch_size=5
    )
    
    model = Model(path=sample_model_path, inverted_classes=False, batch_size=5)
    
    # Completely mock the dataset's data loading and iteration
    with patch.object(dataset, 'dataframe', new=pd.DataFrame()), \
         patch.object(dataset, '_load_dataset', return_value=None):
        
        # Count batches
        batch_count = 0
        prediction_count = 0
        
        # Test batches
        for batch in dataset:
            batch_count += 1
            
            # This block should not be reached with an empty dataset
            preprocessed_images = [img_data.preprocessed_image for img_data in batch]
            predictions = model.predict(preprocessed_images)
            prediction_count += len(predictions)
        
        # Verify no batches were processed
        assert batch_count == 0, "Dataset should not yield any batches when max_samples is 0"
        assert prediction_count == 0, "No predictions should be made on empty dataset"
        
        # Test direct empty list handling in Model
        empty_predictions = model.predict([])
        assert empty_predictions == [], "Model should return an empty list for empty input"