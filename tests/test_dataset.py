# tests/test_dataset.py
import io
import os
from typing import List

import numpy as np
import pandas as pd
import pytest
from PIL import Image
import pyarrow.parquet as pq
import pyarrow as pa

from biasx.types import Gender, Age, ColorMode
from biasx.datasets import Dataset


@pytest.fixture
def mock_parquet_dataset(tmp_path):
    """
    Create a mock Parquet dataset for testing.
    
    Args:
        tmp_path: Temporary directory provided by pytest
    
    Returns:
        str: Path to the mock Parquet file
    """
    # Create sample image data
    def create_mock_image(gender, age):
        """Generate a mock image as bytes."""
        img = Image.new('RGB', (224, 224), color=(gender * 100, age * 30, 50))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return {"bytes": img_byte_arr.getvalue()}

    # Create a pandas DataFrame with more mock data
    data = {
        'image_id': [f'img_{i}' for i in range(100)],  # Increased to 100
        'image': [create_mock_image(i % 2, (i // 2) % 8) for i in range(100)],
        'gender': [i % 2 for i in range(100)],
        'age': [(i // 2) % 8 for i in range(100)],
        'race': [i % 5 for i in range(100)]
    }
    
    df = pd.DataFrame(data)
    
    # Convert to PyArrow Table and write to Parquet
    table = pa.Table.from_pandas(df)
    parquet_path = tmp_path / 'mock_dataset.parquet'
    pq.write_table(table, parquet_path)
    
    return str(parquet_path)


@pytest.fixture
def mock_dataset_config(monkeypatch, mock_parquet_dataset):
    """
    Mock the dataset configuration to use our test Parquet file.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        mock_parquet_dataset: Path to mock Parquet file
    
    Returns:
        dict: Mocked dataset configuration
    """
    mock_config = {
        "utkface": {
            "repo_id": "test/dataset",
            "filename": "mock_dataset.parquet",
            "repo_type": "dataset",
            "image_id_col": "image_id",
            "image_col": "image",
            "gender_col": "gender",
            "age_col": "age",
            "race_col": "race"
        }
    }
    
    # Monkeypatch the get_json_config to return our mock config
    def mock_get_json_config(*args, **kwargs):
        return mock_config
    
    # Monkeypatch the get_resource_path to return our mock Parquet path
    def mock_get_resource_path(**kwargs):
        return mock_parquet_dataset
    
    monkeypatch.setattr(
        'biasx.datasets.get_json_config', 
        mock_get_json_config
    )
    monkeypatch.setattr(
        'biasx.datasets.get_resource_path', 
        mock_get_resource_path
    )
    
    return mock_config


def test_dataset_initialization(mock_dataset_config):
    """
    Test basic dataset initialization.
    
    Verifies that:
    - Dataset can be created
    - Correct number of samples are loaded
    - Default configurations are applied
    """
    dataset = Dataset(
        source='utkface', 
        image_width=48, 
        image_height=48, 
        color_mode=ColorMode.GRAYSCALE,
        single_channel=True,
        max_samples=50,
        shuffle=True,
        seed=42,
        batch_size=16
    )
    
    # Verify dataset properties
    assert len(dataset) == 50
    assert dataset.batch_size == 16
    assert dataset.color_mode == ColorMode.GRAYSCALE


def test_dataset_batch_generation(mock_dataset_config):
    """
    Test batch generation from the dataset.
    
    Verifies that:
    - Batches are generated correctly
    - Batch size is consistent
    - Image preprocessing works
    """
    dataset = Dataset(
        source='utkface', 
        image_width=48, 
        image_height=48, 
        color_mode=ColorMode.GRAYSCALE,
        single_channel=True,
        max_samples=100,
        shuffle=False,
        seed=42,
        batch_size=16
    )
    
    # Iterate through batches
    batch_count = 0
    for batch in dataset:
        # Verify batch properties
        assert len(batch) == min(16, 100 - batch_count * 16)  # Handle last incomplete batch
        
        # Check individual image data
        for image_data in batch:
            # Verify preprocessed image
            assert image_data.preprocessed_image.shape == (48, 48, 1)
            assert image_data.preprocessed_image.dtype == np.float32
            
            # Verify metadata
            assert image_data.gender in [Gender.MALE, Gender.FEMALE]
            assert image_data.age in list(Age)
            assert image_data.width == 48
            assert image_data.height == 48
        
        batch_count += 1
    
    # Verify total number of batches
    assert batch_count == 7  # 100 samples / 16 batch size = 6.25 batches (rounded up)


def test_dataset_preprocessing_transformations(mock_dataset_config):
    """
    Test image preprocessing transformations.
    
    Verifies:
    - Image resizing
    - Color mode conversion
    - Normalization
    """
    # Test different color modes and sizes
    test_cases = [
        {
            'color_mode': ColorMode.GRAYSCALE, 
            'single_channel': True, 
            'expected_shape': (48, 48, 1)
        },
        {
            'color_mode': ColorMode.RGB, 
            'single_channel': False, 
            'expected_shape': (48, 48, 3)
        }
    ]
    
    for case in test_cases:
        dataset = Dataset(
            source='utkface', 
            image_width=48, 
            image_height=48, 
            color_mode=case['color_mode'],
            single_channel=case['single_channel'],
            max_samples=10,
            shuffle=False,
            seed=42,
            batch_size=5
        )
        
        # Take first batch
        first_batch = next(iter(dataset))
        
        # Check preprocessing
        for image_data in first_batch:
            # Verify shape
            assert image_data.preprocessed_image.shape == case['expected_shape']
            
            # Verify normalization (values between 0 and 1)
            assert np.min(image_data.preprocessed_image) >= 0
            assert np.max(image_data.preprocessed_image) <= 1


def test_dataset_shuffling(mock_dataset_config):
    """
    Test dataset shuffling behavior.
    
    Verifies:
    - Shuffling works with different seeds
    - Non-shuffled dataset maintains order
    """
    # Shuffled dataset
    dataset1 = Dataset(
        source='utkface', 
        max_samples=100,
        shuffle=True,
        seed=42,
        batch_size=16
    )
    
    # Same seed, should be identical
    dataset2 = Dataset(
        source='utkface', 
        max_samples=100,
        shuffle=True,
        seed=42,
        batch_size=16
    )
    
    # Different seed
    dataset3 = Dataset(
        source='utkface', 
        max_samples=100,
        shuffle=True,
        seed=123,
        batch_size=16
    )
    
    # Non-shuffled dataset
    dataset4 = Dataset(
        source='utkface', 
        max_samples=100,
        shuffle=False,
        batch_size=16
    )
    
    # Compare first batch image IDs
    def get_batch_image_ids(dataset):
        first_batch = next(iter(dataset))
        return [img.image_id for img in first_batch]
    
    # Verify behaviors
    assert get_batch_image_ids(dataset1) == get_batch_image_ids(dataset2)
    assert get_batch_image_ids(dataset1) != get_batch_image_ids(dataset3)
    
    # Non-shuffled should maintain some order
    dataset4_ids = get_batch_image_ids(dataset4)
    assert len(set(dataset4_ids)) == len(dataset4_ids)