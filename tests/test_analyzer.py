"""Tests for the BiasAnalyzer orchestration component in BiasX."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from biasx.analyzer import BiasAnalyzer
from biasx.types import (
    Gender, Age, Race, FacialFeature, Box, 
    ImageData, Explanation, FeatureAnalysis, 
    DisparityScores, AnalysisResult
)
from biasx.models import Model
from biasx.datasets import Dataset
from biasx.explainers import Explainer
from biasx.calculators import Calculator

# Import the real Config class for isinstance checks
from biasx.config import Config as ConfigClass


def test_analyzer_initialization():
    """Test BiasAnalyzer initialization with different input configurations."""
    # Test with a dictionary configuration
    config_dict = {
        "model": {
            "path": "test_model.h5",
            "inverted_classes": False
        },
        "dataset": {
            "source": "utkface",
            "max_samples": 10
        }
    }
    
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model'), \
         patch('biasx.analyzer.Dataset'), \
         patch('biasx.analyzer.Explainer'), \
         patch('biasx.analyzer.Calculator'), \
         patch('biasx.config.Config.create', return_value=ConfigClass(config_dict)):
        
        # Initialize with dictionary
        analyzer = BiasAnalyzer(config=config_dict)
        
        # Initialize with Config object
        config_obj = ConfigClass(config_dict)
        analyzer = BiasAnalyzer(config=config_obj)
        
        # Initialize with None (should create empty config)
        with patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "default.h5"}})):
            analyzer = BiasAnalyzer(config=None)


def test_analyze_batch():
    """Test the analyze_batch method with mocked components."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Create mock components
        mock_model = MagicMock()
        mock_model.predict.return_value = [(Gender.MALE, 0.8), (Gender.FEMALE, 0.9)]
        
        mock_explainer = MagicMock()
        mock_explainer.explain_batch.return_value = (
            [np.zeros((10, 10)), np.zeros((10, 10))],  # Activation maps
            [[Box(1, 2, 3, 4)], [Box(5, 6, 7, 8)]],    # Activation boxes
            [[Box(1, 2, 3, 4)], [Box(5, 6, 7, 8)]]     # Landmark boxes
        )
        
        # Return instances from mocks
        MockModel.return_value = mock_model
        MockExplainer.return_value = mock_explainer
        
        # Create analyzer
        analyzer = BiasAnalyzer(config={
            "model": {"path": "test_model.h5"},
            "dataset": {"source": "utkface"}
        })
        
        # Create test image data
        test_images = [
            ImageData(
                image_id="test1",
                pil_image=MagicMock(),
                preprocessed_image=np.zeros((48, 48, 1)),
                gender=Gender.MALE
            ),
            ImageData(
                image_id="test2",
                pil_image=MagicMock(),
                preprocessed_image=np.zeros((48, 48, 1)),
                gender=Gender.FEMALE
            )
        ]
        
        # Call analyze_batch
        results = analyzer.analyze_batch(test_images)
        
        # Verify results
        assert len(results) == 2
        assert results[0].image_data == test_images[0]
        assert results[0].predicted_gender == Gender.MALE
        assert results[0].prediction_confidence == 0.8
        assert results[1].image_data == test_images[1]
        assert results[1].predicted_gender == Gender.FEMALE
        assert results[1].prediction_confidence == 0.9
        
        # Verify component interactions
        mock_model.predict.assert_called_once()
        mock_explainer.explain_batch.assert_called_once()


def test_analyze_full_pipeline():
    """Test the complete analyze method with mocked components."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Create mock components
        mock_model = MagicMock()
        mock_model.predict.return_value = [(Gender.MALE, 0.8), (Gender.FEMALE, 0.9)]
        
        mock_explainer = MagicMock()
        mock_explainer.explain_batch.return_value = (
            [np.zeros((10, 10)), np.zeros((10, 10))],  # Activation maps
            [[Box(1, 2, 3, 4)], [Box(5, 6, 7, 8)]],    # Activation boxes
            [[Box(1, 2, 3, 4)], [Box(5, 6, 7, 8)]]     # Landmark boxes
        )
        
        mock_dataset = MagicMock()
        # Setup dataset to return two batches of data
        batch1 = [
            ImageData(image_id="test1", gender=Gender.MALE),
            ImageData(image_id="test2", gender=Gender.FEMALE)
        ]
        batch2 = [
            ImageData(image_id="test3", gender=Gender.MALE)
        ]
        mock_dataset.__iter__.return_value = iter([batch1, batch2])
        
        mock_calculator = MagicMock()
        feature_analyses = {
            FacialFeature.NOSE: FeatureAnalysis(
                feature=FacialFeature.NOSE,
                bias_score=0.3,
                male_probability=0.7,
                female_probability=0.4
            )
        }
        mock_calculator.calculate_feature_biases.return_value = feature_analyses
        mock_calculator.calculate_disparities.return_value = DisparityScores(
            biasx=0.3,
            equalized_odds=0.2
        )
        
        # Return instances from mocks
        MockModel.return_value = mock_model
        MockExplainer.return_value = mock_explainer
        MockDataset.return_value = mock_dataset
        MockCalculator.return_value = mock_calculator
        
        # Create analyzer
        analyzer = BiasAnalyzer(config={
            "model": {"path": "test_model.h5"},
            "dataset": {"source": "utkface"}
        })
        
        # Run the full analysis
        result = analyzer.analyze()
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert len(result.explanations) == 3  # 2 from first batch, 1 from second
        assert result.feature_analyses == feature_analyses
        assert result.disparity_scores.biasx == 0.3
        assert result.disparity_scores.equalized_odds == 0.2
        
        # Verify component interactions
        assert mock_model.predict.call_count == 2  # Once per batch
        assert mock_explainer.explain_batch.call_count == 2  # Once per batch
        mock_calculator.calculate_feature_biases.assert_called_once()
        mock_calculator.calculate_disparities.assert_called_once()


def test_analyze_empty_dataset():
    """Test analyze method with an empty dataset."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Setup empty dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__.return_value = iter([])  # No batches
        MockDataset.return_value = mock_dataset
        
        # Create analyzer
        analyzer = BiasAnalyzer(config={
            "model": {"path": "test_model.h5"},
            "dataset": {"source": "utkface"}
        })
        
        # Run the analysis
        result = analyzer.analyze()
        
        # Verify empty result
        assert isinstance(result, AnalysisResult)
        assert len(result.explanations) == 0
        assert len(result.feature_analyses) == 0
        assert result.disparity_scores.biasx == 0.0
        assert result.disparity_scores.equalized_odds == 0.0


def test_analyze_empty_batch():
    """Test analyze_batch method with an empty batch."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Create mock components
        mock_model = MagicMock()
        mock_explainer = MagicMock()
        
        # Return instances from mocks
        MockModel.return_value = mock_model
        MockExplainer.return_value = mock_explainer
        
        # Create analyzer
        analyzer = BiasAnalyzer(config={
            "model": {"path": "test_model.h5"},
            "dataset": {"source": "utkface"}
        })
        
        # Call analyze_batch with empty list
        results = analyzer.analyze_batch([])
        
        # Verify results
        assert results == []
        
        # Verify component interactions
        mock_model.predict.assert_not_called()
        mock_explainer.explain_batch.assert_not_called()


def test_from_file():
    """Test creating a BiasAnalyzer from a config file."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model'), \
         patch('biasx.analyzer.Dataset'), \
         patch('biasx.analyzer.Explainer'), \
         patch('biasx.analyzer.Calculator'), \
         patch('biasx.config.Config.from_file') as mock_from_file:
        
        mock_config = ConfigClass({"model": {"path": "test_model.h5"}})
        mock_from_file.return_value = mock_config
        
        # Call from_file
        analyzer = BiasAnalyzer.from_file("config.json")
        
        # Verify Config.from_file was called with the file path
        mock_from_file.assert_called_once_with("config.json")


def test_end_to_end_workflow():
    """
    Test a simplified end-to-end workflow with controlled test data.
    
    This test creates a minimal pipeline where we can track data flow
    through all components without requiring actual models or data.
    """
    # Mock core components but with functional behavior
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Setup model to produce consistent predictions
        mock_model = MagicMock()
        
        # Modified predict_func that works with numpy arrays
        def predict_func(images):
            # Determine the number of images in the batch
            if isinstance(images, list):
                batch_size = len(images)
            else:  # numpy array
                batch_size = images.shape[0] if len(images.shape) > 3 else 1
                
            # For the test data setup, we know the first item is male and second is female
            # So return predictions accordingly
            return [(Gender.FEMALE, 0.8) if i == 0 else (Gender.MALE, 0.8) 
                    for i in range(batch_size)]
                    
        mock_model.predict.side_effect = predict_func
        MockModel.return_value = mock_model
        
        # Setup dataset with controlled test data
        mock_dataset = MagicMock()
        test_data = [
            # Batch 1: one male, one female
            [
                ImageData(
                    image_id="male1",
                    pil_image=MagicMock(),
                    preprocessed_image=np.zeros((48, 48, 1)),
                    gender=Gender.MALE
                ),
                ImageData(
                    image_id="female1",
                    pil_image=MagicMock(),
                    preprocessed_image=np.zeros((48, 48, 1)),
                    gender=Gender.FEMALE
                )
            ]
        ]
        mock_dataset.__iter__.return_value = iter(test_data)
        MockDataset.return_value = mock_dataset
        
        # Setup explainer with realistic behavior
        mock_explainer = MagicMock()
        def explain_batch_func(pil_images, preprocessed_images, model, target_classes):
            # Create activation boxes with features
            activation_boxes = []
            for i, target in enumerate(target_classes):
                if target == Gender.MALE:
                    # For predicted males, activate nose and eyes
                    activation_boxes.append([
                        Box(10, 10, 20, 20, feature=FacialFeature.NOSE),
                        Box(5, 5, 15, 15, feature=FacialFeature.LEFT_EYE)
                    ])
                else:
                    # For predicted females, activate lips
                    activation_boxes.append([
                        Box(30, 30, 40, 40, feature=FacialFeature.LIPS)
                    ])
            
            # Create dummy activation maps and landmark boxes
            activation_maps = [np.zeros((10, 10)) for _ in target_classes]
            landmark_boxes = [[] for _ in target_classes]
            
            return activation_maps, activation_boxes, landmark_boxes
            
        mock_explainer.explain_batch.side_effect = explain_batch_func
        MockExplainer.return_value = mock_explainer
        
        # Setup calculator with functional behavior
        mock_calculator = MagicMock()
        
        def calculate_feature_biases_func(explanations):
            # Count feature occurrences by gender
            feature_counts = {}
            for exp in explanations:
                if exp.predicted_gender != exp.image_data.gender:  # Misclassified
                    for box in exp.activation_boxes:
                        if box.feature:
                            if box.feature not in feature_counts:
                                feature_counts[box.feature] = {Gender.MALE: 0, Gender.FEMALE: 0}
                            feature_counts[box.feature][exp.image_data.gender] += 1
            
            # Create feature analyses
            result = {}
            for feature, counts in feature_counts.items():
                male_count = len([e for e in explanations 
                                  if e.image_data.gender == Gender.MALE 
                                  and e.predicted_gender != e.image_data.gender])
                female_count = len([e for e in explanations 
                                   if e.image_data.gender == Gender.FEMALE 
                                   and e.predicted_gender != e.image_data.gender])
                
                male_prob = counts[Gender.MALE] / male_count if male_count > 0 else 0
                female_prob = counts[Gender.FEMALE] / female_count if female_count > 0 else 0
                
                result[feature] = FeatureAnalysis(
                    feature=feature,
                    bias_score=abs(male_prob - female_prob),
                    male_probability=male_prob,
                    female_probability=female_prob
                )
            
            return result
            
        mock_calculator.calculate_feature_biases.side_effect = calculate_feature_biases_func
        
        def calculate_disparities_func(feature_analyses, explanations):
            # Simple implementation for testing
            biasx = sum(fa.bias_score for fa in feature_analyses.values()) / len(feature_analyses) if feature_analyses else 0.0
            return DisparityScores(biasx=biasx, equalized_odds=1.0)  # All examples misclassified
            
        mock_calculator.calculate_disparities.side_effect = calculate_disparities_func
        MockCalculator.return_value = mock_calculator
        
        # Create analyzer
        analyzer = BiasAnalyzer(config={
            "model": {"path": "test_model.h5"},
            "dataset": {"source": "utkface"}
        })
        
        # Run the complete analysis
        result = analyzer.analyze()
        
        # Verify the data flow and structure
        assert isinstance(result, AnalysisResult)
        assert len(result.explanations) == 2
        
        # Verify explanations contain expected data
        assert result.explanations[0].image_data.gender == Gender.MALE
        assert result.explanations[0].predicted_gender == Gender.FEMALE
        assert len(result.explanations[0].activation_boxes) == 1  # Lips for predicted female
        
        assert result.explanations[1].image_data.gender == Gender.FEMALE
        assert result.explanations[1].predicted_gender == Gender.MALE
        assert len(result.explanations[1].activation_boxes) == 2  # Nose and eye for predicted male
        
        # Verify feature analysis captures the pattern
        assert FacialFeature.LIPS in result.feature_analyses
        assert FacialFeature.NOSE in result.feature_analyses
        assert FacialFeature.LEFT_EYE in result.feature_analyses
        
        # Verify the bias scores reflect our constructed scenario
        assert result.disparity_scores.biasx > 0
        assert result.disparity_scores.equalized_odds == 1.0


def test_batch_buffer_handling():
    """Test the buffer handling and batch processing in analyze method."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Setup mocks with minimal functionality
        mock_model = MagicMock()
        mock_model.predict.return_value = [(Gender.MALE, 0.8)]
        
        mock_explainer = MagicMock()
        mock_explainer.explain_batch.return_value = (
            [np.zeros((10, 10))],  # Activation maps
            [[Box(1, 2, 3, 4)]],   # Activation boxes
            [[Box(1, 2, 3, 4)]]    # Landmark boxes
        )
        
        mock_calculator = MagicMock()
        mock_calculator.calculate_feature_biases.return_value = {}
        mock_calculator.calculate_disparities.return_value = DisparityScores()
        
        # Setup a dataset with multiple small batches
        mock_dataset = MagicMock()
        batch1 = [ImageData(image_id="test1", gender=Gender.MALE)]
        batch2 = [ImageData(image_id="test2", gender=Gender.MALE)]
        batch3 = [ImageData(image_id="test3", gender=Gender.MALE)]
        mock_dataset.__iter__.return_value = iter([batch1, batch2, batch3])
        
        # Set returns
        MockModel.return_value = mock_model
        MockExplainer.return_value = mock_explainer
        MockDataset.return_value = mock_dataset
        MockCalculator.return_value = mock_calculator
        
        # Create analyzer with small buffer
        analyzer = BiasAnalyzer(config={
            "model": {"path": "test_model.h5"},
            "dataset": {"source": "utkface"}
        }, batch_size=2)  # Buffer size will be 2*2 = 4
        
        # Run analysis
        result = analyzer.analyze()
        
        # Verify all batches were analyzed
        assert len(result.explanations) == 3
        
        # Verify buffer was properly filled and flushed
        assert mock_model.predict.call_count == 3
        assert mock_explainer.explain_batch.call_count == 3
        assert mock_calculator.calculate_feature_biases.call_count == 1
        assert mock_calculator.calculate_disparities.call_count == 1

def test_buffer_size_threshold():
    """Test the buffer size threshold logic in the analyze method."""
    with patch('biasx.analyzer.Config', ConfigClass), \
         patch('biasx.analyzer.Model') as MockModel, \
         patch('biasx.analyzer.Dataset') as MockDataset, \
         patch('biasx.analyzer.Explainer') as MockExplainer, \
         patch('biasx.analyzer.Calculator') as MockCalculator, \
         patch('biasx.config.Config.create', return_value=ConfigClass({"model": {"path": "test_model.h5"}})):
        
        # Setup model with simple behavior
        mock_model = MagicMock()
        mock_model.predict.return_value = [(Gender.MALE, 0.8)]
        MockModel.return_value = mock_model
        
        # Setup explainer to return consistent data
        mock_explainer = MagicMock()
        mock_explainer.explain_batch.return_value = (
            [np.zeros((10, 10))],
            [[Box(1, 2, 3, 4)]],
            [[Box(1, 2, 3, 4)]]
        )
        MockExplainer.return_value = mock_explainer
        
        # Setup calculator with minimal behavior
        mock_calculator = MagicMock()
        mock_calculator.calculate_feature_biases.return_value = {}
        mock_calculator.calculate_disparities.return_value = DisparityScores()
        MockCalculator.return_value = mock_calculator
        
        # Create a dataset with exactly 1 batch that will trigger analyze_batch once
        mock_dataset = MagicMock()
        batch = [ImageData(image_id="test", gender=Gender.MALE)]
        mock_dataset.__iter__.return_value = iter([batch])
        MockDataset.return_value = mock_dataset
        
        # Create our analyzer with a tiny buffer size to ensure we trigger the flushing
        # This is crucial for hitting the target lines
        analyzer = BiasAnalyzer(config={"model": {"path": "test_model.h5"}}, batch_size=1)
        buffer_size = max(100, analyzer.batch_size * 2)  # This should be 100
        
        # We'll need to replace analyze_batch to control what goes into the buffer
        original_analyze_batch = analyzer.analyze_batch
        
        def mock_analyze_batch(batch_data):
            # Return exactly buffer_size explanations to trigger the threshold
            return [
                Explanation(
                    image_data=ImageData(image_id=f"test{i}", gender=Gender.MALE),
                    predicted_gender=Gender.MALE,
                    prediction_confidence=0.8,
                    activation_map=np.zeros((10, 10)),
                    activation_boxes=[Box(1, 2, 3, 4)],
                    landmark_boxes=[Box(1, 2, 3, 4)]
                )
                for i in range(buffer_size)  # Exactly buffer_size explanations
            ]
        
        analyzer.analyze_batch = mock_analyze_batch
        
        # Run the analysis
        result = analyzer.analyze()