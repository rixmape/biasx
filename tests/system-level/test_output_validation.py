"""Output Validation Tests for BiasX Analysis Results."""

import os
import tempfile
import pytest
import numpy as np
import tensorflow as tf

from biasx import BiasAnalyzer
from biasx.types import (
    Gender, FacialFeature, Box, ImageData, 
    Explanation, FeatureAnalysis, DisparityScores, AnalysisResult
)


def create_test_model(save_path, input_shape=(48, 48, 1)):
    """Create a simple test model for validation."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    model.save(save_path)
    return save_path


def test_analysis_result_structure():
    """
    Test that analysis results have the expected structure.
    
    Verifies all required attributes exist and have the correct types.
    """
    # Create a temporary model for testing
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
        model_path = create_test_model(temp_model_file.name)
    
    try:
        # Prepare configuration
        config = {
            "model": {"path": model_path},
            "dataset": {
                "source": "utkface",
                "max_samples": 20,  # Small number for quick testing
                "image_width": 48,
                "image_height": 48,
            }
        }
        
        # Create analyzer and run analysis
        analyzer = BiasAnalyzer(config)
        result = analyzer.analyze()
        
        # Verify root structure
        assert isinstance(result, AnalysisResult), "Result should be an AnalysisResult instance"
        assert hasattr(result, 'explanations'), "Result should have 'explanations' attribute"
        assert hasattr(result, 'feature_analyses'), "Result should have 'feature_analyses' attribute"
        assert hasattr(result, 'disparity_scores'), "Result should have 'disparity_scores' attribute"
        
        # Verify explanations structure (if any)
        if result.explanations:
            explanation = result.explanations[0]
            assert isinstance(explanation, Explanation), "Explanation should be an Explanation instance"
            assert hasattr(explanation, 'image_data'), "Explanation should have 'image_data' attribute"
            assert hasattr(explanation, 'predicted_gender'), "Explanation should have 'predicted_gender' attribute"
            assert hasattr(explanation, 'prediction_confidence'), "Explanation should have 'prediction_confidence' attribute"
            assert hasattr(explanation, 'activation_map'), "Explanation should have 'activation_map' attribute"
            assert hasattr(explanation, 'activation_boxes'), "Explanation should have 'activation_boxes' attribute"
            assert hasattr(explanation, 'landmark_boxes'), "Explanation should have 'landmark_boxes' attribute"
            
            # Verify image_data structure
            assert isinstance(explanation.image_data, ImageData), "image_data should be an ImageData instance"
            assert hasattr(explanation.image_data, 'gender'), "image_data should have 'gender' attribute"
            
            # Verify activation boxes (if any)
            if explanation.activation_boxes:
                box = explanation.activation_boxes[0]
                assert isinstance(box, Box), "Activation box should be a Box instance"
                assert hasattr(box, 'min_x'), "Box should have 'min_x' attribute"
                assert hasattr(box, 'min_y'), "Box should have 'min_y' attribute"
                assert hasattr(box, 'max_x'), "Box should have 'max_x' attribute"
                assert hasattr(box, 'max_y'), "Box should have 'max_y' attribute"
                assert hasattr(box, 'feature'), "Box should have 'feature' attribute"
        
        # Verify feature analyses structure (if any)
        if result.feature_analyses:
            for feature, analysis in result.feature_analyses.items():
                assert isinstance(feature, FacialFeature), "Feature key should be a FacialFeature enum"
                assert isinstance(analysis, FeatureAnalysis), "Analysis should be a FeatureAnalysis instance"
                assert hasattr(analysis, 'bias_score'), "Analysis should have 'bias_score' attribute"
                assert hasattr(analysis, 'male_probability'), "Analysis should have 'male_probability' attribute"
                assert hasattr(analysis, 'female_probability'), "Analysis should have 'female_probability' attribute"
        
        # Verify disparity scores structure
        assert isinstance(result.disparity_scores, DisparityScores), "disparity_scores should be a DisparityScores instance"
        assert hasattr(result.disparity_scores, 'biasx'), "disparity_scores should have 'biasx' attribute"
        assert hasattr(result.disparity_scores, 'equalized_odds'), "disparity_scores should have 'equalized_odds' attribute"
    
    finally:
        # Clean up temporary model file
        os.unlink(model_path)


def test_analysis_result_values():
    """
    Test that analysis results have valid values.
    
    Verifies numerical values are in expected ranges and other values make sense.
    """
    # Create a temporary model for testing
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
        model_path = create_test_model(temp_model_file.name)
    
    try:
        # Prepare configuration
        config = {
            "model": {"path": model_path},
            "dataset": {
                "source": "utkface",
                "max_samples": 20,  # Small number for quick testing
                "image_width": 48,
                "image_height": 48,
            }
        }
        
        # Create analyzer and run analysis
        analyzer = BiasAnalyzer(config)
        result = analyzer.analyze()
        
        # Verify bias scores are in valid range [0, 1]
        assert 0 <= result.disparity_scores.biasx <= 1, "BiasX score should be between 0 and 1"
        assert 0 <= result.disparity_scores.equalized_odds <= 1, "Equalized odds score should be between 0 and 1"
        
        # Verify feature analyses values (if any)
        if result.feature_analyses:
            for feature, analysis in result.feature_analyses.items():
                assert 0 <= analysis.bias_score <= 1, f"Bias score for {feature} should be between 0 and 1"
                assert 0 <= analysis.male_probability <= 1, f"Male probability for {feature} should be between 0 and 1"
                assert 0 <= analysis.female_probability <= 1, f"Female probability for {feature} should be between 0 and 1"
        
        # Verify explanations values (if any)
        if result.explanations:
            for explanation in result.explanations:
                assert explanation.predicted_gender in [Gender.MALE, Gender.FEMALE], "Predicted gender should be a valid Gender enum"
                assert 0 <= explanation.prediction_confidence <= 1, "Prediction confidence should be between 0 and 1"
                
                # Verify activation map values
                assert isinstance(explanation.activation_map, np.ndarray), "Activation map should be a numpy array"
                
                # Verify activation boxes and landmark boxes values
                for box in explanation.activation_boxes + explanation.landmark_boxes:
                    assert box.min_x <= box.max_x, "Box min_x should be <= max_x"
                    assert box.min_y <= box.max_y, "Box min_y should be <= max_y"
    
    finally:
        # Clean up temporary model file
        os.unlink(model_path)


def test_feature_analysis_consistency():
    """
    Test consistency between feature analyses and explanations.
    
    Verifies that feature analyses are based on the explanations data.
    """
    # Create a temporary model for testing
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
        model_path = create_test_model(temp_model_file.name)
    
    try:
        # Prepare configuration
        config = {
            "model": {"path": model_path},
            "dataset": {
                "source": "utkface",
                "max_samples": 20,  # Small number for quick testing
                "image_width": 48,
                "image_height": 48,
            }
        }
        
        # Create analyzer and run analysis
        analyzer = BiasAnalyzer(config)
        result = analyzer.analyze()
        
        # Skip if no feature analyses or explanations
        if not result.feature_analyses or not result.explanations:
            pytest.skip("Not enough data to verify feature analysis consistency")
        
        # Get set of features mentioned in explanations
        features_in_explanations = set()
        for explanation in result.explanations:
            # Only consider misclassified examples
            if explanation.predicted_gender != explanation.image_data.gender:
                for box in explanation.activation_boxes:
                    if box.feature:
                        features_in_explanations.add(box.feature)
        
        # Check that all features in analyses are in explanations
        for feature in result.feature_analyses.keys():
            assert feature in features_in_explanations, f"Feature {feature} in analyses but not in explanations"
    
    finally:
        # Clean up temporary model file
        os.unlink(model_path)


def test_disparity_scores_calculation():
    """
    Test that disparity scores are calculated correctly.
    
    Verifies the relationship between feature bias scores and overall BiasX score.
    """
    # Create a temporary model for testing
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_model_file:
        model_path = create_test_model(temp_model_file.name)
    
    try:
        # Prepare configuration
        config = {
            "model": {"path": model_path},
            "dataset": {
                "source": "utkface",
                "max_samples": 20,  # Small number for quick testing
                "image_width": 48,
                "image_height": 48,
            }
        }
        
        # Create analyzer and run analysis
        analyzer = BiasAnalyzer(config)
        result = analyzer.analyze()
        
        # Skip if no feature analyses
        if not result.feature_analyses:
            pytest.skip("Not enough data to verify disparity score calculation")
        
        # Calculate expected BiasX score (average of feature bias scores)
        feature_bias_scores = [analysis.bias_score for analysis in result.feature_analyses.values()]
        expected_biasx = sum(feature_bias_scores) / len(feature_bias_scores)
        
        # Check that the BiasX score matches expectation (with a larger tolerance for rounding differences)
        assert abs(result.disparity_scores.biasx - expected_biasx) < 1e-3, \
            f"BiasX score {result.disparity_scores.biasx} doesn't match expected {expected_biasx}"
    
    finally:
        # Clean up temporary model file
        os.unlink(model_path)