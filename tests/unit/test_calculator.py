"""Tests for the bias calculation functionality in BiasX."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from biasx.types import (
    Gender, Age, Race, FacialFeature, Box, 
    ImageData, Explanation, FeatureAnalysis, DisparityScores
)
from biasx.calculators import Calculator


# Helper functions for test data creation
def create_image_data(id="test", gender=Gender.MALE):
    """Create synthetic ImageData object for testing."""
    return ImageData(
        image_id=id,
        gender=gender,
        age=Age.RANGE_20_29,
        race=Race.WHITE
    )


def create_explanation(
    image_id="test",
    true_gender=Gender.MALE,
    pred_gender=Gender.MALE,
    confidence=0.8,
    activation_boxes=None
):
    """Create a synthetic explanation for testing with specified properties."""
    image_data = create_image_data(id=image_id, gender=true_gender)
    
    if activation_boxes is None:
        activation_boxes = []
    
    return Explanation(
        image_data=image_data,
        predicted_gender=pred_gender,
        prediction_confidence=confidence,
        activation_map=np.zeros((10, 10)),
        activation_boxes=activation_boxes,
        landmark_boxes=[]
    )


# Tests for feature bias calculation
def test_calculate_feature_biases_basic():
    """Test basic feature bias calculation functionality."""
    calculator = Calculator(precision=3)
    
    # Create test data with known bias patterns
    explanations = [
        # Misclassified male with left_eye activation
        create_explanation(
            image_id="male1", 
            true_gender=Gender.MALE, 
            pred_gender=Gender.FEMALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.LEFT_EYE)]
        ),
        # Misclassified male with right_eye activation
        create_explanation(
            image_id="male2", 
            true_gender=Gender.MALE, 
            pred_gender=Gender.FEMALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.RIGHT_EYE)]
        ),
        # Misclassified female with left_eye activation
        create_explanation(
            image_id="female1", 
            true_gender=Gender.FEMALE, 
            pred_gender=Gender.MALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.LEFT_EYE)]
        ),
        # Correctly classified male (should not affect calculation)
        create_explanation(
            image_id="male3", 
            true_gender=Gender.MALE, 
            pred_gender=Gender.MALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.NOSE)]
        )
    ]
    
    # Calculate feature biases
    results = calculator.calculate_feature_biases(explanations)
    
    # Verify expected results
    assert FacialFeature.LEFT_EYE in results
    assert FacialFeature.RIGHT_EYE in results
    
    # Expected bias for LEFT_EYE: 0.5 (male) - 1.0 (female) = 0.5
    assert results[FacialFeature.LEFT_EYE].bias_score == 0.5
    assert results[FacialFeature.LEFT_EYE].male_probability == 0.5
    assert results[FacialFeature.LEFT_EYE].female_probability == 1.0
    
    # Expected bias for RIGHT_EYE: 0.5 (male) - 0.0 (female) = 0.5
    assert results[FacialFeature.RIGHT_EYE].bias_score == 0.5
    assert results[FacialFeature.RIGHT_EYE].male_probability == 0.5
    assert results[FacialFeature.RIGHT_EYE].female_probability == 0.0
    
    # NOSE shouldn't be in results as it wasn't activated in misclassifications
    assert FacialFeature.NOSE not in results


def test_calculate_feature_biases_with_precision():
    """Test feature bias calculation with different precision settings."""
    # Test with precision=2
    calculator = Calculator(precision=2)
    
    explanations = [
        # Misclassified male with activation on NOSE
        create_explanation(
            true_gender=Gender.MALE, 
            pred_gender=Gender.FEMALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.NOSE)]
        ),
        # Misclassified female with no activation on NOSE
        create_explanation(
            true_gender=Gender.FEMALE, 
            pred_gender=Gender.MALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.LEFT_EYE)]
        )
    ]
    
    results = calculator.calculate_feature_biases(explanations)
    # Expected bias for NOSE: 1.0 (male) - 0.0 (female) = 1.0
    assert results[FacialFeature.NOSE].bias_score == 1.0


def test_calculate_feature_biases_no_misclassifications():
    """Test feature bias calculation with no misclassifications."""
    calculator = Calculator(precision=3)
    
    # All examples are correctly classified
    explanations = [
        create_explanation(
            true_gender=Gender.MALE, 
            pred_gender=Gender.MALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.NOSE)]
        ),
        create_explanation(
            true_gender=Gender.FEMALE, 
            pred_gender=Gender.FEMALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.LEFT_EYE)]
        )
    ]
    
    results = calculator.calculate_feature_biases(explanations)
    
    # Should return empty dict as no misclassifications
    assert len(results) == 0


def test_calculate_feature_biases_all_misclassifications():
    """Test feature bias calculation with all examples misclassified."""
    calculator = Calculator(precision=3)
    
    # All examples are misclassified
    explanations = [
        create_explanation(
            true_gender=Gender.MALE, 
            pred_gender=Gender.FEMALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.NOSE)]
        ),
        create_explanation(
            true_gender=Gender.FEMALE, 
            pred_gender=Gender.MALE,
            activation_boxes=[Box(0, 0, 10, 10, feature=FacialFeature.NOSE)]
        )
    ]
    
    results = calculator.calculate_feature_biases(explanations)
    
    # NOSE should have equal activation for both genders
    assert FacialFeature.NOSE in results
    assert results[FacialFeature.NOSE].male_probability == 1.0
    assert results[FacialFeature.NOSE].female_probability == 1.0
    assert results[FacialFeature.NOSE].bias_score == 0.0  # No bias when perfectly balanced


def test_calculate_feature_biases_no_feature_annotations():
    """Test feature bias calculation with activation boxes but no feature annotations."""
    calculator = Calculator(precision=3)
    
    # Misclassifications with activation boxes but no feature annotations
    explanations = [
        create_explanation(
            true_gender=Gender.MALE, 
            pred_gender=Gender.FEMALE,
            activation_boxes=[Box(0, 0, 10, 10)]  # No feature annotation
        ),
        create_explanation(
            true_gender=Gender.FEMALE, 
            pred_gender=Gender.MALE,
            activation_boxes=[Box(0, 0, 10, 10)]  # No feature annotation
        )
    ]
    
    results = calculator.calculate_feature_biases(explanations)
    
    # Should return empty dict as no feature annotations
    assert len(results) == 0


# Tests for disparity score calculation
def test_calculate_disparities_basic():
    """Test basic disparity score calculation."""
    calculator = Calculator(precision=3)
    
    # Create a mock feature analyses dictionary
    feature_analyses = {
        FacialFeature.LEFT_EYE: FeatureAnalysis(
            feature=FacialFeature.LEFT_EYE,
            bias_score=0.5,
            male_probability=0.7,
            female_probability=0.2
        ),
        FacialFeature.RIGHT_EYE: FeatureAnalysis(
            feature=FacialFeature.RIGHT_EYE,
            bias_score=0.3,
            male_probability=0.4,
            female_probability=0.1
        )
    }
    
    # Create some test explanations for equalized odds calculation
    explanations = [
        # Male correctly classified
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        # Male incorrectly classified
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.FEMALE),
        # Female correctly classified
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.FEMALE),
        # Female incorrectly classified
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.MALE)
    ]
    
    # Calculate disparities
    disparities = calculator.calculate_disparities(feature_analyses, explanations)
    
    # Verify results
    assert disparities.biasx == 0.4  # Average of 0.5 and 0.3
    assert isinstance(disparities.equalized_odds, float)


def test_calculate_disparities_empty_features():
    """Test disparity score calculation with empty feature analyses."""
    calculator = Calculator(precision=3)
    
    # Empty feature analyses
    feature_analyses = {}
    
    # Some explanations
    explanations = [
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.FEMALE)
    ]
    
    # Calculate disparities
    disparities = calculator.calculate_disparities(feature_analyses, explanations)
    
    # Should return zeros
    assert disparities.biasx == 0.0
    assert disparities.equalized_odds == 0.0


def test_calculate_equalized_odds_score():
    """Test equalized odds score calculation with known outcomes."""
    calculator = Calculator(precision=3)
    
    # Create explanations with known pattern:
    # Male: 8 correct, 2 incorrect
    # Female: 6 correct, 4 incorrect
    explanations = []
    
    # Add 8 correctly classified males
    for i in range(8):
        explanations.append(
            create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE)
        )
    
    # Add 2 incorrectly classified males
    for i in range(2):
        explanations.append(
            create_explanation(true_gender=Gender.MALE, pred_gender=Gender.FEMALE)
        )
    
    # Add 6 correctly classified females
    for i in range(6):
        explanations.append(
            create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.FEMALE)
        )
    
    # Add 4 incorrectly classified females
    for i in range(4):
        explanations.append(
            create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.MALE)
        )
    
    # Expected TPR for Male: 8/10 = 0.8
    # Expected TPR for Female: 6/10 = 0.6
    # TPR disparity: |0.8 - 0.6| = 0.2
    
    # Expected FPR:
    # Male as Female (FP): 2/10 = 0.2
    # Female as Male (FP): 4/10 = 0.4
    # FPR disparity: |0.2 - 0.4| = 0.2
    
    # Max disparity: max(0.2, 0.2) = 0.2
    
    # Calculate equalized odds score
    score = calculator._calculate_equalized_odds_score(explanations)
    
    # Verify result (with small tolerance for floating point)
    assert abs(score - 0.2) < 1e-6


def test_calculate_equalized_odds_score_perfect_classifier():
    """Test equalized odds score with a perfect classifier (no errors)."""
    calculator = Calculator(precision=3)
    
    # Create explanations where all predictions are correct
    explanations = [
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.FEMALE),
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.FEMALE)
    ]
    
    # Calculate equalized odds score
    score = calculator._calculate_equalized_odds_score(explanations)
    
    # Perfect classifier should have 0 disparity
    assert score == 0.0


def test_calculate_equalized_odds_score_single_gender():
    """Test equalized odds score when only one gender is present."""
    calculator = Calculator(precision=3)
    
    # Create explanations with only male examples
    explanations = [
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.FEMALE)
    ]
    
    # Calculate equalized odds score
    score = calculator._calculate_equalized_odds_score(explanations)
    
    # Expect 0.5 based on implementation behavior with single gender
    assert score == 0.5


def test_calculate_equalized_odds_score_extreme_bias():
    """Test equalized odds score with extreme bias (always predicts one gender)."""
    calculator = Calculator(precision=3)
    
    # Create explanations where model always predicts male
    explanations = [
        # Males correctly classified
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE),
        # Females incorrectly classified
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.MALE),
        create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.MALE)
    ]
    
    # Calculate equalized odds score
    score = calculator._calculate_equalized_odds_score(explanations)
    
    # Extreme bias should have max disparity of 1.0
    assert score == 1.0


@pytest.mark.parametrize("male_correct,male_incorrect,female_correct,female_incorrect,expected_score", [
    (10, 0, 10, 0, 0.0),  # Perfect classifier
    (8, 2, 8, 2, 0.0),    # Balanced errors
    (8, 2, 6, 4, 0.2),    # Imbalanced errors
    (10, 0, 0, 10, 1.0),  # All females misclassified
    (0, 10, 10, 0, 1.0),  # All males misclassified
    (0, 10, 0, 10, 0.0),  # All examples misclassified
    (8, 2, 0, 0, 0.8),    # Only males (with errors)
    (0, 0, 8, 2, 0.8)     # Only females (with errors)
])
def test_equalized_odds_parametrized(
    male_correct, male_incorrect, female_correct, female_incorrect, expected_score
):
    """Test equalized odds score with various scenarios."""
    calculator = Calculator(precision=3)
    
    explanations = []
    
    # Add correct male classifications
    for i in range(male_correct):
        explanations.append(
            create_explanation(true_gender=Gender.MALE, pred_gender=Gender.MALE)
        )
    
    # Add incorrect male classifications
    for i in range(male_incorrect):
        explanations.append(
            create_explanation(true_gender=Gender.MALE, pred_gender=Gender.FEMALE)
        )
    
    # Add correct female classifications
    for i in range(female_correct):
        explanations.append(
            create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.FEMALE)
        )
    
    # Add incorrect female classifications
    for i in range(female_incorrect):
        explanations.append(
            create_explanation(true_gender=Gender.FEMALE, pred_gender=Gender.MALE)
        )
    
    # Calculate score
    score = calculator._calculate_equalized_odds_score(explanations)
    
    # Verify result
    assert abs(score - expected_score) < 1e-6


def test_get_feature_activation_map():
    """Test the feature activation map generation."""
    calculator = Calculator(precision=3)
    
    # Create explanations with known feature activations for misclassifications
    explanations = [
        # Correctly classified (should be ignored)
        create_explanation(
            true_gender=Gender.MALE,
            pred_gender=Gender.MALE,
            activation_boxes=[
                Box(0, 0, 10, 10, feature=FacialFeature.NOSE),
                Box(0, 0, 10, 10, feature=FacialFeature.LEFT_EYE)
            ]
        ),
        # Misclassified male with LEFT_EYE and NOSE
        create_explanation(
            true_gender=Gender.MALE,
            pred_gender=Gender.FEMALE,
            activation_boxes=[
                Box(0, 0, 10, 10, feature=FacialFeature.NOSE),
                Box(0, 0, 10, 10, feature=FacialFeature.LEFT_EYE)
            ]
        ),
        # Misclassified female with NOSE only
        create_explanation(
            true_gender=Gender.FEMALE,
            pred_gender=Gender.MALE,
            activation_boxes=[
                Box(0, 0, 10, 10, feature=FacialFeature.NOSE)
            ]
        )
    ]
    
    # Generate activation map
    activation_map = calculator._get_feature_activation_map(explanations)
    
    # Verify results
    assert FacialFeature.NOSE in activation_map
    assert FacialFeature.LEFT_EYE in activation_map
    
    # Verify male probabilities
    assert activation_map[FacialFeature.NOSE][Gender.MALE] == 1.0  # 1/1 misclassified males
    assert activation_map[FacialFeature.LEFT_EYE][Gender.MALE] == 1.0  # 1/1 misclassified males
    
    # Verify female probabilities
    assert activation_map[FacialFeature.NOSE][Gender.FEMALE] == 1.0  # 1/1 misclassified females
    assert activation_map[FacialFeature.LEFT_EYE][Gender.FEMALE] == 0.0  # 0/1 misclassified females