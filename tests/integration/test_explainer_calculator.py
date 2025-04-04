"""Tests for integration between Explainer and Calculator components, ensuring explanation data flows correctly into bias analysis."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from biasx.explainers import Explainer
from biasx.calculators import Calculator
from biasx.types import (
    Gender, LandmarkerSource, CAMMethod, ThresholdMethod, DistanceMetric,
    FacialFeature, Box, Explanation, ImageData, FeatureAnalysis, DisparityScores
)


@pytest.mark.integration
@pytest.mark.explainer_calculator
def test_explainer_to_calculator_data_flow(create_test_explanation):
    """Test that Explainer outputs can be processed by Calculator."""
    # Setup
    calculator = Calculator(precision=3)
    
    # Generate explanations using the fixture
    explanations = [
        create_test_explanation(
            image_id=f"test_{i}",
            true_gender=Gender.MALE if i % 2 == 0 else Gender.FEMALE,
            pred_gender=Gender.FEMALE if i % 2 == 0 else Gender.MALE,  # All misclassified
            activation_boxes=[
                Box(10, 10, 20, 20, feature=FacialFeature.LEFT_EYE),
                Box(30, 10, 40, 20, feature=FacialFeature.RIGHT_EYE)
            ]
        )
        for i in range(4)
    ]
    
    # Calculate feature biases
    feature_analyses = calculator.calculate_feature_biases(explanations)
    
    # Calculate disparity scores
    disparity_scores = calculator.calculate_disparities(feature_analyses, explanations)
    
    # Assertions
    assert FacialFeature.LEFT_EYE in feature_analyses
    assert FacialFeature.RIGHT_EYE in feature_analyses
    
    # Verify feature analysis structure and values
    for feature, analysis in feature_analyses.items():
        assert isinstance(analysis.bias_score, float)
        assert 0 <= analysis.male_probability <= 1
        assert 0 <= analysis.female_probability <= 1
    
    # Verify disparity scores
    assert 0 <= disparity_scores.biasx <= 1
    assert 0 <= disparity_scores.equalized_odds <= 1


@pytest.mark.integration
@pytest.mark.explainer_calculator
@pytest.mark.parametrize("activated_features", [
    [FacialFeature.LEFT_EYE, FacialFeature.RIGHT_EYE],
    [FacialFeature.NOSE, FacialFeature.LIPS],
    [FacialFeature.LEFT_CHEEK, FacialFeature.RIGHT_CHEEK],
    [FacialFeature.FOREHEAD, FacialFeature.CHIN]
])
def test_feature_activation_patterns(activated_features, create_test_explanation):
    """Test processing of different feature activation patterns."""
    # Setup
    calculator = Calculator(precision=3)
    
    # Generate explanations with specific feature activations
    explanations = []
    for i in range(4):
        # For male images, activate first feature from set
        # For female images, activate second feature from set
        if i % 2 == 0:  # Male
            feature = activated_features[0]
            boxes = [Box(10, 10, 20, 20, feature=feature)]
            explanations.append(create_test_explanation(
                true_gender=Gender.MALE,
                pred_gender=Gender.FEMALE,
                activation_boxes=boxes
            ))
        else:  # Female
            feature = activated_features[1]
            boxes = [Box(30, 30, 40, 40, feature=feature)]
            explanations.append(create_test_explanation(
                true_gender=Gender.FEMALE, 
                pred_gender=Gender.MALE,
                activation_boxes=boxes
            ))
    
    # Calculate feature biases
    feature_analyses = calculator.calculate_feature_biases(explanations)
    
    # Calculate disparity scores
    disparity_scores = calculator.calculate_disparities(feature_analyses, explanations)
    
    # Verify the activated features are present in the analyses
    for feature in activated_features:
        assert feature in feature_analyses
        
    # Verify the feature analysis reflects different patterns for male vs female
    # First feature should have high male probability, low female probability
    assert feature_analyses[activated_features[0]].male_probability == 1.0
    assert feature_analyses[activated_features[0]].female_probability == 0.0
    
    # Second feature should have low male probability, high female probability
    assert feature_analyses[activated_features[1]].male_probability == 0.0
    assert feature_analyses[activated_features[1]].female_probability == 1.0
    
    # Both features should have high bias score due to gender disparity
    assert feature_analyses[activated_features[0]].bias_score == 1.0
    assert feature_analyses[activated_features[1]].bias_score == 1.0
    
    # Overall bias should be high
    assert disparity_scores.biasx > 0.5


@pytest.mark.integration
@pytest.mark.explainer_calculator
def test_explainer_to_calculator_with_no_features():
    """Test handling of explanations with no feature annotations."""
    # Setup
    calculator = Calculator(precision=3)
    
    # Create explanations with activation boxes but no feature annotations
    explanations = []
    for i in range(4):
        # Create ImageData
        image_data = ImageData(
            image_id=f"test_{i}",
            pil_image=None,  # Not needed for this test
            preprocessed_image=np.zeros((48, 48, 1)),
            width=48,
            height=48,
            gender=Gender.MALE if i % 2 == 0 else Gender.FEMALE
        )
        
        # Create Explanation with no feature annotations
        explanation = Explanation(
            image_data=image_data,
            predicted_gender=Gender.FEMALE if i % 2 == 0 else Gender.MALE,  # All misclassified
            prediction_confidence=0.8,
            activation_map=np.zeros((48, 48)),
            activation_boxes=[Box(10, 10, 20, 20)],  # No feature specified
            landmark_boxes=[]
        )
        
        explanations.append(explanation)
    
    # Calculate feature biases
    feature_analyses = calculator.calculate_feature_biases(explanations)
    
    # Calculate disparity scores
    disparity_scores = calculator.calculate_disparities(feature_analyses, explanations)
    
    # Verify no features were analyzed (due to missing annotations)
    assert len(feature_analyses) == 0
    
    # Verify disparity scores are default values
    assert disparity_scores.biasx == 0.0
    assert disparity_scores.equalized_odds >= 0.0  # Allow for balanced misclassifications


@pytest.mark.integration
@pytest.mark.explainer_calculator
def test_explainer_calculator_with_mixed_feature_presence(create_test_explanation):
    """Test handling of explanations with a mix of features present and absent."""
    # Setup
    calculator = Calculator(precision=3)
    
    # Create test explanations
    explanations = []
    
    # Explanation 1: Male misclassified as Female, with LEFT_EYE
    explanations.append(create_test_explanation(
        true_gender=Gender.MALE,
        pred_gender=Gender.FEMALE,
        activation_boxes=[Box(10, 10, 20, 20, feature=FacialFeature.LEFT_EYE)]
    ))
    
    # Explanation 2: Female misclassified as Male, with RIGHT_EYE
    explanations.append(create_test_explanation(
        true_gender=Gender.FEMALE,
        pred_gender=Gender.MALE,
        activation_boxes=[Box(10, 10, 20, 20, feature=FacialFeature.RIGHT_EYE)]
    ))
    
    # Explanation 3: Male misclassified as Female, NO FEATURE
    explanations.append(create_test_explanation(
        true_gender=Gender.MALE,
        pred_gender=Gender.FEMALE,
        activation_boxes=[Box(10, 10, 20, 20)]  # No feature
    ))
    
    # Explanation 4: Female correctly classified, with NOSE (should be ignored)
    explanations.append(create_test_explanation(
        true_gender=Gender.FEMALE,
        pred_gender=Gender.FEMALE,  # Correctly classified
        activation_boxes=[Box(10, 10, 20, 20, feature=FacialFeature.NOSE)]
    ))
    
    # Calculate feature biases
    feature_analyses = calculator.calculate_feature_biases(explanations)
    
    # Verify only the features in misclassifications are included
    assert FacialFeature.LEFT_EYE in feature_analyses
    assert FacialFeature.RIGHT_EYE in feature_analyses
    assert FacialFeature.NOSE not in feature_analyses  # Not in misclassification
    
    # Verify male probability for LEFT_EYE is 0.5 (1 out of 2 male misclassifications)
    assert feature_analyses[FacialFeature.LEFT_EYE].male_probability == 0.5
    
    # Verify female probability for RIGHT_EYE is 1.0 (1/1 female misclassifications)
    assert feature_analyses[FacialFeature.RIGHT_EYE].female_probability == 1.0