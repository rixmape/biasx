import pytest
import numpy as np
from PIL import Image

# Import the types to test
from biasx.types import (
    Gender,
    FacialFeature,
    Box,
    ImageData,
    Explanation,
    AnalysisResult,
    FeatureAnalysis,
    DisparityScores
)


class TestEnums:
    """Tests for the enum type definitions."""

    def test_gender_enum(self):
        """Test the Gender enum class."""
        # Check all expected values exist
        assert Gender.MALE is not None
        assert Gender.FEMALE is not None
        
        # Test conversion from int values
        assert Gender(0) == Gender.MALE
        assert Gender(1) == Gender.FEMALE
        
        # Test with invalid value
        with pytest.raises(ValueError):
            Gender(99)
    
    def test_facial_feature_enum(self):
        """Test the FacialFeature enum class."""
        # Check all expected values exist
        assert FacialFeature.LEFT_EYE is not None
        assert FacialFeature.RIGHT_EYE is not None
        assert FacialFeature.NOSE is not None
        assert FacialFeature.LIPS is not None
        assert FacialFeature.LEFT_CHEEK is not None
        assert FacialFeature.RIGHT_CHEEK is not None
        assert FacialFeature.CHIN is not None
        assert FacialFeature.FOREHEAD is not None
        assert FacialFeature.LEFT_EYEBROW is not None
        assert FacialFeature.RIGHT_EYEBROW is not None
        
        # Test string representation
        assert FacialFeature.NOSE.value == "nose"


class TestBoundingBox:
    """Tests for the Box class."""

    def test_bounding_box_creation(self):
        """Test creating a Box object."""
        # Create with min_x, min_y, max_x, max_y
        bbox = Box(10, 20, 50, 60)
        
        # Check attributes
        assert bbox.min_x == 10
        assert bbox.min_y == 20
        assert bbox.max_x == 50
        assert bbox.max_y == 60
    
    def test_bounding_box_properties(self):
        """Test the computed properties of Box."""
        bbox = Box(10, 20, 50, 60)
        
        # Test center property
        assert bbox.center == (30.0, 40.0)
        
        # Test area property
        assert bbox.area == 1600.0
    
    def test_bounding_box_with_feature(self):
        """Test creating a Box with a feature label."""
        bbox = Box(10, 20, 50, 60, feature=FacialFeature.NOSE)
        
        # Check feature attribute
        assert bbox.feature == FacialFeature.NOSE
        
        # Create a box without feature
        bbox_no_feature = Box(10, 20, 50, 60)
        assert bbox_no_feature.feature is None


class TestImageData:
    """Tests for the ImageData class."""
    
    def test_image_data_creation(self):
        """Test creating an ImageData object."""
        # Create a simple image data object
        img_data = ImageData(
            image_id="test_image_001",
            pil_image=Image.new('RGB', (100, 100)),
            width=100,
            height=100,
            gender=Gender.FEMALE,
            age=None,
            race=None
        )
        
        # Check attributes
        assert img_data.image_id == "test_image_001"
        assert img_data.pil_image is not None
        assert img_data.width == 100
        assert img_data.height == 100
        assert img_data.gender == Gender.FEMALE
        assert img_data.age is None
        assert img_data.race is None


class TestExplanation:
    """Tests for the Explanation class."""

    def test_explanation_creation(self):
        """Test creating an Explanation object."""
        # Create image data first
        img_data = ImageData(
            image_id="test_image_001",
            pil_image=Image.new('RGB', (100, 100)),
            width=100,
            height=100,
            gender=Gender.FEMALE
        )
        
        # Sample activation map
        activation_map = np.zeros((10, 10))
        activation_map[3:7, 3:7] = 1.0  # Activate the center region
        
        # Feature activation boxes and landmark boxes
        activation_boxes = [
            Box(3, 3, 7, 7, feature=FacialFeature.NOSE)
        ]
        
        landmark_boxes = [
            Box(2, 2, 8, 4, feature=FacialFeature.LEFT_EYE),
            Box(4, 4, 6, 6, feature=FacialFeature.NOSE),
            Box(3, 7, 7, 9, feature=FacialFeature.LIPS)
        ]
        
        # Create the explanation
        explanation = Explanation(
            image_data=img_data,
            predicted_gender=Gender.MALE,
            prediction_confidence=0.85,
            activation_map=activation_map,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes
        )
        
        # Check the basic attributes
        assert explanation.image_data == img_data
        assert explanation.predicted_gender == Gender.MALE
        assert explanation.prediction_confidence == 0.85
        assert np.array_equal(explanation.activation_map, activation_map)
        assert explanation.activation_boxes == activation_boxes
        assert explanation.landmark_boxes == landmark_boxes


class TestFeatureAnalysis:
    """Tests for the FeatureAnalysis class."""
    
    def test_feature_analysis_creation(self):
        """Test creating a FeatureAnalysis object."""
        # Create a feature analysis object
        feature_analysis = FeatureAnalysis(
            feature=FacialFeature.NOSE,
            bias_score=0.25,
            male_probability=0.65,
            female_probability=0.35
        )
        
        # Check attributes
        assert feature_analysis.feature == FacialFeature.NOSE
        assert feature_analysis.bias_score == 0.25
        assert feature_analysis.male_probability == 0.65
        assert feature_analysis.female_probability == 0.35


class TestDisparityScores:
    """Tests for the DisparityScores class."""
    
    def test_disparity_scores_creation(self):
        """Test creating a DisparityScores object."""
        # Create with default values
        default_scores = DisparityScores()
        assert default_scores.biasx == 0.0
        assert default_scores.equalized_odds == 0.0
        
        # Create with custom values
        custom_scores = DisparityScores(biasx=0.22, equalized_odds=0.15)
        assert custom_scores.biasx == 0.22
        assert custom_scores.equalized_odds == 0.15


class TestAnalysisResult:
    """Tests for the AnalysisResult class."""

    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult object."""
        # Create explanations
        img_data = ImageData(
            image_id="test_image_001",
            pil_image=Image.new('RGB', (100, 100)),
            gender=Gender.FEMALE
        )
        
        activation_map = np.zeros((10, 10))
        explanation = Explanation(
            image_data=img_data,
            predicted_gender=Gender.MALE,
            prediction_confidence=0.85,
            activation_map=activation_map,
            activation_boxes=[],
            landmark_boxes=[]
        )
        
        # Create feature analyses
        feature_analyses = {
            FacialFeature.NOSE: FeatureAnalysis(
                feature=FacialFeature.NOSE,
                bias_score=0.15,
                male_probability=0.60,
                female_probability=0.40
            ),
            FacialFeature.LEFT_EYE: FeatureAnalysis(
                feature=FacialFeature.LEFT_EYE,
                bias_score=0.25,
                male_probability=0.70,
                female_probability=0.30
            )
        }
        
        # Create disparity scores
        disparity_scores = DisparityScores(biasx=0.22, equalized_odds=0.12)
        
        # Create the analysis result
        result = AnalysisResult(
            explanations=[explanation],
            feature_analyses=feature_analyses,
            disparity_scores=disparity_scores
        )
        
        # Check the attributes
        assert len(result.explanations) == 1
        assert result.explanations[0] == explanation
        assert len(result.feature_analyses) == 2
        assert result.feature_analyses[FacialFeature.NOSE].bias_score == 0.15
        assert result.feature_analyses[FacialFeature.LEFT_EYE].bias_score == 0.25
        assert result.disparity_scores.biasx == 0.22
        assert result.disparity_scores.equalized_odds == 0.12
    
    def test_analysis_result_default_values(self):
        """Test creating an AnalysisResult with default values."""
        # Create with default values
        result = AnalysisResult()
        
        # Check default values
        assert len(result.explanations) == 0
        assert len(result.feature_analyses) == 0
        assert result.disparity_scores.biasx == 0.0
        assert result.disparity_scores.equalized_odds == 0.0