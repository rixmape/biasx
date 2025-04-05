"""Tests for the type definitions and data structures in BiasX."""

import pytest
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_triangle

from biasx.types import (
    Age, AnalysisResult, Box, CAMMethod, ColorMode, DatasetSource, DisparityScores,
    DistanceMetric, Explanation, FacialFeature, FeatureAnalysis, Gender, ImageData,
    LandmarkerSource, Race, ResourceMetadata, ThresholdMethod
)


# Tests for enums
class TestEnums:
    """Tests for enumeration types used throughout the library."""
    
    def test_gender_enum(self):
        """Test Gender enum values."""
        assert Gender.MALE == 0
        assert Gender.FEMALE == 1
        
    def test_age_enum(self):
        """Test Age enum ranges."""
        assert Age.RANGE_0_9 == 0
        assert Age.RANGE_70_PLUS == 7
        
    def test_race_enum(self):
        """Test Race enum values."""
        assert Race.WHITE == 0
        assert Race.OTHER == 4
        
    def test_facial_feature_enum(self):
        """Test FacialFeature enum string values."""
        assert FacialFeature.LEFT_EYE.value == "left_eye"
        assert FacialFeature.RIGHT_EYEBROW.value == "right_eyebrow"
        
    def test_dataset_source_enum(self):
        """Test DatasetSource enum string values."""
        assert DatasetSource.UTKFACE.value == "utkface"
        assert DatasetSource.FAIRFACE.value == "fairface"
        
    def test_landmarker_source_enum(self):
        """Test LandmarkerSource enum string values."""
        assert LandmarkerSource.MEDIAPIPE.value == "mediapipe"
        
    def test_color_mode_enum(self):
        """Test ColorMode enum string values."""
        assert ColorMode.GRAYSCALE.value == "L"
        assert ColorMode.RGB.value == "RGB"
        
    def test_distance_metric_enum(self):
        """Test DistanceMetric enum string values."""
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.CITYBLOCK.value == "cityblock"
        assert DistanceMetric.COSINE.value == "cosine"


# Tests for CAMMethod and implementations
class TestCAMMethod:
    """Tests for Class Activation Mapping method enums and implementations."""
    
    def test_gradcam_implementation(self):
        """Test GRADCAM implementation retrieval."""
        impl = CAMMethod.GRADCAM.get_implementation()
        assert callable(impl)
        
    def test_gradcam_plus_plus_implementation(self):
        """Test GRADCAM++ implementation retrieval."""
        impl = CAMMethod.GRADCAM_PLUS_PLUS.get_implementation()
        assert callable(impl)
        
    def test_scorecam_implementation(self):
        """Test SCORECAM implementation retrieval."""
        impl = CAMMethod.SCORECAM.get_implementation()
        assert callable(impl)
        
    def test_all_implementations(self):
        """Test that all CAM methods have valid implementations."""
        # Make sure all enum values have an implementation
        for method in CAMMethod:
            impl = method.get_implementation()
            assert callable(impl)


# Tests for ThresholdMethod and implementations
class TestThresholdMethod:
    """Tests for thresholding method enums and implementations."""
    
    def test_otsu_implementation(self):
        """Test Otsu threshold implementation retrieval."""
        impl = ThresholdMethod.OTSU.get_implementation()
        assert impl == threshold_otsu
        
    def test_sauvola_implementation(self):
        """Test Sauvola threshold implementation retrieval."""
        impl = ThresholdMethod.SAUVOLA.get_implementation()
        assert impl == threshold_sauvola
        
    def test_triangle_implementation(self):
        """Test Triangle threshold implementation retrieval."""
        impl = ThresholdMethod.TRIANGLE.get_implementation()
        assert impl == threshold_triangle
        
    def test_all_implementations(self):
        """Test that all threshold methods have valid implementations."""
        # Make sure all enum values have an implementation
        for method in ThresholdMethod:
            impl = method.get_implementation()
            assert callable(impl)


# Tests for Box class
class TestBoundingBox:
    """Tests for the Box class that represents activation and feature regions."""
    
    def test_bounding_box_creation(self):
        """Test basic Box creation."""
        box = Box(10, 20, 30, 40)
        assert box.min_x == 10
        assert box.min_y == 20
        assert box.max_x == 30
        assert box.max_y == 40
        assert box.feature is None
        
    def test_bounding_box_properties(self):
        """Test Box property calculations (center, area)."""
        box = Box(10, 20, 30, 40)
        assert box.center == (20, 30)
        assert box.area == 400  # (30-10)*(40-20) = 20*20 = 400
        
    def test_bounding_box_with_feature(self):
        """Test Box with assigned facial feature."""
        box = Box(10, 20, 30, 40, feature=FacialFeature.NOSE)
        assert box.feature == FacialFeature.NOSE
        
    def test_bounding_box_with_zero_dimensions(self):
        """Test Box with zero width and height."""
        box = Box(10, 20, 10, 20)  # Zero width and height
        assert box.area == 0
        assert box.center == (10, 20)


# Tests for ResourceMetadata class
class TestResourceMetadata:
    """Tests for the ResourceMetadata class that stores external resource information."""
    
    def test_resource_metadata_minimal_creation(self):
        """Test minimal ResourceMetadata creation."""
        metadata = ResourceMetadata(repo_id="test/repo", filename="data.file")
        assert metadata.repo_id == "test/repo"
        assert metadata.filename == "data.file"
        assert metadata.repo_type == "dataset"  # Default value
        
    def test_resource_metadata_complete_creation(self):
        """Test full ResourceMetadata creation with all fields."""
        metadata = ResourceMetadata(
            repo_id="test/repo",
            filename="data.file",
            repo_type="model",
            image_id_col="id",
            image_col="img",
            gender_col="gender",
            age_col="age",
            race_col="race"
        )
        assert metadata.repo_id == "test/repo"
        assert metadata.filename == "data.file"
        assert metadata.repo_type == "model"
        assert metadata.image_id_col == "id"
        assert metadata.image_col == "img"
        assert metadata.gender_col == "gender"
        assert metadata.age_col == "age"
        assert metadata.race_col == "race"


# Tests for ImageData class
class TestImageData:
    """Tests for the ImageData class that stores image data and metadata."""
    
    def test_image_data_creation(self):
        """Test complete ImageData creation."""
        img = Image.new('RGB', (10, 10))
        img_array = np.zeros((10, 10, 3))
        
        image_data = ImageData(
            image_id="test_image",
            pil_image=img,
            preprocessed_image=img_array,
            width=10,
            height=10,
            gender=Gender.MALE,
            age=Age.RANGE_20_29,
            race=Race.ASIAN
        )
        
        assert image_data.image_id == "test_image"
        assert image_data.pil_image == img
        assert np.array_equal(image_data.preprocessed_image, img_array)
        assert image_data.width == 10
        assert image_data.height == 10
        assert image_data.gender == Gender.MALE
        assert image_data.age == Age.RANGE_20_29
        assert image_data.race == Race.ASIAN
        
    def test_image_data_minimal_creation(self):
        """Test minimal ImageData creation with just an ID."""
        image_data = ImageData(image_id="test_image")
        assert image_data.image_id == "test_image"
        assert image_data.pil_image is None
        assert image_data.preprocessed_image is None
        assert image_data.width is None
        assert image_data.height is None
        assert image_data.gender is None
        assert image_data.age is None
        assert image_data.race is None


# Tests for Explanation class
class TestExplanation:
    """Tests for the Explanation class that contains model explanation data."""
    
    def test_explanation_creation(self):
        """Test Explanation creation with all fields."""
        image_data = ImageData(image_id="test_image")
        activation_map = np.zeros((10, 10))
        activation_boxes = [Box(10, 20, 30, 40, FacialFeature.NOSE)]
        landmark_boxes = [Box(15, 25, 35, 45, FacialFeature.LIPS)]
        
        explanation = Explanation(
            image_data=image_data,
            predicted_gender=Gender.FEMALE,
            prediction_confidence=0.95,
            activation_map=activation_map,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes
        )
        
        assert explanation.image_data == image_data
        assert explanation.predicted_gender == Gender.FEMALE
        assert explanation.prediction_confidence == 0.95
        assert np.array_equal(explanation.activation_map, activation_map)
        assert explanation.activation_boxes == activation_boxes
        assert explanation.landmark_boxes == landmark_boxes


# Tests for FeatureAnalysis class
class TestFeatureAnalysis:
    """Tests for the FeatureAnalysis class that contains feature-specific analysis."""
    
    def test_feature_analysis_creation(self):
        """Test FeatureAnalysis creation with all fields."""
        analysis = FeatureAnalysis(
            feature=FacialFeature.NOSE,
            bias_score=0.45,
            male_probability=0.3,
            female_probability=0.75
        )
        
        assert analysis.feature == FacialFeature.NOSE
        assert analysis.bias_score == 0.45
        assert analysis.male_probability == 0.3
        assert analysis.female_probability == 0.75


# Tests for DisparityScores class
class TestDisparityScores:
    """Tests for the DisparityScores class that contains bias metrics."""
    
    def test_disparity_scores_creation(self):
        """Test DisparityScores creation with explicit values."""
        scores = DisparityScores(biasx=0.35, equalized_odds=0.42)
        assert scores.biasx == 0.35
        assert scores.equalized_odds == 0.42
        
    def test_disparity_scores_default_values(self):
        """Test DisparityScores creation with default values."""
        scores = DisparityScores()
        assert scores.biasx == 0.0
        assert scores.equalized_odds == 0.0


# Tests for AnalysisResult class
class TestAnalysisResult:
    """Tests for the AnalysisResult class that contains complete analysis results."""
    
    def test_analysis_result_creation(self):
        """Test AnalysisResult creation with all components."""
        # Create dependencies for the test
        image_data = ImageData(image_id="test_image")
        explanation = Explanation(
            image_data=image_data,
            predicted_gender=Gender.FEMALE,
            prediction_confidence=0.95,
            activation_map=np.zeros((10, 10)),
            activation_boxes=[],
            landmark_boxes=[]
        )
        
        feature_analysis = FeatureAnalysis(
            feature=FacialFeature.NOSE,
            bias_score=0.45,
            male_probability=0.3,
            female_probability=0.75
        )
        
        disparity_scores = DisparityScores(biasx=0.35, equalized_odds=0.42)
        
        # Create the analysis result
        result = AnalysisResult(
            explanations=[explanation],
            feature_analyses={FacialFeature.NOSE: feature_analysis},
            disparity_scores=disparity_scores
        )
        
        # Verify all fields
        assert result.explanations == [explanation]
        assert result.feature_analyses == {FacialFeature.NOSE: feature_analysis}
        assert result.disparity_scores == disparity_scores
        
    def test_analysis_result_default_values(self):
        """Test AnalysisResult creation with default values."""
        result = AnalysisResult()
        assert result.explanations == []
        assert result.feature_analyses == {}
        assert isinstance(result.disparity_scores, DisparityScores)
        assert result.disparity_scores.biasx == 0.0
        assert result.disparity_scores.equalized_odds == 0.0