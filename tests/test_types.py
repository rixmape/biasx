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
    def test_gender_enum(self):
        assert Gender.MALE == 0
        assert Gender.FEMALE == 1
        
    def test_age_enum(self):
        assert Age.RANGE_0_9 == 0
        assert Age.RANGE_70_PLUS == 7
        
    def test_race_enum(self):
        assert Race.WHITE == 0
        assert Race.OTHER == 4
        
    def test_facial_feature_enum(self):
        assert FacialFeature.LEFT_EYE.value == "left_eye"
        assert FacialFeature.RIGHT_EYEBROW.value == "right_eyebrow"
        
    def test_dataset_source_enum(self):
        assert DatasetSource.UTKFACE.value == "utkface"
        assert DatasetSource.FAIRFACE.value == "fairface"
        
    def test_landmarker_source_enum(self):
        assert LandmarkerSource.MEDIAPIPE.value == "mediapipe"
        
    def test_color_mode_enum(self):
        assert ColorMode.GRAYSCALE.value == "L"
        assert ColorMode.RGB.value == "RGB"
        
    def test_distance_metric_enum(self):
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.CITYBLOCK.value == "cityblock"
        assert DistanceMetric.COSINE.value == "cosine"

# Tests for CAMMethod and implementations
class TestCAMMethod:
    def test_gradcam_implementation(self):
        impl = CAMMethod.GRADCAM.get_implementation()
        assert callable(impl)
        
    def test_gradcam_plus_plus_implementation(self):
        impl = CAMMethod.GRADCAM_PLUS_PLUS.get_implementation()
        assert callable(impl)
        
    def test_scorecam_implementation(self):
        impl = CAMMethod.SCORECAM.get_implementation()
        assert callable(impl)
        
    def test_all_implementations(self):
        # Make sure all enum values have an implementation
        for method in CAMMethod:
            impl = method.get_implementation()
            assert callable(impl)

# Tests for ThresholdMethod and implementations
class TestThresholdMethod:
    def test_otsu_implementation(self):
        impl = ThresholdMethod.OTSU.get_implementation()
        assert impl == threshold_otsu
        
    def test_sauvola_implementation(self):
        impl = ThresholdMethod.SAUVOLA.get_implementation()
        assert impl == threshold_sauvola
        
    def test_triangle_implementation(self):
        impl = ThresholdMethod.TRIANGLE.get_implementation()
        assert impl == threshold_triangle
        
    def test_all_implementations(self):
        # Make sure all enum values have an implementation
        for method in ThresholdMethod:
            impl = method.get_implementation()
            assert callable(impl)

# Tests for Box class
class TestBoundingBox:
    def test_bounding_box_creation(self):
        box = Box(10, 20, 30, 40)
        assert box.min_x == 10
        assert box.min_y == 20
        assert box.max_x == 30
        assert box.max_y == 40
        assert box.feature is None
        
    def test_bounding_box_properties(self):
        box = Box(10, 20, 30, 40)
        assert box.center == (20, 30)
        assert box.area == 400  # (30-10)*(40-20) = 20*20 = 400
        
    def test_bounding_box_with_feature(self):
        box = Box(10, 20, 30, 40, feature=FacialFeature.NOSE)
        assert box.feature == FacialFeature.NOSE
        
    def test_bounding_box_with_zero_dimensions(self):
        box = Box(10, 20, 10, 20)  # Zero width and height
        assert box.area == 0
        assert box.center == (10, 20)

# Tests for ResourceMetadata class
class TestResourceMetadata:
    def test_resource_metadata_minimal_creation(self):
        metadata = ResourceMetadata(repo_id="test/repo", filename="data.file")
        assert metadata.repo_id == "test/repo"
        assert metadata.filename == "data.file"
        assert metadata.repo_type == "dataset"  # Default value
        
    def test_resource_metadata_complete_creation(self):
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
    def test_image_data_creation(self):
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
    def test_explanation_creation(self):
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
    def test_feature_analysis_creation(self):
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
    def test_disparity_scores_creation(self):
        scores = DisparityScores(biasx=0.35, equalized_odds=0.42)
        assert scores.biasx == 0.35
        assert scores.equalized_odds == 0.42
        
    def test_disparity_scores_default_values(self):
        scores = DisparityScores()
        assert scores.biasx == 0.0
        assert scores.equalized_odds == 0.0

# Tests for AnalysisResult class
class TestAnalysisResult:
    def test_analysis_result_creation(self):
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
        
        result = AnalysisResult(
            explanations=[explanation],
            feature_analyses={FacialFeature.NOSE: feature_analysis},
            disparity_scores=disparity_scores
        )
        
        assert result.explanations == [explanation]
        assert result.feature_analyses == {FacialFeature.NOSE: feature_analysis}
        assert result.disparity_scores == disparity_scores
        
    def test_analysis_result_default_values(self):
        result = AnalysisResult()
        assert result.explanations == []
        assert result.feature_analyses == {}
        assert isinstance(result.disparity_scores, DisparityScores)
        assert result.disparity_scores.biasx == 0.0
        assert result.disparity_scores.equalized_odds == 0.0