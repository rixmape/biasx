"""Provides enumerations and dataclasses that define the data structures used throughout the library."""

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tf_keras_vis
from PIL import Image
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_triangle
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam


class Gender(enum.IntEnum):
    """Gender classification labels."""

    MALE = 0
    FEMALE = 1


class Age(enum.IntEnum):
    """Age range classification labels."""

    RANGE_0_9 = 0
    RANGE_10_19 = 1
    RANGE_20_29 = 2
    RANGE_30_39 = 3
    RANGE_40_49 = 4
    RANGE_50_59 = 5
    RANGE_60_69 = 6
    RANGE_70_PLUS = 7


class Race(enum.IntEnum):
    """Race classification labels."""

    WHITE = 0
    BLACK = 1
    ASIAN = 2
    INDIAN = 3
    OTHER = 4


class FacialFeature(enum.Enum):
    """Facial feature types used for landmark identification."""

    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    LIPS = "lips"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    CHIN = "chin"
    FOREHEAD = "forehead"
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"


class DatasetSource(enum.Enum):
    """Available dataset sources."""

    UTKFACE = "utkface"
    FAIRFACE = "fairface"


class LandmarkerSource(enum.Enum):
    """Available facial landmark detection models."""

    MEDIAPIPE = "mediapipe"


class ColorMode(enum.Enum):
    """Image color modes."""

    GRAYSCALE = "L"
    RGB = "RGB"


class CAMMethod(enum.Enum):
    """Class activation mapping methods."""

    GRADCAM = "gradcam"
    GRADCAM_PLUS_PLUS = "gradcam++"
    SCORECAM = "scorecam"

    def get_implementation(self) -> tf_keras_vis.ModelVisualization:
        """Get the implementation class for this CAM method."""
        implementations = {
            "gradcam": Gradcam,
            "gradcam++": GradcamPlusPlus,
            "scorecam": Scorecam,
        }
        return implementations[self.value]


class ThresholdMethod(enum.Enum):
    """Thresholding methods for activation map processing."""

    OTSU = "otsu"
    SAUVOLA = "sauvola"
    TRIANGLE = "triangle"


    def get_implementation(self) -> Callable[[np.ndarray], Any]:
        """Get the implementation function for this threshold method."""
        implementations = {
            "otsu": threshold_otsu,
            "sauvola": threshold_sauvola,
            "triangle": threshold_triangle,
        }
        return implementations[self.value]


class DistanceMetric(enum.Enum):
    """Distance metrics for comparing spatial coordinates."""

    CITYBLOCK = "cityblock"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


@dataclass
class Box:
    """Represents a bounding box with an optional feature label."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    feature: Optional[FacialFeature] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Compute center coordinates of the box."""
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def area(self) -> float:
        """Compute area of the box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)


@dataclass
class ResourceMetadata:
    """Metadata for a resource from HuggingFace Hub."""

    repo_id: str
    filename: str
    repo_type: str = "dataset"

    image_id_col: str = ""
    image_col: str = ""
    gender_col: str = ""
    age_col: str = ""
    race_col: str = ""


@dataclass
class ImageData:
    """Container for image data and its attributes."""

    image_id: str
    pil_image: Optional[Image.Image] = None
    preprocessed_image: Optional[np.ndarray] = None
    width: Optional[int] = None
    height: Optional[int] = None
    gender: Optional[Gender] = None
    age: Optional[Age] = None
    race: Optional[Race] = None


@dataclass
class Explanation:
    """Analysis results and explanations for a single image."""

    image_data: ImageData
    predicted_gender: Gender
    prediction_confidence: float
    activation_map: np.ndarray
    activation_boxes: List[Box]
    landmark_boxes: List[Box]


@dataclass
class FeatureAnalysis:
    """Analysis results for a specific facial feature."""

    feature: FacialFeature
    bias_score: float
    male_probability: float
    female_probability: float


@dataclass
class DisparityScores:
    """Collection of metrics for measuring bias."""

    biasx: float = 0.0
    equalized_odds: float = 0.0


@dataclass
class AnalysisResult:
    """Complete results of a bias analysis run."""

    explanations: List[Explanation] = field(default_factory=list)
    feature_analyses: Dict[FacialFeature, FeatureAnalysis] = field(default_factory=dict)
    disparity_scores: DisparityScores = field(default_factory=DisparityScores)
