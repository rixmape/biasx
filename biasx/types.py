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
    """Gender classification labels used in datasets and model outputs.

    Attributes:
        MALE: Represents the male gender, typically assigned the integer value 0.
        FEMALE: Represents the female gender, typically assigned the integer value 1.
    """

    MALE = 0
    FEMALE = 1


class Age(enum.IntEnum):
    """Age range classification labels, often used in datasets like UTKFace.

    Attributes:
        RANGE_0_9: Age range 0-9 years.
        RANGE_10_19: Age range 10-19 years.
        RANGE_20_29: Age range 20-29 years.
        RANGE_30_39: Age range 30-39 years.
        RANGE_40_49: Age range 40-49 years.
        RANGE_50_59: Age range 50-59 years.
        RANGE_60_69: Age range 60-69 years.
        RANGE_70_PLUS: Age range 70 years and above.
    """

    RANGE_0_9 = 0
    RANGE_10_19 = 1
    RANGE_20_29 = 2
    RANGE_30_39 = 3
    RANGE_40_49 = 4
    RANGE_50_59 = 5
    RANGE_60_69 = 6
    RANGE_70_PLUS = 7


class Race(enum.IntEnum):
    """Race classification labels used in datasets.

    Attributes:
        WHITE: Represents the White race category.
        BLACK: Represents the Black race category.
        ASIAN: Represents the Asian race category.
        INDIAN: Represents the Indian race category.
        OTHER: Represents other race categories not listed.
    """

    WHITE = 0
    BLACK = 1
    ASIAN = 2
    INDIAN = 3
    OTHER = 4


class FacialFeature(enum.Enum):
    """Enumeration of facial features identifiable via landmark detection.

    Used to label landmark groups and potentially activation map regions.

    Attributes:
        LEFT_EYE: The region corresponding to the left eye.
        RIGHT_EYE: The region corresponding to the right eye.
        NOSE: The region corresponding to the nose.
        LIPS: The region corresponding to the lips.
        LEFT_CHEEK: The region corresponding to the left cheek.
        RIGHT_CHEEK: The region corresponding to the right cheek.
        CHIN: The region corresponding to the chin.
        FOREHEAD: The region corresponding to the forehead.
        LEFT_EYEBROW: The region corresponding to the left eyebrow.
        RIGHT_EYEBROW: The region corresponding to the right eyebrow.
    """

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
    """Identifiers for supported dataset sources.

    Used in configuration to specify which dataset to load.

    Attributes:
        UTKFACE: Represents the UTKFace dataset.
        FAIRFACE: Represents the FairFace dataset.
    """

    UTKFACE = "utkface"
    FAIRFACE = "fairface"


class LandmarkerSource(enum.Enum):
    """Identifiers for supported facial landmark detection models/providers.

    Used in configuration to specify the landmarker implementation.

    Attributes:
        MEDIAPIPE: Represents the MediaPipe face landmarker model.
    """

    MEDIAPIPE = "mediapipe"


class ColorMode(enum.Enum):
    """Image color modes compatible with PIL (Pillow).

    Used in dataset configuration to specify target image format.

    Attributes:
        GRAYSCALE: Represents grayscale ('L' mode in PIL).
        RGB: Represents standard Red-Green-Blue color ('RGB' mode in PIL).
    """

    GRAYSCALE = "L"
    RGB = "RGB"


class CAMMethod(enum.Enum):
    """Supported Class Activation Mapping (CAM) methods.

    Used in configuration to select the algorithm for generating visual
    explanations (heatmaps).

    Attributes:
        GRADCAM: Represents the Grad-CAM algorithm.
        GRADCAM_PLUS_PLUS: Represents the Grad-CAM++ algorithm.
        SCORECAM: Represents the Score-CAM algorithm.
    """

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
    """Supported thresholding methods for processing activation maps.

    Used in configuration to select the algorithm for binarizing heatmaps
    after initial percentile filtering. Relies on `skimage.filters`.

    Attributes:
        OTSU: Represents Otsu's thresholding method.
        SAUVOLA: Represents Sauvola's thresholding method (local).
        TRIANGLE: Represents the Triangle thresholding method.
    """

    OTSU = "otsu"
    SAUVOLA = "sauvola"
    TRIANGLE = "triangle"


    def get_implementation(self) -> Callable[[np.ndarray], Any]:
        """Get the corresponding implementation function from `skimage.filters`.

        Returns the specific thresholding function (e.g., `threshold_otsu`)
        associated with the enum member.

        Returns:
            The `skimage.filters` function implementing the selected method.
        """
        implementations = {
            "otsu": threshold_otsu,
            "sauvola": threshold_sauvola,
            "triangle": threshold_triangle,
        }
        return implementations[self.value]


class DistanceMetric(enum.Enum):
    """Supported distance metrics for comparing spatial coordinates.

    Used in configuration to specify how the distance between activation box
    centers and landmark box centers is calculated. Values correspond to
    valid metrics for `scipy.spatial.distance.cdist`.

    Attributes:
        CITYBLOCK: Represents the Manhattan distance (L1 norm).
        COSINE: Represents the Cosine distance.
        EUCLIDEAN: Represents the standard Euclidean distance (L2 norm).
    """

    CITYBLOCK = "cityblock"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


@dataclass
class Box:
    """Represents a rectangular bounding box with an optional feature label.

    Used for both facial landmark features and activation map regions.

    Attributes:
        min_x (int): The minimum x-coordinate (left edge) of the box.
        min_y (int): The minimum y-coordinate (top edge) of the box.
        max_x (int): The maximum x-coordinate (right edge) of the box.
        max_y (int): The maximum y-coordinate (bottom edge) of the box.
        feature (Optional[FacialFeature]): The facial feature associated with
            this box, if identified (e.g., FacialFeature.NOSE). Defaults to None.
    """

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
    """Metadata for a resource, typically downloaded from HuggingFace Hub.

    Used to store information about datasets and models defined in configuration files.

    Attributes:
        repo_id (str): The repository ID on HuggingFace Hub (e.g., 'janko/utkface-dataset').
        filename (str): The specific filename within the repository (e.g., 'utkface_aligned_cropped.parquet').
        repo_type (str): The type of repository on HuggingFace Hub (e.g., 'dataset', 'model').
                         Defaults to 'dataset'.
        image_id_col (str): The name of the column containing image identifiers in a dataset.
                            Defaults to "". Relevant only for datasets.
        image_col (str): The name of the column containing image data (e.g., bytes) in a dataset.
                         Defaults to "". Relevant only for datasets.
        gender_col (str): The name of the column containing gender labels in a dataset.
                          Defaults to "". Relevant only for datasets.
        age_col (str): The name of the column containing age labels in a dataset.
                       Defaults to "". Relevant only for datasets.
        race_col (str): The name of the column containing race labels in a dataset.
                        Defaults to "". Relevant only for datasets.
    """

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
    """Container for data associated with a single image sample.

    Includes identifiers, raw and preprocessed image representations,
    dimensions, and demographic labels.

    Attributes:
        image_id (str): A unique identifier for the image.
        pil_image (Optional[Image.Image]): The raw image loaded as a PIL object.
                                           Defaults to None.
        preprocessed_image (Optional[np.ndarray]): The image after numerical
            preprocessing (e.g., resizing, normalization, type conversion),
            ready for model input. Defaults to None.
        width (Optional[int]): The width of the `preprocessed_image`. Defaults to None.
        height (Optional[int]): The height of the `preprocessed_image`. Defaults to None.
        gender (Optional[Gender]): The ground truth gender label for the image.
                                   Defaults to None.
        age (Optional[Age]): The ground truth age label for the image. Defaults to None.
        race (Optional[Race]): The ground truth race label for the image. Defaults to None.
    """

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
    """Container for the analysis results and explanations for a single image.

    Combines the input image data with model predictions and visual explanation
    outputs (activation maps, boxes).

    Attributes:
        image_data (ImageData): The original data associated with the image.
        predicted_gender (Gender): The gender predicted by the model.
        prediction_confidence (float): The model's confidence score for the prediction.
        activation_map (np.ndarray): The raw 2D heatmap generated by the CAM method.
        activation_boxes (List[Box]): Bounding boxes derived from the activation map.
                                      Boxes may have their `feature` attribute set if linked
                                      to a landmark.
        landmark_boxes (List[Box]): Bounding boxes detected for facial landmarks,
                                    with their `feature` attribute set.
    """

    image_data: ImageData
    predicted_gender: Gender
    prediction_confidence: float
    activation_map: np.ndarray
    activation_boxes: List[Box]
    landmark_boxes: List[Box]


@dataclass
class FeatureAnalysis:
    """Container for bias analysis results specific to a single facial feature.

    Stores the calculated probability of a feature being activated during
    misclassifications for each gender group, and the resulting bias score.

    Attributes:
        feature (FacialFeature): The facial feature being analyzed.
        bias_score (float): The absolute difference between `male_probability` and
                            `female_probability`. A measure of bias associated
                            with this feature.
        male_probability (float): The probability that this feature was activated
                                  in images where the true gender was male, but the
                                  model predicted incorrectly.
        female_probability (float): The probability that this feature was activated
                                    in images where the true gender was female, but the
                                    model predicted incorrectly.
    """

    feature: FacialFeature
    bias_score: float
    male_probability: float
    female_probability: float


@dataclass
class DisparityScores:
    """Container for overall bias and fairness disparity metrics for the model.

    Aggregates results across features and performance metrics.

    Attributes:
        biasx (float): An overall bias score, calculated as the average of the
                       absolute `bias_score` values across all analyzed features
                       in `FeatureAnalysis`. Defaults to 0.0.
        equalized_odds (float): A fairness metric representing the maximum disparity
                                between male and female groups in either the True
                                Positive Rate (TPR) or the False Positive Rate (FPR).
                                A score of 0 indicates perfect equality in these error rates.
                                Defaults to 0.0.
    """

    biasx: float = 0.0
    equalized_odds: float = 0.0


@dataclass
class AnalysisResult:
    """Container for the complete results of a bias analysis run.

    Holds all generated explanations, feature-specific analyses, and overall
    disparity scores.

    Attributes:
        explanations (List[Explanation]): A list containing an `Explanation` object
            for every image analyzed in the run. Defaults to an empty list.
        feature_analyses (Dict[FacialFeature, FeatureAnalysis]): A dictionary mapping
            each analyzed `FacialFeature` to its corresponding `FeatureAnalysis` object.
            Defaults to an empty dictionary.
        disparity_scores (DisparityScores): An object containing the calculated overall
            bias metrics (`biasx`, `equalized_odds`). Defaults to a `DisparityScores`
            object with default (0.0) values.
    """

    explanations: List[Explanation] = field(default_factory=list)
    feature_analyses: Dict[FacialFeature, FeatureAnalysis] = field(default_factory=dict)
    disparity_scores: DisparityScores = field(default_factory=DisparityScores)
