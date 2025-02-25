from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypedDict

Gender = Literal[0, 1]

FacialFeature = Literal[
    "left_eye",
    "right_eye",
    "nose",
    "lips",
    "left_cheek",
    "right_cheek",
    "chin",
    "forehead",
    "left_eyebrow",
    "right_eyebrow",
]

ColorMode = Literal["L", "RGB"]
CAMMethod = Literal["gradcam", "gradcam++", "scorecam"]

ThresholdMethod = Literal[
    "isodata",
    "li",
    "local",
    "mean",
    "minimum",
    "multiotsu",
    "niblack",
    "otsu",
    "sauvola",
    "triangle",
    "yen",
]

DistanceMetric = Literal[
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulczynski1",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
]

FairnessMetric = Literal[
    "overall_bias",
    "equalized_odds",
    "demographic_parity",
    "disparate_impact",
    "predictive_parity",
    "equal_opportunity",
    "accuracy_parity",
    "treatment_equality",
]


class ConfusionMatrixStats(TypedDict):
    TP: int
    FP: int
    TN: int
    FN: int


ConfusionMatrix = Dict[Gender, ConfusionMatrixStats]


class ProcessedResults(TypedDict):
    by_gender: Dict[Gender, list]
    misclassified: list
    misclassified_by_gender: Dict[Gender, list]
    total_count: int


@dataclass
class Box:
    """Represents a bounding box with an optional feature label."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    feature: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        """Compute center coordinates of the box."""
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def area(self) -> float:
        """Compute area of the box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def to_dict(self) -> dict[str, Any]:
        """Convert box to dictionary format."""
        return {
            **{"minX": self.min_x, "minY": self.min_y, "maxX": self.max_x, "maxY": self.max_y},
            **({"feature": self.feature} if self.feature else {}),
        }


@dataclass
class Explanation:
    """Encapsulates analysis results and explanations for a single image."""

    image_path: str
    true_gender: Gender
    predicted_gender: Gender
    prediction_confidence: float
    activation_map_path: Optional[str]
    activation_boxes: list[Box]
    landmark_boxes: list[Box]

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to dictionary format for serialization."""
        return {
            "imagePath": self.image_path,
            "trueGender": self.true_gender,
            "predictedGender": self.predicted_gender,
            "predictionConfidence": self.prediction_confidence,
            "activationMapPath": self.activation_map_path,
            "activationBoxes": [box.to_dict() for box in self.activation_boxes],
            "landmarkBoxes": [box.to_dict() for box in self.landmark_boxes],
        }


FeatureScore = Dict[FacialFeature, float]
FeatureProbability = Dict[str, Dict[Gender, float]]
FairnessScores = Dict[FairnessMetric, float]
