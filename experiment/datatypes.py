from enum import Enum, auto
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_validator


class OutputLevel(Enum):
    """Defines the level of detail for experiment artifacts and outputs."""

    NONE = auto()
    RESULTS_ONLY = auto()
    FULL = auto()


class DatasetSource(Enum):
    """Enumerates the possible source datasets for the experiment."""

    UTKFACE = "utkface"
    FAIRFACE = "fairface"


class DatasetSplit(Enum):
    """Enumerates the standard dataset splits used in machine learning."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Gender(Enum):
    """Enumerates the gender categories used in the dataset and model."""

    MALE = 0
    FEMALE = 1


class ProtectedAttribute(Enum):
    """Enumerates the protected attributes considered for bias analysis."""

    GENDER = "gender"
    RACE = "race"
    AGE = "age"


class Feature(Enum):
    """Enumerates the facial features detected by the landmarker."""

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


class BoundingBox(BaseModel):
    """Represents a rectangular bounding box defined by min/max coordinates."""

    min_x: int = Field(..., ge=0)
    min_y: int = Field(..., ge=0)
    max_x: int = Field(..., ge=0)
    max_y: int = Field(..., ge=0)

    @computed_field
    @property
    def area(self) -> int:
        """Calculates the area of the bounding box."""
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        return max(0, width) * max(0, height)

    @field_validator("max_x")
    @classmethod
    def check_x_coords(cls, v: int, info: ValidationInfo) -> int:
        """Validates that max_x is not less than min_x."""
        if "min_x" in info.data and v < info.data["min_x"]:
            raise ValueError("max_x must be greater than or equal to min_x")
        return v

    @field_validator("max_y")
    @classmethod
    def check_y_coords(cls, v: int, info: ValidationInfo) -> int:
        """Validates that max_y is not less than min_y."""
        if "min_y" in info.data and v < info.data["min_y"]:
            raise ValueError("max_y must be greater than or equal to min_y")
        return v


class FeatureDetails(BaseModel):
    """Stores details about a detected facial feature, including its location and attention."""

    feature: Feature
    bbox: BoundingBox = Field(default_factory=BoundingBox)
    attention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_key_feature: bool = Field(default=False)


class GenderPerformanceMetrics(BaseModel):
    """Holds detailed performance metrics (TP, FP, TN, FN) for a specific gender class."""

    positive_class: Gender
    tp: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    tn: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)

    @computed_field
    @property
    def tpr(self) -> float:
        """Calculates the True Positive Rate (Recall/Sensitivity)."""
        return self.tp / max(self.tp + self.fn, 1)

    @computed_field
    @property
    def fpr(self) -> float:
        """Calculates the False Positive Rate (Fall-out)."""
        return self.fp / max(self.fp + self.tn, 1)

    @computed_field
    @property
    def tnr(self) -> float:
        """Calculates the True Negative Rate (Specificity)."""
        return self.tn / max(self.tn + self.fp, 1)

    @computed_field
    @property
    def fnr(self) -> float:
        """Calculates the False Negative Rate (Miss Rate)."""
        return self.fn / max(self.fn + self.tp, 1)

    @computed_field
    @property
    def ppv(self) -> float:
        """Calculates the Positive Predictive Value (Precision)."""
        return self.tp / max(self.tp + self.fp, 1)

    @computed_field
    @property
    def npv(self) -> float:
        """Calculates the Negative Predictive Value."""
        return self.tn / max(self.tn + self.fn, 1)

    @computed_field
    @property
    def fdr(self) -> float:
        """Calculates the False Discovery Rate."""
        return self.fp / max(self.fp + self.tp, 1)

    @computed_field
    @property
    def _for(self) -> float:
        """Calculates the False Omission Rate."""
        return self.fn / max(self.fn + self.tn, 1)


class BiasMetrics(BaseModel):
    """Stores various computed bias metrics comparing performance across groups."""

    demographic_parity: float = Field(..., ge=0.0)
    equalized_odds: float = Field(..., ge=0.0)
    conditional_use_accuracy_equality: float = Field(..., ge=0.0)
    mean_feature_distribution_bias: float = Field(..., ge=0.0)


class Explanation(BaseModel):
    """Contains all relevant information for a single data point's prediction and explanation."""

    image_id: str = Field(..., min_length=1)
    label: Gender
    prediction: Gender
    race: str
    age: int
    confidence_scores: List[float] = Field(default_factory=list)
    heatmap_path: Optional[str] = Field(default=None)
    detected_features: List[FeatureDetails] = Field(default_factory=list)

    @field_validator("confidence_scores")
    @classmethod
    def check_confidence_scores(cls, v: List[float]) -> List[float]:
        """Validates the length and range of confidence scores."""
        if len(v) != len(Gender):
            raise ValueError(f"confidence_scores must have length {len(Gender)}")
        if not all(0.0 <= score <= 1.0 for score in v):
            raise ValueError("All confidence scores must be between 0.0 and 1.0")
        return v


class FeatureDistribution(BaseModel):
    """Represents the distribution of a specific key feature across genders."""

    feature: Feature
    male_distribution: float = Field(..., ge=0.0, le=1.0)
    female_distribution: float = Field(..., ge=0.0, le=1.0)

    @computed_field
    @property
    def distribution_bias(self) -> float:
        """Calculates the absolute difference in feature distribution between genders."""
        return abs(self.male_distribution - self.female_distribution)


class AnalysisResult(BaseModel):
    """Aggregates all results from the bias analysis phase."""

    feature_distributions: List[FeatureDistribution] = Field(default_factory=list)
    male_performance_metrics: Optional[GenderPerformanceMetrics] = Field(default=None)
    female_performance_metrics: Optional[GenderPerformanceMetrics] = Field(default=None)
    bias_metrics: Optional[BiasMetrics] = Field(default=None)
    analyzed_images: List[Explanation] = Field(default_factory=list)


class ModelHistory(BaseModel):
    """Stores the training and validation history (loss, accuracy) over epochs."""

    train_loss: List[float] = Field(default_factory=list)
    train_accuracy: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    val_accuracy: List[float] = Field(default_factory=list)

    @field_validator("train_accuracy", "val_accuracy")
    @classmethod
    def check_accuracy_values(cls, v: List[float]) -> List[float]:
        """Validates that accuracy values are within the valid range [0.0, 1.0]."""
        if not all(0.0 <= acc <= 1.0 for acc in v):
            raise ValueError("Accuracy values must be between 0.0 and 1.0")
        return v

    @field_validator("train_loss", "val_loss")
    @classmethod
    def check_loss_values(cls, v: List[float]) -> List[float]:
        """Validates that loss values are non-negative."""
        if not all(loss >= 0.0 for loss in v):
            raise ValueError("Loss values must be non-negative")
        return v


class ExperimentResult(BaseModel):
    """Encapsulates the final results of a single experiment run."""

    id: str = Field(..., min_length=1)
    config: dict = Field(default_factory=dict)
    history: Optional[ModelHistory] = Field(default=None)
    analysis: Optional[AnalysisResult] = Field(default=None)
