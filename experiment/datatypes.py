from enum import Enum, auto
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_validator


class OutputLevel(Enum):
    NONE = auto()
    RESULTS_ONLY = auto()
    FULL = auto()


class DatasetSource(Enum):
    UTKFACE = "utkface"
    FAIRFACE = "fairface"


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Gender(Enum):
    MALE = 0
    FEMALE = 1


class Feature(Enum):
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
    min_x: int = Field(..., ge=0)
    min_y: int = Field(..., ge=0)
    max_x: int = Field(..., ge=0)
    max_y: int = Field(..., ge=0)

    @computed_field
    @property
    def area(self) -> int:
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        return max(0, width) * max(0, height)

    @field_validator("max_x")
    @classmethod
    def check_x_coords(cls, v: int, info: ValidationInfo) -> int:
        if "min_x" in info.data and v < info.data["min_x"]:
            raise ValueError("max_x must be greater than or equal to min_x")
        return v

    @field_validator("max_y")
    @classmethod
    def check_y_coords(cls, v: int, info: ValidationInfo) -> int:
        if "min_y" in info.data and v < info.data["min_y"]:
            raise ValueError("max_y must be greater than or equal to min_y")
        return v


class FeatureDetails(BaseModel):
    feature: Feature
    bbox: BoundingBox = Field(default_factory=BoundingBox)
    attention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_key_feature: bool = Field(default=False)


class GenderPerformanceMetrics(BaseModel):
    positive_class: Gender
    tp: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    tn: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)

    @computed_field
    @property
    def tpr(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @computed_field
    @property
    def fpr(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @computed_field
    @property
    def tnr(self) -> float:
        return self.tn / max(self.tn + self.fp, 1)

    @computed_field
    @property
    def fnr(self) -> float:
        return self.fn / max(self.fn + self.tp, 1)

    @computed_field
    @property
    def ppv(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @computed_field
    @property
    def npv(self) -> float:
        return self.tn / max(self.tn + self.fn, 1)

    @computed_field
    @property
    def fdr(self) -> float:
        return self.fp / max(self.fp + self.tp, 1)

    @computed_field
    @property
    def _for(self) -> float:
        return self.fn / max(self.fn + self.tn, 1)


class BiasMetrics(BaseModel):
    demographic_parity: float = Field(..., ge=0.0)
    equalized_odds: float = Field(..., ge=0.0)
    conditional_use_accuracy_equality: float = Field(..., ge=0.0)
    mean_feature_distribution_bias: float = Field(..., ge=0.0)


class Explanation(BaseModel):
    image_id: str = Field(..., min_length=1)
    label: Gender
    prediction: Gender
    confidence_scores: List[float] = Field(default_factory=list)
    heatmap_path: Optional[str] = Field(default=None)
    detected_features: List[FeatureDetails] = Field(default_factory=list)

    @field_validator("confidence_scores")
    @classmethod
    def check_confidence_scores(cls, v: List[float]) -> List[float]:
        if len(v) != len(Gender):
            raise ValueError(f"confidence_scores must have length {len(Gender)}")
        if not all(0.0 <= score <= 1.0 for score in v):
            raise ValueError("All confidence scores must be between 0.0 and 1.0")
        return v


class FeatureDistribution(BaseModel):
    feature: Feature
    male_distribution: float = Field(..., ge=0.0, le=1.0)
    female_distribution: float = Field(..., ge=0.0, le=1.0)

    @computed_field
    @property
    def distribution_bias(self) -> float:
        return abs(self.male_distribution - self.female_distribution)


class AnalysisResult(BaseModel):
    feature_distributions: List[FeatureDistribution] = Field(default_factory=list)
    male_performance_metrics: Optional[GenderPerformanceMetrics] = Field(default=None)
    female_performance_metrics: Optional[GenderPerformanceMetrics] = Field(default=None)
    bias_metrics: Optional[BiasMetrics] = Field(default=None)
    analyzed_images: List[Explanation] = Field(default_factory=list)


class ModelHistory(BaseModel):
    train_loss: List[float] = Field(default_factory=list)
    train_accuracy: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    val_accuracy: List[float] = Field(default_factory=list)

    @field_validator("train_accuracy", "val_accuracy")
    @classmethod
    def check_accuracy_values(cls, v: List[float]) -> List[float]:
        if not all(0.0 <= acc <= 1.0 for acc in v):
            raise ValueError("Accuracy values must be between 0.0 and 1.0")
        return v

    @field_validator("train_loss", "val_loss")
    @classmethod
    def check_loss_values(cls, v: List[float]) -> List[float]:
        if not all(loss >= 0.0 for loss in v):
            raise ValueError("Loss values must be non-negative")
        return v


class ExperimentResult(BaseModel):
    id: str = Field(..., min_length=1)
    config: dict = Field(default_factory=dict)
    history: Optional[ModelHistory] = Field(default=None)
    analysis: Optional[AnalysisResult] = Field(default=None)
