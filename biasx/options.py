from dataclasses import dataclass
from typing import NotRequired, Optional, TypedDict

from typing_extensions import Required

from .types import CAMMethod, ColorMode, DistanceMetric, ImageSize, ThresholdMethod


class ConfigDict(TypedDict, total=False):
    """Dictionary representation of the configuration."""

    model_path: Required[str]
    target_size: NotRequired[ImageSize]
    color_mode: NotRequired[ColorMode]
    single_channel: NotRequired[bool]

    max_faces: NotRequired[int]
    cam_method: NotRequired[CAMMethod]
    cutoff_percentile: NotRequired[int]
    threshold_method: NotRequired[ThresholdMethod]
    overlap_threshold: NotRequired[float]
    distance_metric: NotRequired[DistanceMetric]

    ndigits: NotRequired[int]


@dataclass(frozen=True)
class Config:
    """Configuration for the bias analysis pipeline."""

    model_path: str
    target_size: Optional[ImageSize] = (128, 128)
    color_mode: Optional[ColorMode] = "L"
    single_channel: Optional[bool] = False

    max_faces: Optional[int] = 1
    cam_method: Optional[CAMMethod] = "gradcam++"
    cutoff_percentile: Optional[int] = 90
    threshold_method: Optional[ThresholdMethod] = "otsu"
    overlap_threshold: Optional[float] = 0.2
    distance_metric: Optional[DistanceMetric] = "euclidean"

    ndigits: int = 3

    @classmethod
    def from_path(cls, path: str) -> "Config":
        return cls(**{k: v for k, v in ConfigDict(model_path=path).items() if v is not None})

    @classmethod
    def from_dict(cls, config: dict) -> "Config":
        return cls(**{k: v for k, v in config.items() if v is not None})

    def to_dict(self) -> ConfigDict:
        return ConfigDict(**{k: v for k, v in self.__dict__.items() if v is not None})
