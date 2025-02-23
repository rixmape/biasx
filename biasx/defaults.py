from typing import TypedDict

from typing_extensions import NotRequired, Required

from .types import CAMMethod, ColorMode, DistanceMetric, ImageSize, ThresholdMethod


class ModelConfig(TypedDict, total=False):
    target_size: NotRequired[ImageSize]
    color_mode: NotRequired[ColorMode]
    single_channel: NotRequired[bool]


class ExplainerConfig(TypedDict, total=False):
    max_faces: NotRequired[int]
    cam_method: NotRequired[CAMMethod]
    cutoff_percentile: NotRequired[int]
    threshold_method: NotRequired[ThresholdMethod]
    overlap_threshold: NotRequired[float]
    distance_metric: NotRequired[DistanceMetric]


class AnalysisConfig(TypedDict, total=False):
    ndigits: NotRequired[int]
    shuffle: NotRequired[bool]
    seed: NotRequired[int]
    max_samples: NotRequired[int]


class BaseConfig(TypedDict, total=False):
    model_path: Required[str]
    model_options: NotRequired[ModelConfig]
    explainer_options: NotRequired[ExplainerConfig]
    calculator_options: NotRequired[AnalysisConfig]


def create_default_config(model_path: str) -> BaseConfig:
    """Create a complete configuration with all defaults"""
    return {
        "model_path": model_path,
        "model_options": {
            "target_size": (128, 128),
            "color_mode": "L",
            "single_channel": False,
        },
        "explainer_options": {
            "max_faces": 1,
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
            "overlap_threshold": 0.2,
            "distance_metric": "euclidean",
        },
        "calculator_options": {
            "ndigits": 3,
            "shuffle": True,
            "seed": 69,
            "max_samples": -1,
        },
    }


def merge_configs(base: dict, update: dict) -> dict:
    """Deep merge two configuration dictionaries, update takes precedence"""
    merged = base.copy()
    for k, v in update.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged
