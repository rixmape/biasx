from typing import TypedDict

from typing_extensions import NotRequired, Required

from .types import CAMMethod, ColorMode, DatasetSource, DistanceMetric, ThresholdMethod


class ModelConfig(TypedDict, total=False):
    path: Required[str]
    image_width: NotRequired[int]
    image_height: NotRequired[int]
    color_mode: NotRequired[ColorMode]
    single_channel: NotRequired[bool]
    inverted_classes: NotRequired[bool]


class ExplainerConfig(TypedDict, total=False):
    max_faces: NotRequired[int]
    cam_method: NotRequired[CAMMethod]
    cutoff_percentile: NotRequired[int]
    threshold_method: NotRequired[ThresholdMethod]
    overlap_threshold: NotRequired[float]
    distance_metric: NotRequired[DistanceMetric]
    activation_maps_path: NotRequired[str]


class CalculatorConfig(TypedDict, total=False):
    ndigits: NotRequired[int]


class DatasetConfig(TypedDict, total=False):
    source: NotRequired[DatasetSource]
    max_samples: NotRequired[int]
    shuffle: NotRequired[bool]
    seed: NotRequired[int]


class BaseConfig(TypedDict, total=False):
    model_config: NotRequired[ModelConfig]
    explainer_config: NotRequired[ExplainerConfig]
    calculator_config: NotRequired[CalculatorConfig]
    dataset_config: NotRequired[DatasetConfig]


def create_default_config(model_path: str) -> BaseConfig:
    """Create a complete configuration with all defaults"""
    return {
        "model_config": {
            "path": model_path,
            "image_width": 224,
            "image_height": 224,
            "color_mode": "L",
            "single_channel": False,
            "inverted_classes": False,
        },
        "explainer_config": {
            "max_faces": 1,
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
            "overlap_threshold": 0.2,
            "distance_metric": "euclidean",
            "activation_maps_path": "outputs/activation_maps",
        },
        "calculator_config": {
            "ndigits": 3,
        },
        "dataset_config": {
            "source": "utkface",
            "max_samples": -1,
            "shuffle": True,
            "seed": 69,
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
