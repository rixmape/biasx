"""Handles loading, merging, and validation of configuration settings."""

import json
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from .types import CAMMethod, ColorMode, DatasetSource, DistanceMetric, LandmarkerSource, ThresholdMethod
from .utils import parse_enum

T = TypeVar("T")
C = TypeVar("C")

DEFAULTS = {
    "model": {
        "inverted_classes": False,
    },
    "explainer": {
        "landmarker_source": LandmarkerSource.MEDIAPIPE,
        "cam_method": CAMMethod.GRADCAM_PLUS_PLUS,
        "cutoff_percentile": 90,
        "threshold_method": ThresholdMethod.OTSU,
        "overlap_threshold": 0.2,
        "distance_metric": DistanceMetric.EUCLIDEAN,
    },
    "dataset": {
        "source": DatasetSource.UTKFACE,
        "image_width": 224,
        "image_height": 224,
        "color_mode": ColorMode.GRAYSCALE,
        "single_channel": False,
        "max_samples": 100,
        "shuffle": True,
        "seed": 69,
    },
    "calculator": {
        "precision": 3,
    },
    "landmarker": {
        "max_faces": 1,
        "source": LandmarkerSource.MEDIAPIPE,
    },
}

ENUM_TYPES = {
    "landmarker_source": LandmarkerSource,
    "cam_method": CAMMethod,
    "threshold_method": ThresholdMethod,
    "distance_metric": DistanceMetric,
    "source": DatasetSource,
    "color_mode": ColorMode,
}


def configurable(component_name: Optional[str] = None) -> Callable:
    def decorator(cls: Type[C]) -> Type[C]:
        orig_init = cls.__init__

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            section = component_name or cls.__name__.lower()

            config = DEFAULTS.get(section, {}).copy()
            config.update({k: v for k, v in kwargs.items() if v is not None})

            for key in list(config.keys()):
                if key in ENUM_TYPES:
                    default_value = DEFAULTS.get(section, {}).get(key)
                    config[key] = parse_enum(config[key], ENUM_TYPES[key], default_value)

            orig_init(self, *args, **config)

        cls.__init__ = new_init
        return cls

    return decorator


class Config:
    """Main configuration class for the bias analysis pipeline."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from a dictionary."""
        self.config_dict = config_dict

        if isinstance(config_dict, dict) and not isinstance(config_dict, Config):
            if "model" not in config_dict or "path" not in config_dict.get("model", {}):
                raise ValueError("model.path is required")

        self.model_path = config_dict.get("model", {}).get("path")

        self.model = self._prepare_section("model")
        self.explainer = self._prepare_section("explainer")
        self.dataset = self._prepare_section("dataset")
        self.calculator = self._prepare_section("calculator")

    def _prepare_section(self, config_key: str) -> Dict[str, Any]:
        """Prepare a configuration section with defaults and enum conversion."""
        section = DEFAULTS.get(config_key, {}).copy()

        user_config = self.config_dict.get(config_key, {})
        section.update({k: v for k, v in user_config.items() if v is not None})

        for key, value in list(section.items()):
            if key in ENUM_TYPES and value is not None:
                section[key] = parse_enum(value, ENUM_TYPES[key])

        return section

    @classmethod
    def create(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dict."""
        return cls(config_dict)

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Create configuration from a JSON file."""
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls.create(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "model": self.model,
            "explainer": self.explainer,
            "dataset": self.dataset,
            "calculator": self.calculator,
        }

    def save(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
