"""Handles loading, merging, and validation of configuration settings."""

import json
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from .types import CAMMethod, ColorMode, DatasetSource, DistanceMetric, LandmarkerSource, ThresholdMethod

T = TypeVar("T")
C = TypeVar("C")

DEFAULTS = {
    "model": {
        "inverted_classes": False,
        "batch_size": 32,
    },
    "explainer": {
        "landmarker_source": "mediapipe",
        "cam_method": "gradcam++",
        "cutoff_percentile": 90,
        "threshold_method": "otsu",
        "overlap_threshold": 0.2,
        "distance_metric": "euclidean",
        "max_faces": 1,
        "batch_size": 32,
    },
    "dataset": {
        "source": "utkface",
        "image_width": 224,
        "image_height": 224,
        "color_mode": "L",
        "single_channel": False,
        "max_samples": 100,
        "shuffle": True,
        "seed": 69,
        "batch_size": 32,
    },
    "calculator": {
        "precision": 3,
    },
    "analyzer": {
        "batch_size": 32,
    },
}

ENUM_MAPPING = {
    "landmarker_source": LandmarkerSource,
    "cam_method": CAMMethod,
    "threshold_method": ThresholdMethod,
    "distance_metric": DistanceMetric,
    "source": DatasetSource,
    "color_mode": ColorMode,
}


def configurable(component_name: Optional[str] = None) -> Callable:
    """Decorator to apply configuration with defaults to a class."""

    def decorator(cls: Type[C]) -> Type[C]:
        orig_init = cls.__init__

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            section = component_name or cls.__name__.lower()

            config = {**DEFAULTS.get(section, {}), **{k: v for k, v in kwargs.items() if v is not None}}

            for key, value in list(config.items()):
                if key in ENUM_MAPPING and isinstance(value, str):
                    try:
                        enum_class = ENUM_MAPPING[key]
                        config[key] = enum_class(value)
                    except (ValueError, KeyError):
                        try:
                            config[key] = enum_class[value]
                        except KeyError:
                            pass  # Keep original value if conversion fails

            orig_init(self, *args, **config)

        cls.__init__ = new_init
        return cls

    return decorator


class Config:
    """Main configuration class for the bias analysis pipeline."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize configuration from a dictionary."""
        self.config_dict = config_dict

        if not isinstance(config_dict, dict) or "model" not in config_dict or "path" not in config_dict.get("model", {}):
            raise ValueError("model.path is required in configuration")

        self.model_path = config_dict.get("model", {}).get("path")

        self.model = self._prepare_section("model")
        self.explainer = self._prepare_section("explainer")
        self.dataset = self._prepare_section("dataset")
        self.calculator = self._prepare_section("calculator")
        self.analyzer = self._prepare_section("analyzer")

    def _prepare_section(self, section_name: str) -> Dict[str, Any]:
        """Prepare a configuration section with defaults and enum conversion."""
        config = {**DEFAULTS.get(section_name, {}), **self.config_dict.get(section_name, {})}

        for key, value in list(config.items()):
            if key in ENUM_MAPPING and isinstance(value, str):
                enum_class = ENUM_MAPPING[key]
                try:
                    config[key] = enum_class(value)
                except (ValueError, KeyError):
                    try:
                        config[key] = enum_class[value]
                    except KeyError:
                        pass  # Keep original value if conversion fails

        return config

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
        return {"model": self.model, "explainer": self.explainer, "dataset": self.dataset, "calculator": self.calculator, "analyzer": self.analyzer}

    def save(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
