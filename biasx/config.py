"""
Configuration module for the bias analysis pipeline.
Handles loading, merging, and validation of configuration settings.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from .types import CAMMethod, ColorMode, DatasetSource, DistanceMetric, ThresholdMethod


@dataclass
class ModelConfig:
    """Configuration for the classification model."""

    path: str
    image_width: int = 224
    image_height: int = 224
    color_mode: ColorMode = "L"
    single_channel: bool = False
    inverted_classes: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for unpacking with **."""
        return asdict(self)


@dataclass
class ExplainerConfig:
    """Configuration for the visual explainer."""

    max_faces: int = 1
    cam_method: CAMMethod = "gradcam++"
    cutoff_percentile: int = 90
    threshold_method: ThresholdMethod = "otsu"
    overlap_threshold: float = 0.2
    distance_metric: DistanceMetric = "euclidean"
    activation_maps_path: str = "outputs/activation_maps"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for unpacking with **."""
        return asdict(self)


@dataclass
class CalculatorConfig:
    """Configuration for the bias calculator."""

    ndigits: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for unpacking with **."""
        return asdict(self)


@dataclass
class DatasetConfig:
    """Configuration for the dataset loader."""

    source: DatasetSource = "utkface"
    max_samples: int = -1
    shuffle: bool = True
    seed: int = 69

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for unpacking with **."""
        return asdict(self)


@dataclass
class Config:
    """Main configuration class for the bias analysis pipeline."""

    model_config: ModelConfig
    explainer_config: ExplainerConfig = field(default_factory=ExplainerConfig)
    calculator_config: CalculatorConfig = field(default_factory=CalculatorConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def create(cls, config: Dict[str, Any]) -> "Config":
        """Create configuration from dict, merging with defaults."""
        if "model_config" not in config or "path" not in config["model_config"]:
            raise ValueError("model_config.path is required")

        model_config = ModelConfig(path=config["model_config"]["path"])

        model_config = cls._update_from_dict(model_config, config.get("model_config", {}))
        explainer_config = cls._update_from_dict(ExplainerConfig(), config.get("explainer_config", {}))
        calculator_config = cls._update_from_dict(CalculatorConfig(), config.get("calculator_config", {}))
        dataset_config = cls._update_from_dict(DatasetConfig(), config.get("dataset_config", {}))

        return cls(model_config=model_config, explainer_config=explainer_config, calculator_config=calculator_config, dataset_config=dataset_config)

    @staticmethod
    def _update_from_dict(instance, update_dict):
        """Update a dataclass instance with values from a dictionary."""
        if not update_dict:
            return instance

        instance_dict = asdict(instance)

        for k, v in update_dict.items():
            if k in instance_dict:
                instance_dict[k] = v

        return type(instance)(**instance_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return asdict(self)
