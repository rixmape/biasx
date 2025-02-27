"""
Configuration module for the BiasX library.
Handles loading, merging, and validation of configuration settings.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from .types import CAMMethod, ColorMode, DatasetSource, DistanceMetric, ThresholdMethod


@dataclass
class ModelConfig:
    """Configuration for the classification model."""

    path: str
    inverted_classes: bool = False


@dataclass
class ExplainerConfig:
    """Configuration for the visual explainer."""

    cam_method: CAMMethod = CAMMethod.GRADCAM_PLUS_PLUS
    cutoff_percentile: int = 90
    threshold_method: ThresholdMethod = ThresholdMethod.OTSU
    overlap_threshold: float = 0.2
    distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN


@dataclass
class DatasetConfig:
    """Configuration for the dataset loader."""

    source: DatasetSource = DatasetSource.UTKFACE
    image_width: int = 224
    image_height: int = 224
    color_mode: ColorMode = ColorMode.GRAYSCALE
    single_channel: bool = False
    max_samples: int = 100
    shuffle: bool = True
    seed: int = 69


@dataclass
class CalculatorConfig:
    """Configuration for the bias calculator."""

    precision: int = 3


@dataclass
class Config:
    """Main configuration class for the bias analysis pipeline."""

    model_config: ModelConfig
    explainer_config: ExplainerConfig
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    calculator_config: CalculatorConfig = field(default_factory=CalculatorConfig)

    @classmethod
    def create(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dict, merging with defaults."""
        if "model_config" not in config_dict or "path" not in config_dict["model_config"]:
            raise ValueError("model_config.path is required")

        model_config = cls._create_model_config(config_dict.get("model_config", {}))
        explainer_config = cls._create_explainer_config(config_dict.get("explainer_config", {}))
        dataset_config = cls._create_dataset_config(config_dict.get("dataset_config", {}))
        calculator_config = cls._create_calculator_config(config_dict.get("calculator_config", {}))

        return cls(
            model_config=model_config,
            explainer_config=explainer_config,
            dataset_config=dataset_config,
            calculator_config=calculator_config,
        )

    @staticmethod
    def _create_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary."""
        return ModelConfig(path=config_dict["path"], inverted_classes=config_dict.get("inverted_classes", False))

    @staticmethod
    def _create_explainer_config(config_dict: Dict[str, Any]) -> ExplainerConfig:
        """Create ExplainerConfig from dictionary with enum conversion."""
        cam_method = config_dict.get("cam_method")
        if isinstance(cam_method, str):
            cam_method = CAMMethod(cam_method)

        threshold_method = config_dict.get("threshold_method")
        if isinstance(threshold_method, str):
            threshold_method = ThresholdMethod(threshold_method)

        distance_metric = config_dict.get("distance_metric")
        if isinstance(distance_metric, str):
            distance_metric = DistanceMetric(distance_metric)

        return ExplainerConfig(
            cam_method=cam_method or CAMMethod.GRADCAM_PLUS_PLUS,
            cutoff_percentile=config_dict.get("cutoff_percentile", 90),
            threshold_method=threshold_method or ThresholdMethod.OTSU,
            overlap_threshold=config_dict.get("overlap_threshold", 0.2),
            distance_metric=distance_metric or DistanceMetric.EUCLIDEAN,
        )

    @staticmethod
    def _create_dataset_config(config_dict: Dict[str, Any]) -> DatasetConfig:
        """Create DatasetConfig from dictionary with enum conversion."""
        source = config_dict.get("source")
        if isinstance(source, str):
            source = DatasetSource(source)

        color_mode = config_dict.get("color_mode")
        if isinstance(color_mode, str):
            color_mode = ColorMode(color_mode)

        return DatasetConfig(
            source=source or DatasetSource.UTKFACE,
            image_width=config_dict.get("image_width", 224),
            image_height=config_dict.get("image_height", 224),
            color_mode=color_mode or ColorMode.GRAYSCALE,
            single_channel=config_dict.get("single_channel", False),
            max_samples=config_dict.get("max_samples", 100),
            shuffle=config_dict.get("shuffle", True),
            seed=config_dict.get("seed", 69),
        )

    @staticmethod
    def _create_calculator_config(config_dict: Dict[str, Any]) -> CalculatorConfig:
        """Create CalculatorConfig from dictionary."""
        return CalculatorConfig(precision=config_dict.get("precision", 3))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "model_config": asdict(self.model_config),
            "explainer_config": asdict(self.explainer_config),
            "dataset_config": asdict(self.dataset_config),
            "calculator_config": asdict(self.calculator_config),
        }
