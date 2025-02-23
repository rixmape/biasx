from dataclasses import asdict, dataclass
from typing import Any, Union

from .defaults import BaseConfig, CalculatorConfig, DatasetConfig, ExplainerConfig, ModelConfig, create_default_config, merge_configs


@dataclass(frozen=True)
class Config:
    """Configuration class for the bias analysis pipeline"""

    model_path: str
    dataset_path: str
    model_options: ModelConfig
    explainer_options: ExplainerConfig
    calculator_options: CalculatorConfig
    dataset_options: DatasetConfig

    @classmethod
    def create(cls, config: Union[str, dict, BaseConfig]) -> "Config":
        """Create configuration from path or dict, merging with defaults"""
        if isinstance(config, str):
            raise ValueError("String config must include both model_path and dataset_path")

        if "model_path" not in config or "dataset_path" not in config:
            raise ValueError("Both model_path and dataset_path are required")

        base_config = merge_configs(create_default_config(config["model_path"], config["dataset_path"]), config)

        return cls(
            model_path=base_config["model_path"],
            dataset_path=base_config["dataset_path"],
            model_options=base_config["model_options"],
            explainer_options=base_config["explainer_options"],
            calculator_options=base_config["calculator_options"],
            dataset_options=base_config["dataset_options"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "model_options": asdict(self.model_options),
            "explainer_options": asdict(self.explainer_options),
            "calculator_options": asdict(self.calculator_options),
            "dataset_options": asdict(self.dataset_options),
        }
