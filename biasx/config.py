from dataclasses import asdict, dataclass
from typing import Any, Union

from .defaults import (
    BaseConfig,
    CalculatorConfig,
    DatasetConfig,
    ExplainerConfig,
    ModelConfig,
    create_default_config,
    merge_configs,
)


@dataclass(frozen=True)
class Config:
    """Configuration class for the bias analysis pipeline"""

    model_config: ModelConfig
    explainer_config: ExplainerConfig
    calculator_config: CalculatorConfig
    dataset_config: DatasetConfig

    @classmethod
    def create(cls, config: Union[dict, BaseConfig]) -> "Config":
        """Create configuration from path or dict, merging with defaults"""
        if "model_config" not in config or "path" not in config["model_config"]:
            raise ValueError("model_config.path is required")

        base_config = merge_configs(
            create_default_config(config["model_config"]["path"]),
            config,
        )

        return cls(
            model_config=base_config["model_config"],
            explainer_config=base_config["explainer_config"],
            calculator_config=base_config["calculator_config"],
            dataset_config=base_config["dataset_config"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {
            "model_config": asdict(self.model_config),
            "explainer_config": asdict(self.explainer_config),
            "calculator_config": asdict(self.calculator_config),
            "dataset_config": asdict(self.dataset_config),
        }
