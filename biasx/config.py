from dataclasses import asdict, dataclass
from typing import Any, Union

from .defaults import AnalysisConfig, BaseConfig, ExplainerConfig, ModelConfig, create_default_config, merge_configs


@dataclass(frozen=True)
class Config:
    """Configuration class for the bias analysis pipeline"""

    model_path: str
    model_options: ModelConfig
    explainer_options: ExplainerConfig
    calculator_options: AnalysisConfig

    @classmethod
    def create(cls, config: Union[str, dict, BaseConfig]) -> "Config":
        """Create configuration from path or dict, merging with defaults"""
        if isinstance(config, str):
            base_config = create_default_config(config)
        else:
            if "model_path" not in config:
                raise ValueError("model_path is required")
            base_config = merge_configs(create_default_config(config["model_path"]), config)

        return cls(model_path=base_config["model_path"], model_options=base_config["model_options"], explainer_options=base_config["explainer_options"], calculator_options=base_config["calculator_options"])

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format"""
        return {"model_path": self.model_path, "model_options": asdict(self.model_options), "explainer_options": asdict(self.explainer_options), "calculator_options": asdict(self.calculator_options)}
