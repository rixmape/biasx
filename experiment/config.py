import hashlib
from functools import cache, cached_property
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# isort: off
from datatypes import OutputLevel, DatasetSource, Feature, Gender, ProtectedAttribute


class CoreConfig(BaseModel):
    target_male_proportion: float = Field(..., ge=0.0, le=1.0)
    mask_gender: Optional[Gender] = Field(default=None)
    mask_features: Optional[List[Feature]] = Field(default=None)
    mask_pixel_padding: int = Field(default=2, ge=0)
    key_feature_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    random_seed: int = Field(default=42, ge=0)
    protected_attribute: Optional[ProtectedAttribute] = Field(default=ProtectedAttribute.GENDER)


class DatasetConfig(BaseModel):
    source_name: DatasetSource = DatasetSource.UTKFACE
    target_size: int = Field(default=5000, gt=0)
    validation_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    image_size: int = Field(default=48, gt=0)
    use_grayscale: bool = False

    @model_validator(mode="after")
    def check_split_ratios_sum(self) -> "DatasetConfig":
        if self.validation_ratio + self.test_ratio >= 1.0:
            raise ValueError("Validation and test ratios must sum to less than 1.0")
        return self


class ModelConfig(BaseModel):
    batch_size: int = Field(default=64, gt=0)
    epochs: int = Field(default=10, gt=0)


class OutputConfig(BaseModel):
    base_path: str = "outputs"
    log_path: str = "logs"
    level: OutputLevel = OutputLevel.FULL

    @field_validator("base_path", "log_path")
    @classmethod
    def check_paths(cls, v: str) -> str:
        if not v:
            raise ValueError("Directory names cannot be empty")
        return v


class Config(BaseModel):
    core: CoreConfig = Field(default_factory=CoreConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @cached_property
    def experiment_id(self) -> str:
        """Generates a unique experiment ID based on the current configuration."""
        config_json = self.model_dump_json()
        hash_object = hashlib.sha256(config_json.encode())
        return hash_object.hexdigest()[:16]
