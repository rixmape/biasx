from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# isort: off
from datatypes import ArtifactSavingLevel, DatasetSource, MaskDetails


class CoreConfig(BaseModel):
    replicate_count: int = Field(default=5, gt=0)
    male_proportion_targets: List[float] = Field(default=[0.5])
    masking_configs: Optional[List[MaskDetails]] = Field(default=None)
    mask_pixel_padding: int = Field(default=2, ge=0)
    key_feature_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    base_random_seed: int = Field(default=42, ge=0)

    @field_validator("male_proportion_targets")
    @classmethod
    def check_proportions(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("male_proportion_targets cannot be empty")
        if not all(0.0 <= p <= 1.0 for p in v):
            raise ValueError("All male_proportion_targets must be between 0.0 and 1.0")
        return v


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
    base_dir: str = "outputs"
    log_dir: str = "logs"
    artifact_level: ArtifactSavingLevel = ArtifactSavingLevel.FULL

    @field_validator("base_dir", "log_dir")
    @classmethod
    def check_dir_names(cls, v: str) -> str:
        if not v:
            raise ValueError("Directory names cannot be empty")
        return v


class ExperimentsConfig(BaseModel):
    core: CoreConfig = Field(default_factory=CoreConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
