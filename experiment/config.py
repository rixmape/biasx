from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Config:
    """Dataclass containing experiment parameters, dataset configurations, and model training settings."""

    # Experiment parameters
    replicate: int
    male_ratios: list[float]
    mask_genders: list[int]
    mask_features: list[str]
    mask_padding: int = 0
    feature_attention_threshold: Optional[float] = 0.5
    base_seed: Optional[int] = 42
    results_path: Optional[str] = "outputs"
    log_path: Optional[str] = "logs"

    # Dataset parameters
    dataset_name: Literal["utkface", "fairface"] = "utkface"
    dataset_size: int = 5000
    val_split: float = 0.1
    test_split: float = 0.2
    image_size: int = 48
    grayscale: bool = True

    # Model parameters
    batch_size: int = 64
    epochs: int = 10
