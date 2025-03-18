"""Configuration management for gender bias experiments."""

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class DatasetConfig:
    """Configuration for facial dataset with controlled gender bias."""

    dataset_name: str = "utkface"
    dataset_size: int = 1000
    gender_ratios: dict[int, float] = None
    random_seed: int = 42
    target_gender: Optional[int] = None
    target_feature: Optional[str] = None
    padding: int = 2


@dataclass
class ClassifierConfig:
    """Configuration for gender classification model training."""

    epochs: int = 10
    batch_size: int = 64
    val_split: float = 0.2
    test_split: float = 0.1
    input_shape: Tuple[int, int, int] = (48, 48, 1)
    random_seed: int = 42

