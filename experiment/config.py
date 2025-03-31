from dataclasses import dataclass, field
from typing import Literal, Optional

# isort: off
from datatypes import Gender, FeatureName


@dataclass
class Config:
    """Configuration dataclass for experiment parameters, dataset, and model settings."""

    replicate: int
    male_ratios: Optional[list[float]] = field(default_factory=lambda: [0.5])
    mask_genders: Optional[list[int]] = field(default_factory=list)
    mask_features: Optional[list[str]] = field(default_factory=list)
    mask_padding: int = 0
    feature_attention_threshold: Optional[float] = 0.5
    base_seed: Optional[int] = 42
    results_path: str = "outputs"
    log_path: str = "logs"

    dataset_name: Literal["utkface", "fairface"] = "utkface"
    dataset_size: int = 5000
    val_split: float = 0.1
    test_split: float = 0.2
    image_size: int = 48
    grayscale: bool = True

    batch_size: int = 64
    epochs: int = 10

    def _validate_experiment_params(self):
        """Validates experiment-specific parameters."""
        if not isinstance(self.replicate, int) or self.replicate <= 0:
            raise ValueError(f"Replicate count must be a positive integer. Got {self.replicate}.")

        if self.male_ratios is not None:
            if not self.male_ratios:
                raise ValueError("male_ratios list cannot be empty if provided.")
            for ratio in self.male_ratios:
                if not isinstance(ratio, float) or not (0.0 <= ratio <= 1.0):
                    raise ValueError(f"Each male_ratio must be a float between 0.0 and 1.0. Got {ratio}.")

        if self.mask_genders is not None:
            allowed_genders = [g.value for g in Gender]
            for gender_val in self.mask_genders:
                if not isinstance(gender_val, int) or gender_val not in allowed_genders:
                    raise ValueError(f"Each mask_gender must be an integer representing a Gender value ({allowed_genders}). Got {gender_val}.")

        if self.mask_features is not None:
            allowed_features = [f.value for f in FeatureName]
            for feature_name in self.mask_features:
                if not isinstance(feature_name, str) or feature_name not in allowed_features:
                    raise ValueError(f"Each mask_feature must be a valid FeatureName string ({allowed_features}). Got {feature_name}.")

        if not isinstance(self.mask_padding, int) or self.mask_padding < 0:
            raise ValueError(f"Mask padding must be a non-negative integer. Got {self.mask_padding}.")

        if self.feature_attention_threshold is not None:
            if not isinstance(self.feature_attention_threshold, float) or not (0.0 <= self.feature_attention_threshold <= 1.0):
                raise ValueError(f"Feature attention threshold must be a float between 0.0 and 1.0 if specified. Got {self.feature_attention_threshold}.")

        if self.base_seed is not None and (not isinstance(self.base_seed, int) or self.base_seed < 0):
            raise ValueError(f"Base seed must be a non-negative integer if specified. Got {self.base_seed}.")

        if not isinstance(self.results_path, str) or not self.results_path:
            raise ValueError("Results path must be a non-empty string.")

        if not isinstance(self.log_path, str) or not self.log_path:
            raise ValueError("Log path must be a non-empty string.")

    def _validate_dataset_params(self):
        """Validates dataset-specific parameters."""
        if not isinstance(self.dataset_size, int) or self.dataset_size <= 0:
            raise ValueError(f"Dataset size must be a positive integer. Got {self.dataset_size}.")

        if not isinstance(self.val_split, float) or not (0.0 <= self.val_split < 1.0):
            raise ValueError(f"Validation split must be a float between 0.0 and 1.0 (exclusive of 1.0). Got {self.val_split}.")

        if not isinstance(self.test_split, float) or not (0.0 <= self.test_split < 1.0):
            raise ValueError(f"Test split must be a float between 0.0 and 1.0 (exclusive of 1.0). Got {self.test_split}.")

        if self.val_split + self.test_split >= 1.0:
            raise ValueError(f"Validation and test splits must sum to less than 1.0. Got sum: {self.val_split + self.test_split}.")

        if not isinstance(self.image_size, int) or self.image_size <= 0:
            raise ValueError(f"Image size must be a positive integer. Got {self.image_size}.")

    def _validate_model_params(self):
        """Validates model-specific parameters."""
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"Batch size must be a positive integer. Got {self.batch_size}.")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"Epochs must be a positive integer. Got {self.epochs}.")

    def __post_init__(self):
        """Runs all validation checks after initialization."""
        self._validate_experiment_params()
        self._validate_dataset_params()
        self._validate_model_params()
