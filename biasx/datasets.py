import json
import os
from typing import Any, Optional

import numpy as np

from .types import Explanation


class FaceDataset:
    """Manages the facial image dataset used for bias analysis."""

    def __init__(
        self,
        dataset_path: str,
        max_samples: int = -1,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 69,
    ):
        """
        Initialize the dataset from a directory of facial images.

        Args:
            dataset_path: Path to directory containing facial images
            max_samples: Maximum number of samples to load (-1 for all)
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(paths)
        if max_samples > 0:
            paths = paths[:max_samples]

        self.image_paths = []
        self.genders = []

        for path in paths:
            filename = os.path.basename(path).split(".")[0]
            gender = int(filename.split("_")[1])
            self.image_paths.append(path)
            self.genders.append(gender)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.image_paths[idx], self.genders[idx]


class AnalysisDataset:
    """Manages storage and serialization of analysis results."""

    def __init__(self):
        """Initialize empty analysis dataset."""
        self.bias_score = None
        self.feature_scores = {}
        self.feature_probabilities = {}
        self.explanations: list[Explanation] = []

    def add_explanation(self, explanation: Explanation) -> None:
        """Add a single image analysis result."""
        self.explanations.append(explanation)

    def set_bias_metrics(
        self,
        bias_score: float,
        feature_scores: dict[str, float],
        feature_probabilities: dict[str, dict[int, float]],
    ) -> None:
        """Set computed bias metrics."""
        self.bias_score = bias_score
        self.feature_scores = feature_scores
        self.feature_probabilities = feature_probabilities

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary format."""
        return {
            "biasScore": self.bias_score,
            "featureScores": self.feature_scores,
            "featureProbabilities": self.feature_probabilities,
            "explanations": [exp.to_dict() for exp in self.explanations],
        }

    def save(self, output_path: str) -> None:
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
