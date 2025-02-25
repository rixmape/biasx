import json
import os
from typing import Any, Dict, Optional

import numpy as np

from .types import Explanation, FairnessScores, FeatureProbability, FeatureScore


class FaceDataset:
    """Manages the facial image dataset used for bias analysis."""

    def __init__(self, dataset_path: str, max_samples: int, shuffle: bool, seed: int):
        """Initialize the dataset from a directory of facial images."""
        self.image_paths = self._load_image_paths(dataset_path, max_samples, shuffle, seed)
        self.genders = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in self.image_paths]

    @staticmethod
    def _load_image_paths(dataset_path: str, max_samples: int, shuffle: bool, seed: int) -> list[str]:
        """Load and optionally shuffle image paths from the dataset directory."""
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(paths)
        return paths[:max_samples] if max_samples > 0 else paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.image_paths[idx], self.genders[idx]


class AnalysisDataset:
    """Manages storage and serialization of analysis results."""

    def __init__(self):
        """Initialize an empty analysis dataset."""
        self.feature_scores: FeatureScore = {}
        self.feature_probabilities: FeatureProbability = {}
        self.fairness_scores: FairnessScores = {}
        self.explanations: list[Explanation] = []

    def add_explanation(self, explanation: Explanation) -> None:
        """Add a single image analysis result."""
        self.explanations.append(explanation)

    def set_bias_metrics(self, feature_scores: FeatureScore, feature_probabilities: FeatureProbability, fairness_scores: FairnessScores) -> None:
        """Set computed bias and fairness metrics."""
        self.feature_scores = feature_scores
        self.feature_probabilities = feature_probabilities
        self.fairness_scores = fairness_scores

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format for serialization."""
        return {
            **self.fairness_scores,  # All fairness scores at root level
            "featureScores": self.feature_scores,
            "featureProbabilities": self.feature_probabilities,
            "explanations": [exp.to_dict() for exp in self.explanations],
        }

    def save(self, output_path: str) -> None:
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load_activation_map(activation_map_path: str) -> Optional[np.ndarray]:
        """Load a compressed activation map from file."""
        try:
            with np.load(activation_map_path) as data:
                return data["activation_map"]
        except (FileNotFoundError, KeyError):
            return None
