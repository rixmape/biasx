import json
import os
from typing import Any

import numpy as np

from .types import Box


class FaceDataset:
    """Manages the facial image dataset used for bias analysis."""

    def __init__(self, dataset_path: str, max_samples: int = -1):
        """
        Initialize the dataset from a directory of facial images.

        Args:
            dataset_path: Path to directory containing facial images
            max_samples: Maximum number of samples to load (-1 for all)
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        self.image_paths = []
        self.genders = []

        paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
        if max_samples > 0:
            paths = paths[:max_samples]

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
        self.explanations = []

    def add_explanation(self, image_path: str, true_gender: int, predicted_gender: int, activation_map: np.ndarray, activation_boxes: list[Box], landmark_boxes: list[Box]) -> None:
        """Add a single image analysis result."""
        self.explanations.append(
            {
                "imagePath": image_path,
                "trueGender": true_gender,
                "predictedGender": predicted_gender,
                "activationMap": activation_map.tolist(),
                "activationBoxes": [box.to_dict() for box in activation_boxes],
                "landmarkBoxes": [box.to_dict() for box in landmark_boxes],
            }
        )

    def set_bias_metrics(self, bias_score: float, feature_scores: dict[str, float], feature_probabilities: dict[str, dict[int, float]]) -> None:
        """Set computed bias metrics."""
        self.bias_score = bias_score
        self.feature_scores = feature_scores
        self.feature_probabilities = feature_probabilities

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary format."""
        return {"biasScore": self.bias_score, "featureScores": self.feature_scores, "featureProbabilities": self.feature_probabilities, "explanations": self.explanations}

    def save(self, output_path: str) -> None:
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
