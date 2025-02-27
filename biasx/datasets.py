import json
import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .types import DatasetSource, Explanation, FairnessScores, FeatureProbability, FeatureScore


class FaceDataset:
    """Manages the facial image dataset used for bias analysis."""

    DATASET_INFO = {
        "utkface": {
            "repo_id": "rixmape/utkface",
            "filename": "data/train-00000-of-00001.parquet",
            "repo_type": "dataset",
            "image_col": "image",
            "gender_col": "gender",
        },
        "fairface": {
            "repo_id": "rixmape/fairface",
            "filename": "data/train-00000-of-00001.parquet",
            "repo_type": "dataset",
            "image_col": "image",
            "gender_col": "gender",
        },
    }

    def __init__(
        self,
        source: DatasetSource = "utkface",
        max_samples: int = -1,
        shuffle: bool = True,
        seed: int = 69,
    ):
        """Initialize the dataset from a HuggingFace dataset source."""
        self.source = source
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.temp_dir = tempfile.TemporaryDirectory()

        # Get dataset info
        if source not in self.DATASET_INFO:
            raise ValueError(f"Unknown dataset source: {source}. Available options: {', '.join(self.DATASET_INFO.keys())}")

        self.dataset_info = self.DATASET_INFO[source]

        # Download and process dataset
        self.image_paths, self.genders = self._load_dataset()

    def _load_dataset(self) -> tuple[list[str], list[int]]:
        """Download and process dataset from HuggingFace."""
        # Download the parquet file
        parquet_path = hf_hub_download(
            repo_id=self.dataset_info["repo_id"],
            filename=self.dataset_info["filename"],
            repo_type=self.dataset_info["repo_type"],
        )

        # Read the parquet file
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Process the dataset
        return self._process_dataset(df)

    def _process_dataset(self, df) -> tuple[list[str], list[int]]:
        """Process dataset and extract image paths and genders."""
        image_paths = []
        genders = []

        image_col = self.dataset_info["image_col"]
        gender_col = self.dataset_info["gender_col"]

        if self.max_samples > 0:
            df = df.sample(min(self.max_samples, len(df)), random_state=self.seed) if self.shuffle else df.head(self.max_samples)
        elif self.shuffle:
            df = df.sample(frac=1, random_state=self.seed)

        # Process each row
        for idx, row in df.iterrows():
            # Save image to temp directory
            img = row[image_col]
            if isinstance(img, dict) and "bytes" in img:  # HuggingFace format
                img_data = img["bytes"]
            else:
                img_data = img

            img_path = os.path.join(self.temp_dir.name, f"{idx}.jpg")
            with open(img_path, "wb") as f:
                f.write(img_data)

            # Get gender (ensure it's binary 0 or 1)
            gender = int(row[gender_col])

            image_paths.append(img_path)
            genders.append(gender)

        return image_paths, genders

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.image_paths[idx], self.genders[idx]

    def __del__(self):
        """Clean up temporary directory when object is destroyed."""
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()


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
