from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Box:
    """Represents a bounding box with an optional feature label."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    feature: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        """Compute center coordinates of the box."""
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def area(self) -> float:
        """Compute area of the box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def to_dict(self) -> dict[str, Any]:
        """Convert box to dictionary format."""
        return {**{"minX": self.min_x, "minY": self.min_y, "maxX": self.max_x, "maxY": self.max_y}, **({"feature": self.feature} if self.feature else {})}


@dataclass
class Explanation:
    """Encapsulates analysis results and explanations for a single image."""

    image_path: str
    true_gender: int
    predicted_gender: int
    activation_map: np.ndarray
    activation_boxes: list[Box]
    landmark_boxes: list[Box]

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to dictionary format for serialization."""
        return {
            "imagePath": self.image_path,
            "trueGender": self.true_gender,
            "predictedGender": self.predicted_gender,
            "activationMap": self.activation_map.tolist(),
            "activationBoxes": [box.to_dict() for box in self.activation_boxes],
            "landmarkBoxes": [box.to_dict() for box in self.landmark_boxes],
        }
