from dataclasses import dataclass
from enum import Enum
from typing import Optional

import tensorflow as tf


class Gender(Enum):
    """Enum representing genders with predefined values for MALE and FEMALE."""

    MALE = 0
    FEMALE = 1


@dataclass
class FeatureBox:
    """Dataclass encapsulating the coordinates and name of a facial feature's bounding box, with an optional importance score."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    name: str
    importance: Optional[float] = None

    def get_area(self) -> int:
        """Calculates and returns the area of the bounding box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)


@dataclass
class DatasetSplits:
    """Dataclass that holds the training, validation, and testing TensorFlow datasets."""

    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
