from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Gender(Enum):
    """Enum representing genders with predefined values for MALE and FEMALE."""

    MALE = 0
    FEMALE = 1


class FeatureName(Enum):
    """Facial feature types used for landmark identification."""

    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    LIPS = "lips"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    CHIN = "chin"
    FOREHEAD = "forehead"
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"


@dataclass
class FacialFeature:
    """Dataclass encapsulating the coordinates and name of a facial feature's bounding box, with an optional attention score."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    name: str
    attention: Optional[float] = 0

    def get_area(self) -> int:
        """Calculates and returns the area of the bounding box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)
