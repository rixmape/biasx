from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Box:
    """Represents a bounding box with optional feature label."""

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
        data = {"minX": self.min_x, "minY": self.min_y, "maxX": self.max_x, "maxY": self.max_y}
        if self.feature is not None:
            data["feature"] = self.feature
        return data
