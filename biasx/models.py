from typing import Optional

import keras
import numpy as np
from PIL import Image


class ClassificationModel:
    """Handles loading and inference of the face classification model."""

    def __init__(
        self,
        model_path: str,
        target_size: Optional[tuple[int, int]] = (128, 128),
        color_mode: Optional[str] = "L",
        single_channel: Optional[bool] = False,
    ):
        """Initialize the classification model."""
        self.model = keras.models.load_model(model_path)
        self.target_size = target_size
        self.color_mode = color_mode
        self.single_channel = single_channel

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for model input."""
        image = Image.open(image_path).convert(self.color_mode).resize(self.target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=-1) if self.color_mode == "L" and not self.single_channel else image_array

    def predict(self, image: np.ndarray) -> tuple[int, float]:
        """Make a prediction on a single image"""
        output = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        return int(output[0][0]), float(output[0][1])
