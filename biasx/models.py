from typing import Optional

import keras
import numpy as np
from PIL import Image


class ClassificationModel:
    """Handles loading and inference of the face classification model."""

    def __init__(
        self,
        model_path: str,
        target_size: tuple[int, int],
        color_mode: Optional[str] = "L",
        single_channel: Optional[bool] = False,
    ):
        """
        Initialize the classification model.

        Args:
            model_path: Path to saved model file
            color_mode: Color mode for input images. See PIL.Image.convert() for options.
            target_size: Size to resize input images to

        """
        self.model = keras.models.load_model(model_path)
        self.target_size = target_size
        self.color_mode = color_mode
        self.single_channel = single_channel

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for model input."""
        with Image.open(image_path) as image:
            image = image.convert(self.color_mode)
            image = image.resize(self.target_size)
            image = np.array(image, dtype=np.float32) / 255.0
            if self.color_mode == "L" and not self.single_channel:
                image = np.expand_dims(image, axis=-1)
            return image

    def predict(self, image: np.ndarray) -> int:
        """Generate prediction for a single preprocessed image."""
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        return int(np.argmax(prediction, axis=1)[0])
