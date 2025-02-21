from typing import Any

import keras
import numpy as np
from PIL import Image


class ClassificationModel:
    """Handles loading and inference of the face classification model."""

    def __init__(self, model_path: str, input_shape: tuple[int, int, int], preprocessing: dict[str, Any]):
        """
        Initialize the classification model.

        Args:
            model_path: Path to saved model file
            input_shape: Expected shape of input images (height, width, channels)
            preprocessing: dictionary of preprocessing parameters
        """
        self.model = keras.models.load_model(model_path)
        self.input_shape = input_shape
        self.preprocessing = preprocessing

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for model input."""
        with Image.open(image_path) as image:
            image = image.convert(self.preprocessing["color_mode"])
            image = image.resize(self.preprocessing["target_size"])
            return np.array(image, dtype=np.float32) / 255.0

    def predict(self, image: np.ndarray) -> int:
        """Generate prediction for a single preprocessed image."""
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        return int(np.argmax(prediction, axis=1)[0])
