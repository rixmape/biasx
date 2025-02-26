import keras
import numpy as np
from PIL import Image

from .types import ColorMode, Gender


class ClassificationModel:
    """Handles loading and inference of the face classification model."""

    def __init__(
        self,
        path: str,
        image_width: int,
        image_height: int,
        color_mode: ColorMode,
        single_channel: bool,
        inverted_classes: bool,
    ):
        """Initialize the classification model."""
        self.model = keras.models.load_model(path)
        self.image_width = image_width
        self.image_height = image_height
        self.color_mode = color_mode
        self.single_channel = single_channel
        self.inverted_classes = inverted_classes

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for model input."""
        image = Image.open(image_path).convert(self.color_mode).resize((self.image_width, self.image_height))
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=-1) if self.color_mode == "L" and not self.single_channel else img_array

    def predict(self, image: np.ndarray) -> tuple[Gender, float]:
        """Make single prediction with confidence score."""
        batch = np.expand_dims(image, axis=0)
        output = self.model.predict(batch, verbose=0)

        probs = keras.activations.softmax(output)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = pred_idx if not self.inverted_classes else 1 - pred_idx

        return pred_class, confidence
