"""Model handling module for BiasX."""

from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from .config import configurable
from .types import Gender


@configurable("model")
class Model:
    """Handles loading and inference for facial classification models."""

    def __init__(self, path: str, inverted_classes: bool = False, **kwargs):
        """Initialize the classification model."""
        self.model = tf.keras.models.load_model(path)
        self.inverted_classes = inverted_classes
        self._metadata = None

    def _prepare_input(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """Prepare image for model input."""
        return np.expand_dims(preprocessed_image, axis=0) if preprocessed_image.ndim < 4 else preprocessed_image

    def _get_probabilities(self, image: np.ndarray) -> np.ndarray:
        """Get probability distribution from model output."""
        output = self.model.predict(image, verbose=0)
        return tf.nn.softmax(output)[0] if len(output.shape) > 1 else output[0]

    def _map_class_index(self, index: int) -> int:
        """Map class index accounting for inverted classes."""
        return index if not self.inverted_classes else 1 - index

    def predict(self, preprocessed_image: np.ndarray) -> Tuple[Gender, float]:
        """Make prediction from a preprocessed image."""
        batch = self._prepare_input(preprocessed_image)
        probs = self._get_probabilities(batch)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = Gender(self._map_class_index(pred_idx))

        return pred_class, confidence

    def get_class_probabilities(self, preprocessed_image: np.ndarray) -> Dict[Gender, float]:
        """Get probability distribution across all classes."""
        batch = self._prepare_input(preprocessed_image)
        probs = self._get_probabilities(batch)

        return {Gender(self._map_class_index(i)): float(prob) for i, prob in enumerate(probs)}

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        if self._metadata is None:
            self._metadata = {
                "name": getattr(self.model, "name", "unknown"),
                "layers": len(self.model.layers),
                "input_shape": self.model.input_shape[1:] if hasattr(self.model, "input_shape") else None,
                "output_shape": self.model.output_shape[1:] if hasattr(self.model, "output_shape") else None,
                "parameters": self.model.count_params(),
                "inverted_classes": self.inverted_classes,
            }
        return self._metadata
