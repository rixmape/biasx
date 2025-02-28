"""Model handling module for BiasX."""

from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from .config import configurable
from .types import Gender


@configurable("model")
class Model:
    """Handles loading and inference for facial classification models."""

    def __init__(self, path: str, inverted_classes: bool, **kwargs):
        """Initialize the classification model."""
        self.model = tf.keras.models.load_model(path)
        self.inverted_classes = inverted_classes
        self._metadata = None

    def predict(self, preprocessed_image: np.ndarray) -> Tuple[Gender, float]:
        """Make prediction from a preprocessed image."""
        batch = np.expand_dims(preprocessed_image, axis=0)
        output = self.model.predict(batch, verbose=0)
        probs = tf.nn.softmax(output)[0] if len(output.shape) > 1 else output[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = Gender(pred_idx if not self.inverted_classes else 1 - pred_idx)

        return pred_class, confidence

    def get_class_probabilities(self, preprocessed_image: np.ndarray) -> Dict[Gender, float]:
        """Get probability distribution across all classes."""
        batch = np.expand_dims(preprocessed_image, axis=0)
        output = self.model.predict(batch, verbose=0)
        probs = tf.nn.softmax(output)[0] if len(output.shape) > 1 else output[0]

        return {Gender(i if not self.inverted_classes else 1 - i): float(prob) for i, prob in enumerate(probs)}

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
