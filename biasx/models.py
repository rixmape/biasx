"""
Model handling module for BiasX.
Provides classes for model inference.
"""

import numpy as np
import tensorflow as tf

from .types import Gender


class ClassificationModel:
    """Handles loading and inference for facial classification models."""

    def __init__(
        self,
        path: str,
        inverted_classes: bool = False,
    ):
        """Initialize the classification model."""
        self.model = tf.keras.models.load_model(path)
        self.inverted_classes = inverted_classes

    def predict(self, preprocessed_image: np.ndarray) -> tuple[Gender, float]:
        """Make prediction from a preprocessed image."""
        batch = np.expand_dims(preprocessed_image, axis=0)
        output = self.model.predict(batch, verbose=0)
        probs = tf.nn.softmax(output)[0] if len(output.shape) > 1 else output[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = Gender(pred_idx if not self.inverted_classes else 1 - pred_idx)

        return pred_class, confidence
