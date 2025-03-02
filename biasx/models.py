"""Model handling module for BiasX."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from .config import configurable
from .types import Gender


@configurable("model")
class Model:
    """Handles loading and inference for facial classification models."""

    def __init__(self, path: str, inverted_classes: bool = False, batch_size: int = 32, **kwargs):
        """Initialize the classification model."""
        self.model = tf.keras.models.load_model(path)
        self.inverted_classes = inverted_classes
        self.batch_size = batch_size
        self._metadata = None

    def _prepare_input(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Prepare images for model input as a batch."""
        if len(preprocessed_images) == 0:
            return np.empty((0,) + self.model.input_shape[1:])

        if isinstance(preprocessed_images, list):
            processed_batch = []
            for img in preprocessed_images:
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                processed_batch.append(img)
            return np.stack(processed_batch)

        batch = preprocessed_images
        if batch.ndim == 2:
            batch = np.expand_dims(batch, axis=-1)
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=0)

        return batch

    def _get_probabilities(self, batch: np.ndarray) -> np.ndarray:
        """Get probability distributions from model output for a batch."""
        if len(batch) == 0:
            return np.empty((0, 2))

        output = self.model.predict(batch, verbose=0, batch_size=self.batch_size)
        if len(output.shape) > 1 and output.shape[1] > 1:
            return tf.nn.softmax(output).numpy()
        return output

    def _process_predictions(self, probs: np.ndarray) -> List[Tuple[Gender, float]]:
        """Process probability array into prediction tuples."""
        results = []
        for sample_probs in probs:
            pred_idx = int(np.argmax(sample_probs))
            confidence = float(sample_probs[pred_idx])
            pred_class = Gender(pred_idx if not self.inverted_classes else 1 - pred_idx)
            results.append((pred_class, confidence))
        return results

    def predict(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> List[Tuple[Gender, float]]:
        """Make predictions from preprocessed images."""
        batch = self._prepare_input(preprocessed_images)
        if len(batch) == 0:
            return []

        probs = self._get_probabilities(batch)
        return self._process_predictions(probs)

    def get_class_probabilities(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> List[Dict[Gender, float]]:
        """Get probability distributions across all classes for a batch."""
        batch = self._prepare_input(preprocessed_images)
        if len(batch) == 0:
            return []

        probs = self._get_probabilities(batch)

        results = []
        for sample_probs in probs:
            class_probs = {}
            for i, prob in enumerate(sample_probs):
                gender_class = Gender(i if not self.inverted_classes else 1 - i)
                class_probs[gender_class] = float(prob)
            results.append(class_probs)

        return results
