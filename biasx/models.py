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
        if isinstance(preprocessed_images, list):
            if not preprocessed_images:
                return np.empty((0,) + self.model.input_shape[1:])

            processed_batch = []
            for img in preprocessed_images:
                if img.ndim == 2:  # Grayscale image without channel dimension
                    img = np.expand_dims(img, axis=-1)
                processed_batch.append(img)

            return np.stack(processed_batch)

        if preprocessed_images.ndim == 2:  # Single grayscale without channel
            preprocessed_images = np.expand_dims(preprocessed_images, axis=-1)

        if preprocessed_images.ndim == 3:  # Single image with channels
            return np.expand_dims(preprocessed_images, axis=0)

        return preprocessed_images

    def _get_probabilities(self, images: np.ndarray) -> np.ndarray:
        """Get probability distributions from model output for a batch."""
        if len(images) == 0:
            return np.empty((0, 2))

        output = self.model.predict(images, verbose=0, batch_size=self.batch_size)
        if len(output.shape) > 1 and output.shape[1] > 1:
            return tf.nn.softmax(output).numpy()
        return output

    def _map_class_index(self, index: int) -> int:
        """Map class index accounting for inverted classes."""
        return index if not self.inverted_classes else 1 - index

    def predict(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> List[Tuple[Gender, float]]:
        """Make predictions from preprocessed images."""
        batch = self._prepare_input(preprocessed_images)
        if len(batch) == 0:
            return []

        probs = self._get_probabilities(batch)
        results = []

        for sample_probs in probs:
            pred_idx = int(np.argmax(sample_probs))
            confidence = float(sample_probs[pred_idx])
            pred_class = Gender(self._map_class_index(pred_idx))
            results.append((pred_class, confidence))

        return results

    def get_class_probabilities(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> List[Dict[Gender, float]]:
        """Get probability distributions across all classes for a batch."""
        batch = self._prepare_input(preprocessed_images)
        if len(batch) == 0:
            return []

        probs = self._get_probabilities(batch)
        results = []

        for sample_probs in probs:
            results.append({Gender(self._map_class_index(i)): float(prob) for i, prob in enumerate(sample_probs)})

        return results

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
                "batch_size": self.batch_size,
            }
        return self._metadata
