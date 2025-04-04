"""Model handling module for BiasX."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from .config import configurable
from .types import Gender


@configurable("model")
class Model:
    """Handles loading and inference for facial classification models.

    This class loads a Keras model from a specified path, handles batching
    and preprocessing of input images, performs inference, and processes
    the model's output probabilities into classified gender labels and
    confidence scores. It accounts for potentially inverted class labels.

    Attributes:
        model (tf.keras.Model): The loaded Keras model instance.
        inverted_classes (bool): If True, swaps the interpretation of the model's
            output classes (e.g., if the model outputs 0 for female and 1 for
            male, setting this to True maps 0 to MALE and 1 to FEMALE).
        batch_size (int): The batch size to use during model prediction
            (`model.predict`).
        _metadata (Any): Placeholder for potential future metadata storage.
                         Currently initialized to None and not used.
    """

    def __init__(self, path: str, inverted_classes: bool, batch_size: int, **kwargs):
        """Initialize the classification model handler.

        Loads the Keras model from the given file path and stores configuration
        parameters.

        Args:
            path (str): The file path to the saved Keras model (.h5 or SavedModel dir).
            inverted_classes (bool): Whether the model's output classes (0 and 1)
                should be interpreted in reverse order compared to the `Gender` enum
                (MALE=0, FEMALE=1).
            batch_size (int): The batch size to use when calling `model.predict`.
            **kwargs: Additional keyword arguments passed via configuration.
        """
        self.model = tf.keras.models.load_model(path)
        self.inverted_classes = inverted_classes
        self.batch_size = batch_size
        self._metadata = None

    def _prepare_input(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Prepare images for model input, ensuring correct batch format.

        Takes either a single preprocessed image, a list of images, or a batch
        of images (as NumPy arrays) and formats them into a single NumPy batch
        array suitable for `model.predict`. Handles adding batch and channel
        dimensions if necessary.

        Args:
            preprocessed_images (Union[np.ndarray, List[np.ndarray]]): A single
                image, list of images, or batch of images as NumPy arrays. Assumes
                images are already numerically preprocessed (e.g., normalized).

        Returns:
            A NumPy array representing the batch of images ready for model input,
            matching the model's expected input shape (e.g., NHWC or NCHW).
            Returns an empty array with the correct shape dimensions (excluding
            batch size 0) if the input is empty.
        """
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
        """Get raw probability distributions from model output for a batch.

        Performs inference using `self.model.predict` on the prepared batch.
        If the model output appears to be logits (more than one output unit),
        it applies a softmax function to convert them into probabilities.

        Args:
            batch (np.ndarray): A batch of preprocessed images ready for model input.

        Returns:
            A NumPy array where each row contains the probability distribution
            (e.g., [prob_class_0, prob_class_1]) for the corresponding input image.
            Returns an empty array of shape (0, num_classes) if the input batch is empty.
        """
        if len(batch) == 0:
            return np.empty((0, 2))

        output = self.model.predict(batch, verbose=0, batch_size=self.batch_size)
        if len(output.shape) > 1 and output.shape[1] > 1:
            return tf.nn.softmax(output).numpy()
        return output

    def _process_predictions(self, probs: np.ndarray) -> List[Tuple[Gender, float]]:
        """Process model output probabilities into gender predictions and confidences.

        Takes the raw probability outputs from the model, determines the predicted
        class index (highest probability), calculates the confidence score (the
        probability of the predicted class), and maps the predicted index to a
        `Gender` enum value, accounting for the `inverted_classes` setting.

        Args:
            probs (np.ndarray): A NumPy array of probability distributions, where
                each row corresponds to an image and columns correspond to classes
                (e.g., output from `_get_probabilities`).

        Returns:
            A list of tuples. Each tuple contains:
                - predicted_gender (biasx.types.Gender): The predicted gender enum.
                - confidence (float): The model's confidence in the prediction
                  (probability of the predicted class).
        """
        results = []
        for sample_probs in probs:
            pred_idx = int(np.argmax(sample_probs))
            confidence = float(sample_probs[pred_idx])
            pred_class = Gender(pred_idx if not self.inverted_classes else 1 - pred_idx)
            results.append((pred_class, confidence))
        return results

    def predict(self, preprocessed_images: Union[np.ndarray, List[np.ndarray]]) -> List[Tuple[Gender, float]]:
        """Make gender predictions from preprocessed images.

        This is the main public method for getting predictions. It orchestrates
        the process:

        1. Prepares the input images using `_prepare_input`.
        2. Gets raw probabilities from the model using `_get_probabilities`.
        3. Processes these probabilities into gender labels and confidences
           using `_process_predictions`.

        Args:
            preprocessed_images (Union[np.ndarray, List[np.ndarray]]): A single
                preprocessed image, a list of images, or a batch of images
                (NumPy arrays).

        Returns:
            A list of tuples, one for each input image. Each tuple contains:
                - predicted_gender (biasx.types.Gender): The predicted gender.
                - confidence (float): The prediction confidence.
            Returns an empty list if the input is empty.
        """
        batch = self._prepare_input(preprocessed_images)
        if len(batch) == 0:
            return []

        probs = self._get_probabilities(batch)
        return self._process_predictions(probs)
