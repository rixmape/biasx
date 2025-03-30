import gc
from typing import Any

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from config import Config
from masker import FeatureMasker
from datatypes import FacialFeature
from utils import setup_logger


class VisualExplainer:
    """Class that generates visual explanations by computing Grad-CAM++ heatmaps and extracting feature attention from masked image regions."""

    def __init__(self, config: Config, masker: FeatureMasker, log_path: str):
        self.config = config
        self.logger = setup_logger(name="visual_explainer", log_path=log_path)
        self.masker = masker

    def _get_heatmap(self, visualizer: Any, image: np.ndarray, true_label: int) -> np.ndarray:
        """Generates a Grad-CAM++ heatmap for an image based on its true label using the provided visualizer function."""
        score_fn = lambda output: output[0][true_label]
        expanded_img = np.expand_dims(image, axis=0)
        heatmap = visualizer(score_fn, expanded_img, penultimate_layer="block3_conv3")[0]

        if np.mean(heatmap) < 0.5:
            quantiles = ", ".join([f"q{i}={np.quantile(heatmap, val):.4f}" for i, val in enumerate([0.25, 0.5, 0.75])])
            self.logger.warning(f"CAM too low: mean={np.mean(heatmap):.4f}, std={np.std(heatmap):.4f}, {quantiles}")

        return heatmap

    def _select_key_features(self, features: list[FacialFeature], heatmap: np.ndarray) -> list[FacialFeature]:
        """Filters feature boxes based on their computed attention from the heatmap, retaining only those above a specified threshold."""

        key_features = []
        for feature in features:
            roi = heatmap[
                max(0, feature.min_y) : min(heatmap.shape[0], feature.max_y),
                max(0, feature.min_x) : min(heatmap.shape[1], feature.max_x),
            ]
            attention = np.mean(roi)

            if attention < self.config.feature_attention_threshold:
                quantiles = ", ".join([f"q{i}={np.quantile(roi, val):.4f}" for i, val in enumerate([0.25, 0.5, 0.75])])
                self.logger.warning(f"Feature '{feature.name}' ignored: attention={attention:.4f}, std={np.std(roi):.4f}, {quantiles}")
                continue

            key_feature = FacialFeature(min_x=feature.min_x, min_y=feature.min_y, max_x=feature.max_x, max_y=feature.max_y, name=feature.name, attention=attention)
            key_features.append(key_feature)

        if not key_features:
            self.logger.warning("No features exceeded the attention threshold")

        return key_features

    def explain(self, model: tf.keras.Model, test_data: tf.data.Dataset) -> list[list[FacialFeature]]:
        """Iterates over test dataset images to generate heatmaps, extract feature boxes, and compile key features for explanation."""
        self.logger.info("Starting visual explanation generation")

        modifier = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
        visualizer = GradcamPlusPlus(model, model_modifier=modifier)

        key_features = []
        batch_count = 0
        image_count = 0
        empty_feature_count = 0

        self.logger.info(f"Processing batches for feature attention with {self.config.feature_attention_threshold:.4f} threshold")

        for batch in test_data:
            batch_count += 1
            images, labels = batch
            batch_size = len(images)

            self.logger.debug(f"Processing batch {batch_count} with {batch_size} images")

            for i in range(batch_size):
                image = images[i].numpy()
                label = int(labels[i].numpy())
                image_count += 1

                heatmap = self._get_heatmap(visualizer, image, label)
                boxes = self.masker.get_features(image)

                if not boxes:
                    self.logger.warning(f"No facial features detected in image {image_count}. This may indicate an issue with face detection.")
                    empty_feature_count += 1

                filtered_boxes = self._select_key_features(boxes, heatmap)
                key_features.append(filtered_boxes)

            if batch_count % 5 == 0:
                self.logger.info(f"Processed {image_count} images across {batch_count} batches")

            tf.keras.backend.clear_session()
            gc.collect()

        self.logger.info(f"Processed {image_count} images across {batch_count} batches")

        return key_features
