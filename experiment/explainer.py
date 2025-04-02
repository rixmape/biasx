import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from config import Config
from datatypes import OutputLevel, FeatureDetails
from masker import FeatureMasker


class VisualExplainer:
    """Generates visual explanations (heatmaps) and calculates feature attention."""

    def __init__(self, config: Config, logger: logging.Logger, masker: FeatureMasker):
        """Initializes the VisualExplainer with configuration, masker, and logger."""
        self.config = config
        self.logger = logger
        self.masker = masker
        self.logger.info("Completed visual explainer initialization")

    def _calculate_heatmap(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        true_label_index: int,
        image_id: str,
    ) -> np.ndarray:
        """Calculates a normalized heatmap for a given image and label."""
        target_class = lambda output: output[0][true_label_index]
        image_batch = np.expand_dims(image_np.astype(np.float32), axis=0)
        target_layer = "block3_conv3"

        model_layers = [layer.name for layer in model.layers]
        if target_layer not in model_layers:
            self.logger.error(f"[{image_id}] Target layer '{target_layer}' not found: model_layers={model_layers}")
            return np.zeros(image_np.shape[:2], dtype=np.float32)

        try:
            heatmap = heatmap_generator(target_class, image_batch, penultimate_layer=target_layer)[0]
        except Exception as e:
            self.logger.error(f"[{image_id}] Heatmap generation via GradCAM++ failed: {e}", exc_info=True)
            return np.zeros(image_np.shape[:2], dtype=np.float32)

        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val <= min_val:
            self.logger.warning(f"[{image_id}] Heatmap range invalid (max <= min), returning zeros.")
            return np.zeros_like(heatmap, dtype=np.float32)

        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
        return normalized_heatmap.astype(np.float32)

    def _calculate_single_feature_attention(
        self,
        feature: FeatureDetails,
        heatmap: np.ndarray,
        image_id: str,
    ) -> float:
        """Calculates the mean attention score within a feature's bounding box."""
        heatmap_height, heatmap_width = heatmap.shape[:2]

        min_y, max_y = max(0, feature.bbox.min_y), min(heatmap_height, feature.bbox.max_y)
        min_x, max_x = max(0, feature.bbox.min_x), min(heatmap_width, feature.bbox.max_x)

        if min_y >= max_y or min_x >= max_x:
            self.logger.debug(f"[{image_id}] Invalid/Empty attention region after clamping for feature {feature.feature.name}: box=({min_x}, {min_y}, {max_x}, {max_y})")
            return 0.0

        feature_attention_region = heatmap[min_y:max_y, min_x:max_x]

        if feature_attention_region.size == 0:
            self.logger.debug(f"[{image_id}] Feature attention region is empty for feature {feature.feature.name}: box=({min_x}, {min_y}, {max_x}, {max_y})")
            return 0.0

        return float(np.mean(feature_attention_region))

    def _save_heatmap(
        self,
        heatmap: np.ndarray,
        image_id: str,
    ) -> Optional[str]:
        """Saves the heatmap array to disk if artifact saving is enabled."""
        if self.config.output.level != OutputLevel.FULL:
            self.logger.debug(f"[{image_id}] Skipping heatmap saving: artifact level is {self.config.output.level.name}")
            return None

        path = os.path.join(self.config.output.base_path, self.config.experiment_id, "heatmaps")
        os.makedirs(path, exist_ok=True)

        filename = f"{image_id}.npy"
        filepath = os.path.join(path, filename)

        try:
            np.save(filepath, heatmap.astype(np.float16))
            heatmap_rel_path = os.path.relpath(filepath, self.config.output.base_path)
            return heatmap_rel_path
        except Exception as e:
            self.logger.error(f"[{image_id}] Failed to save heatmap to {filepath}: {e}", exc_info=True)
            return None

    def _compute_feature_details(
        self,
        features: List[FeatureDetails],
        heatmap: np.ndarray,
        image_id: str,
    ) -> List[FeatureDetails]:
        """Computes and adds attention scores and key feature flags to feature details."""
        if not features:
            self.logger.warning(f"[{image_id}] No features provided for attention calculation.")
            return []

        if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
            self.logger.warning(f"[{image_id}] Cannot compute feature attention: heatmap is invalid or all zeros. Setting scores to 0.")
            for feature_detail in features:
                feature_detail.attention_score = 0.0
                feature_detail.is_key_feature = False
            return features

        key_features_found = 0
        for feature_detail in features:
            attention_score = self._calculate_single_feature_attention(feature_detail, heatmap, image_id)
            is_key = attention_score >= self.config.core.key_feature_threshold

            feature_detail.attention_score = float(attention_score)
            feature_detail.is_key_feature = bool(is_key)

            if is_key:
                key_features_found += 1

        if key_features_found == 0:
            self.logger.info(f"[{image_id}] No key features above threshold {self.config.core.key_feature_threshold}.")

        return features

    def get_heatmap_generator(
        self,
        model: tf.keras.Model,
    ) -> GradcamPlusPlus:
        """Creates and returns a GradCAM++ heatmap generator for the given model."""
        try:
            replace_to_linear = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
            generator = GradcamPlusPlus(model, model_modifier=replace_to_linear, clone=True)
            self.logger.debug("Successfully created GradCAM++ heatmap generator.")
            return generator
        except Exception as e:
            self.logger.error(f"Failed to create GradCAM++ generator: {e}", exc_info=True)
            raise

    def generate_explanation(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        label: int,
        image_id: str,
    ) -> Tuple[List[FeatureDetails], Optional[str]]:
        """Generates feature details with attention and saves heatmap for a single image."""
        heatmap = self._calculate_heatmap(heatmap_generator, model, image_np, label, image_id)
        heatmap_path = self._save_heatmap(heatmap, image_id)

        detected_features = self.masker.get_features(image_np, image_id)
        if not detected_features:
            self.logger.warning(f"[{image_id}] No features detected by masker; returning empty feature list for explanation.")
            return [], heatmap_path

        feature_details_with_attention = self._compute_feature_details(detected_features, heatmap, image_id)

        return feature_details_with_attention, heatmap_path
