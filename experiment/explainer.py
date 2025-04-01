import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from config import ExperimentsConfig
from datatypes import ArtifactSavingLevel, FeatureDetails
from masker import FeatureMasker
from utils import setup_logger


class VisualExplainer:

    def __init__(self, config: ExperimentsConfig, masker: FeatureMasker, log_path: str):
        self.config = config
        self.masker = masker
        self.logger = setup_logger(name="visual_explainer", log_path=log_path)

    def setup_heatmap_generator(
        self,
        model: tf.keras.Model,
    ) -> GradcamPlusPlus:
        replace_to_linear = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
        return GradcamPlusPlus(model, model_modifier=replace_to_linear, clone=True)

    def _calculate_heatmap(
        self,
        heatmap_generator: GradcamPlusPlus,
        image_np: np.ndarray,
        true_label_index: int,
    ) -> np.ndarray:
        target_class = lambda output: output[0][true_label_index]
        image_batch = np.expand_dims(image_np.astype(np.float32), axis=0)

        target_layer = "block3_conv3"

        model_layer_names = [layer.name for layer in heatmap_generator.model.layers]
        if target_layer not in model_layer_names:
            self.logger.error(f"Target layer '{target_layer}' not found in model layers: {model_layer_names}")
            return np.zeros(image_np.shape[:2], dtype=np.float32)

        heatmap = heatmap_generator(target_class, image_batch, penultimate_layer=target_layer)[0]

        min_val, max_val = np.min(heatmap), np.max(heatmap)
        normalized_heatmap = (heatmap - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(heatmap)
        return normalized_heatmap

    def _calculate_single_feature_attention(
        self,
        feature: FeatureDetails,
        heatmap: np.ndarray,
    ) -> float:
        heatmap_height, heatmap_width = heatmap.shape[:2]
        min_y, max_y = max(0, feature.min_y), min(heatmap_height, feature.max_y)
        min_x, max_x = max(0, feature.min_x), min(heatmap_width, feature.max_x)

        if min_y >= max_y or min_x >= max_x:
            return 0.0

        feature_attention_region = heatmap[min_y:max_y, min_x:max_x]
        return np.mean(feature_attention_region) if feature_attention_region.size > 0 else 0.0

    def _save_heatmap(
        self,
        heatmap: np.ndarray,
        image_id: str,
        heatmap_dir: str,
    ) -> str:
        if self.config.output.artifact_level != ArtifactSavingLevel.FULL:
            return ""

        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

        heatmap_filename = f"{image_id}.npy"
        heatmap_full_path = os.path.join(heatmap_dir, heatmap_filename)
        np.save(heatmap_full_path, heatmap.astype(np.float16))
        heatmap_rel_path = os.path.relpath(heatmap_full_path, self.config.output.base_dir)
        return heatmap_rel_path

    def _compute_feature_details(
        self,
        features: List[FeatureDetails],
        heatmap: np.ndarray,
        image_id: str,
    ) -> List[FeatureDetails]:
        if not features:
            self.logger.warning(f"[{image_id}] No features detected by masker for attention calculation.")
            return []

        for feature_detail in features:
            attention_score = self._calculate_single_feature_attention(feature_detail, heatmap)
            is_key = attention_score >= self.config.core.key_feature_threshold

            feature_detail.attention_score = float(attention_score)
            feature_detail.is_key_feature = bool(is_key)

            if is_key:
                self.logger.debug(f"[{image_id}] Feature '{feature_detail.feature.name}' identified as key (attention: {attention_score:.4f}).")

        return features

    def generate_explanation_for_image(
        self,
        heatmap_generator: GradcamPlusPlus,
        image_np: np.ndarray,
        label: int,
        image_id: str,
        heatmap_dir: str,
    ) -> Tuple[Optional[str], List[FeatureDetails]]:

        heatmap = self._calculate_heatmap(heatmap_generator, image_np, label)
        heatmap_path = self._save_heatmap(heatmap, image_id, heatmap_dir)
        detected_features = self.masker.get_features(image_np, image_id)
        feature_details = self._compute_feature_details(detected_features, heatmap, image_id)

        return heatmap_path, feature_details
