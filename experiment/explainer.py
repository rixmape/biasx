import gc
from typing import Any

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from config import Config
from masker import FeatureMasker
from datatypes import FeatureBox, Gender
from utils import setup_logger

logger = setup_logger(name="experiment.explainer")


class VisualExplainer:
    """Class that generates visual explanations by computing Grad-CAM heatmaps and extracting feature importance from masked image regions."""

    def __init__(self, config: Config, masker: FeatureMasker):
        self.config = config
        self.masker = masker
        logger.info(f"Initializing VisualExplainer with feature attention threshold: {config.feature_attention_threshold}")
        logger.debug(f"Configuration: mask_padding={config.mask_padding}, image_size={config.image_size}x{config.image_size}")

    def _get_heatmap(self, visualizer: Any, image: np.ndarray, true_label: int) -> np.ndarray:
        """Generates a Grad-CAM heatmap for an image based on its true label using the provided visualizer function."""
        logger.debug(f"Generating Grad-CAM heatmap for image with true label: {Gender(true_label).name}")
        score_fn = lambda output: output[0][true_label]
        expanded_img = np.expand_dims(image, axis=0)
        heatmap = visualizer(score_fn, expanded_img, penultimate_layer="block3_conv3")[0]

        logger.debug(f"Heatmap generated with shape: {heatmap.shape}, intensity range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        return heatmap

    def _compute_feature_importance(self, box: FeatureBox, heatmap: np.ndarray) -> float:
        """Computes the average intensity within a feature box region of the heatmap to determine its importance."""
        roi = heatmap[max(0, box.min_y) : min(heatmap.shape[0], box.max_y), max(0, box.min_x) : min(heatmap.shape[1], box.max_x)]
        return float(np.mean(roi))

    def _filter_feature_boxes(self, boxes: list[FeatureBox], heatmap: np.ndarray) -> list[FeatureBox]:
        """Filters and orders feature boxes based on whether their computed importance exceeds the configured threshold."""

        filtered = []
        for b in boxes:
            if b.get_area() == 0:
                logger.warning(f"Feature '{b.name}' has zero area and will be ignored")
                continue

            imp = self._compute_feature_importance(b, heatmap)
            copied_box = FeatureBox(min_x=b.min_x, min_y=b.min_y, max_x=b.max_x, max_y=b.max_y, name=b.name, importance=imp)

            if copied_box.importance > self.config.feature_attention_threshold:
                filtered.append(copied_box)
                logger.debug(f"Feature '{b.name}' selected: importance {imp:.4f} > threshold {self.config.feature_attention_threshold}")
            else:
                logger.debug(f"Feature '{b.name}' filtered out: importance {imp:.4f} <= threshold {self.config.feature_attention_threshold}")

        filtered_boxes = sorted(filtered, key=lambda x: x.importance, reverse=True)
        logger.debug(f"Kept {len(filtered_boxes)}/{len(boxes)} features after importance filtering")

        if not filtered_boxes:
            logger.warning("No features exceeded the importance threshold. Consider lowering the threshold value.")

        return filtered_boxes

    def explain(self, model: tf.keras.Model, test_dataset: tf.data.Dataset) -> list[list[FeatureBox]]:
        """Iterates over test dataset images to generate heatmaps, extract feature boxes, and compile key features for explanation."""
        logger.info("Starting visual explanation generation")

        logger.debug("Modifying model output layer for Grad-CAM visualization")
        modifier = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
        visualizer = GradcamPlusPlus(model, model_modifier=modifier)

        key_features = []
        batch_count = 0
        image_count = 0
        empty_feature_count = 0

        logger.info("Processing test batches for feature importance")
        for batch in test_dataset:
            batch_count += 1
            images, labels = batch
            batch_size = len(images)

            logger.debug(f"Processing batch {batch_count} with {batch_size} images")

            for i in range(batch_size):
                img = images[i].numpy()
                label = int(labels[i].numpy())
                image_count += 1

                logger.debug(f"Analyzing image {image_count} with gender label: {Gender(label).name}")

                heatmap = self._get_heatmap(visualizer, img, label)
                boxes = self.masker.get_feature_boxes(img)

                if not boxes:
                    logger.warning(f"No facial features detected in image {image_count}. This may indicate an issue with face detection.")
                    empty_feature_count += 1

                logger.debug(f"Filtering {len(boxes)} feature boxes with importance threshold: {self.config.feature_attention_threshold}")
                filtered_boxes = self._filter_feature_boxes(boxes, heatmap)
                key_features.append(filtered_boxes)

                if i % 10 == 0:
                    logger.debug(f"Processed {i}/{batch_size} images in current batch")

            logger.debug(f"Completed batch {batch_count}")

            if batch_count % 5 == 0:
                logger.info(f"Visual explanation progress: {image_count} images processed so far")

            tf.keras.backend.clear_session()
            gc.collect()

        logger.info(f"Visual explanation complete: analyzed {image_count} images across {batch_count} batches")

        if empty_feature_count > 0:
            logger.warning(f"{empty_feature_count}/{image_count} images ({empty_feature_count/image_count:.1%}) had no detected facial features")

        feature_counts = [len(features) for features in key_features]
        avg_features = sum(feature_counts) / max(len(feature_counts), 1)
        logger.info(f"Average number of important features identified per image: {avg_features:.2f}")

        empty_importance_count = sum(1 for features in key_features if not features)
        if empty_importance_count > 0:
            logger.warning(f"{empty_importance_count}/{image_count} images ({empty_importance_count/image_count:.1%}) had no features exceeding the importance threshold. Consider lowering the threshold value.")

        return key_features
