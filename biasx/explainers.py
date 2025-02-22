import json
from typing import Literal, Optional

import keras
import mediapipe
import numpy as np
import skimage.filters
import tensorflow as tf
import tf_keras_vis
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from skimage.measure import label, regionprops
import tf_keras_vis.gradcam
import tf_keras_vis.gradcam_plus_plus
import tf_keras_vis.scorecam

from .types import Box


class VisualExplainer:
    """Generates and processes visual explanations for model decisions."""

    CAM_METHODS = {
        "gradcam": tf_keras_vis.gradcam.Gradcam,
        "gradcam++": tf_keras_vis.gradcam_plus_plus.GradcamPlusPlus,
        "scorecam": tf_keras_vis.scorecam.Scorecam,
    }

    THRESHOLD_METHODS = {
        "otsu": skimage.filters.threshold_otsu,
        "niblack": skimage.filters.threshold_niblack,
        "sauvola": skimage.filters.threshold_sauvola,
    }

    def __init__(
        self,
        landmark_model_path: str,
        landmark_map_path: str,
        cam_method: Literal["gradcam", "gradcam++", "scorecam"] = "gradcam++",
        cutoff_percentile: Optional[int] = 90,
        threshold_method: Optional[Literal["otsu", "niblack", "sauvola"]] = "otsu",
    ):
        """Initialize the visual explanation generator."""
        self.detector = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=landmark_model_path), num_faces=1))
        self.landmark_map = self._load_json(landmark_map_path)
        self.cam_method = self.CAM_METHODS[cam_method]
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = self.THRESHOLD_METHODS[threshold_method]

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _modify_model(m: tf.keras.Model) -> None:
        m.layers[-1].activation = tf.keras.activations.linear

    @staticmethod
    def _score_function(output: tf.Tensor, target_class: int) -> tf.Tensor:
        return output[0][target_class]

    def generate_heatmap(self, model: keras.Model, image: np.ndarray, target_class: int) -> np.ndarray:
        """Generate class activation map for an image."""
        visualizer = self.cam_method(model, model_modifier=self._modify_model, clone=True)
        image = np.expand_dims(image, axis=-1) if image.ndim == 2 else image  # Ensure three-channel format
        return visualizer(lambda output: self._score_function(output, target_class), image, penultimate_layer=-1)[0]

    def process_heatmap(self, heatmap: np.ndarray) -> list[Box]:
        """Process heatmap into activation boxes."""
        heatmap[heatmap < np.percentile(heatmap, self.cutoff_percentile)] = 0
        binary = heatmap > self.threshold_method(heatmap)

        return [Box(min_col, min_row, max_col, max_row) for region in regionprops(label(binary)) for min_row, min_col, max_row, max_col in [region.bbox]]

    def detect_landmarks(self, image_path: str, image_size: tuple[int, int]) -> list[Box]:
        """Detect facial landmarks in an image."""
        result = self.detector.detect(mediapipe.Image.create_from_file(image_path))
        if not result.face_landmarks:
            return []

        points = [(int(round(point.x * image_size[1])), int(round(point.y * image_size[0]))) for point in result.face_landmarks[0]]

        return [
            Box(min_x=min(x for x, _ in feature_points), min_y=min(y for _, y in feature_points), max_x=max(x for x, _ in feature_points), max_y=max(y for _, y in feature_points), feature=feature)
            for feature, indices in self.landmark_map.items()
            for feature_points in [[points[i] for i in indices]]
        ]

    def match_landmarks(self, activation_boxes: list[Box], landmark_boxes: list[Box]) -> list[Box]:
        """Match activation boxes with nearest landmarks."""
        if not activation_boxes or not landmark_boxes:
            return activation_boxes

        matched_boxes = []
        for a_box in activation_boxes:
            nearest = min(landmark_boxes, key=lambda l: (l.center[0] - a_box.center[0]) ** 2 + (l.center[1] - a_box.center[1]) ** 2)
            if (max(0, min(a_box.max_x, nearest.max_x) - max(a_box.min_x, nearest.min_x)) * max(0, min(a_box.max_y, nearest.max_y) - max(a_box.min_y, nearest.min_y))) / a_box.area >= 0.2:
                matched_boxes.append(Box(a_box.min_x, a_box.min_y, a_box.max_x, a_box.max_y, feature=nearest.feature))
            else:
                matched_boxes.append(a_box)
        return matched_boxes
