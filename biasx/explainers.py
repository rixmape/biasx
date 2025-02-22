from typing import Literal, Optional

import keras
import mediapipe
import numpy as np
import skimage.filters
import tensorflow as tf
import tf_keras_vis
import tf_keras_vis.gradcam
import tf_keras_vis.gradcam_plus_plus
import tf_keras_vis.scorecam
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from skimage.measure import label, regionprops

from .types import Box


class LandmarkMapping:
    """Maps facial features to their corresponding landmark indices."""

    def __init__(self):
        """Initialize the facial feature to landmark index mapping."""
        # fmt: off
        self.mapping = {
            "left_eye": [249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466],
            "right_eye": [7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246],
            "nose": [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440],
            "lips": [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
            "left_cheek": [454, 447, 345, 346, 347, 330, 425, 427, 434, 416, 435, 288, 361, 323, 280, 352, 366, 411, 376, 401, 433],
            "right_cheek": [234, 227, 116,117,118, 101, 205, 207, 214, 192, 215, 58, 132, 93, 127, 50, 123, 137, 177, 147, 187, 213],
            "left_eyebrow": [276, 282, 283, 285, 293, 295, 296, 300, 334, 336],
            "right_eyebrow": [46, 52, 53, 55, 63, 65, 66, 70, 105, 107],
        }
        # fmt: on

    def get_indices(self, feature: str) -> list[int]:
        """Get landmark indices for a given facial feature."""
        return self.mapping.get(feature, [])

    def get_features(self) -> list[str]:
        """Get list of all facial features."""
        return list(self.mapping.keys())


class FacialLandmarker:
    """Handles facial landmark detection using MediaPipe."""

    DEFAULT_MODEL_PATH = "biasx/models/mediapipe_landmarker.task"

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """Initialize the facial landmark detector."""
        self.detector = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=1))
        self.mapping = LandmarkMapping()

    def detect(self, image_path: str, image_size: tuple[int, int]) -> list[Box]:
        """Detect facial landmarks in an image."""
        result = self.detector.detect(mediapipe.Image.create_from_file(image_path))
        if not result.face_landmarks:
            return []

        points = [(int(round(point.x * image_size[1])), int(round(point.y * image_size[0]))) for point in result.face_landmarks[0]]

        return [
            Box(
                min_x=min(x for x, _ in feature_points),
                min_y=min(y for _, y in feature_points),
                max_x=max(x for x, _ in feature_points),
                max_y=max(y for _, y in feature_points),
                feature=feature,
            )
            for feature, indices in self.mapping.mapping.items()
            for feature_points in [[points[i] for i in indices]]
        ]


class ClassActivationMapper:
    """Handles generation and processing of class activation maps."""

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
        cam_method: Literal["gradcam", "gradcam++", "scorecam"] = "gradcam++",
        cutoff_percentile: int = 90,
        threshold_method: Literal["otsu", "niblack", "sauvola"] = "otsu",
    ):
        """Initialize the activation map generator."""
        self.cam_method = self.CAM_METHODS[cam_method]
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = self.THRESHOLD_METHODS[threshold_method]

    @staticmethod
    def _modify_model(m: tf.keras.Model) -> None:
        """Modify model for activation map generation."""
        m.layers[-1].activation = tf.keras.activations.linear

    @staticmethod
    def _score_function(output: tf.Tensor, target_class: int) -> tf.Tensor:
        """Score function for class activation mapping."""
        return output[0][target_class]

    def generate_heatmap(self, model: keras.Model, image: np.ndarray, target_class: int) -> np.ndarray:
        """Generate class activation map for an image."""
        visualizer = self.cam_method(model, model_modifier=self._modify_model, clone=True)
        image = np.expand_dims(image, axis=-1) if image.ndim == 2 else image  # Always set to three channels
        return visualizer(lambda output: self._score_function(output, target_class), image, penultimate_layer=-1)[0]

    def process_heatmap(self, heatmap: np.ndarray) -> list[Box]:
        """Process heatmap into activation boxes."""
        heatmap[heatmap < np.percentile(heatmap, self.cutoff_percentile)] = 0
        binary = heatmap > self.threshold_method(heatmap)
        return [Box(min_col, min_row, max_col, max_row) for region in regionprops(label(binary)) for min_row, min_col, max_row, max_col in [region.bbox]]


class VisualExplainer:
    """Generates and processes visual explanations for model decisions."""

    def __init__(
        self,
        landmarker: Optional[FacialLandmarker] = None,
        activation_mapper: Optional[ClassActivationMapper] = None,
        overlap_threshold: float = 0.2,
    ):
        """Initialize the visual explanation generator."""
        self.landmarker = landmarker or FacialLandmarker()
        self.activation_mapper = activation_mapper or ClassActivationMapper()
        self.overlap_threshold = overlap_threshold

    def get_features(self) -> list[str]:
        """Get list of supported facial features."""
        return self.landmarker.mapping.get_features()

    def generate_heatmap(self, model: keras.Model, image: np.ndarray, target_class: int) -> np.ndarray:
        """Generate class activation map for an image."""
        return self.activation_mapper.generate_heatmap(model, image, target_class)

    def process_heatmap(self, heatmap: np.ndarray) -> list[Box]:
        """Process heatmap into activation boxes."""
        return self.activation_mapper.process_heatmap(heatmap)

    def detect_landmarks(self, image_path: str, image_size: tuple[int, int]) -> list[Box]:
        """Detect facial landmarks in an image."""
        return self.landmarker.detect(image_path, image_size)

    def match_landmarks(self, activation_boxes: list[Box], landmark_boxes: list[Box]) -> list[Box]:
        """Match activation boxes with nearest landmarks."""
        if not activation_boxes or not landmark_boxes:
            return activation_boxes

        matched_boxes = []
        for a_box in activation_boxes:
            nearest = min(landmark_boxes, key=lambda l: (l.center[0] - a_box.center[0]) ** 2 + (l.center[1] - a_box.center[1]) ** 2)
            overlap_area = max(0, min(a_box.max_x, nearest.max_x) - max(a_box.min_x, nearest.min_x)) * max(0, min(a_box.max_y, nearest.max_y) - max(a_box.min_y, nearest.min_y))
            if overlap_area / a_box.area >= self.overlap_threshold:
                matched_boxes.append(Box(a_box.min_x, a_box.min_y, a_box.max_x, a_box.max_y, feature=nearest.feature))
            else:
                matched_boxes.append(a_box)
        return matched_boxes
