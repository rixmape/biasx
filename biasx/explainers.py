import json

import keras
import mediapipe
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from tf_keras_vis.gradcam import GradcamPlusPlus

from .types import Box


class VisualExplainer:
    """Generates and processes visual explanations for model decisions."""

    def __init__(self, landmark_model_path: str, landmark_map_path: str, image_size: tuple[int, int]):
        """
        Initialize the visual explanation generator.

        Args:
            landmark_model_path: Path to facial landmark detection model
            landmark_map_path: Path to landmark feature mapping file
            image_size: Size of input images (height, width)
        """
        base_options = BaseOptions(model_asset_path=landmark_model_path)
        options = FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self.detector = FaceLandmarker.create_from_options(options)

        with open(landmark_map_path, "r") as f:
            self.landmark_map = json.load(f)

        self.image_size = image_size

    def generate_heatmap(self, model: keras.Model, image: np.ndarray, target_class: int) -> np.ndarray:
        """Generate class activation map for an image."""

        def model_modifier(m: tf.keras.Model) -> None:
            m.layers[-1].activation = tf.keras.activations.linear

        def score_function(output: tf.Tensor) -> tf.Tensor:
            return output[0][target_class]

        visualizer = GradcamPlusPlus(model, model_modifier=model_modifier, clone=True)
        return visualizer(score_function, image[..., np.newaxis], penultimate_layer=-1)[0]

    def process_heatmap(self, heatmap: np.ndarray) -> list[Box]:
        """Process heatmap into activation boxes."""
        # Binarize heatmap
        heatmap[heatmap < np.percentile(heatmap, 90)] = 0
        binary = heatmap > threshold_otsu(heatmap)

        # Extract regions
        boxes = []
        for region in regionprops(label(binary)):
            min_row, min_col, max_row, max_col = region.bbox
            boxes.append(Box(min_col, min_row, max_col, max_row))
        return boxes

    def detect_landmarks(self, image_path: str) -> list[Box]:
        """Detect facial landmarks in an image."""
        image = mediapipe.Image.create_from_file(image_path)
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return []

        landmarks = result.face_landmarks[0]
        points = [(int(round(point.x * self.image_size[1])), int(round(point.y * self.image_size[0]))) for point in landmarks]

        boxes = []
        for feature, indices in self.landmark_map.items():
            feature_points = [points[i] for i in indices]
            boxes.append(
                Box(
                    min_x=min(x for x, _ in feature_points),
                    min_y=min(y for _, y in feature_points),
                    max_x=max(x for x, _ in feature_points),
                    max_y=max(y for _, y in feature_points),
                    feature=feature,
                )
            )
        return boxes

    def match_landmarks(self, activation_boxes: list[Box], landmark_boxes: list[Box]) -> list[Box]:
        """Match activation boxes with nearest landmarks."""
        if not (activation_boxes and landmark_boxes):
            return []

        result = []
        for a_box in activation_boxes:
            nearest = min(landmark_boxes, key=lambda l: ((l.center[0] - a_box.center[0]) ** 2 + (l.center[1] - a_box.center[1]) ** 2))

            x_overlap = max(0, min(a_box.max_x, nearest.max_x) - max(a_box.min_x, nearest.min_x))
            y_overlap = max(0, min(a_box.max_y, nearest.max_y) - max(a_box.min_y, nearest.min_y))

            if (x_overlap * y_overlap) / a_box.area >= 0.2:
                result.append(Box(min_x=a_box.min_x, min_y=a_box.min_y, max_x=a_box.max_x, max_y=a_box.max_y, feature=nearest.feature))
            else:
                result.append(a_box)  # No match, keep original box
        return result
