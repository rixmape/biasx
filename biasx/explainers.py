import json
import os

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
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from .models import ClassificationModel
from .types import Box, CAMMethod, DistanceMetric, Gender, ThresholdMethod


class LandmarkMapping:
    """Maps facial features to their corresponding landmark indices."""

    def __init__(self):
        """Initialize the facial feature to landmark index mapping by loading from JSON."""
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> dict:
        """Load the landmark mapping from a JSON file."""
        mapping_path = self._get_mapping_path()
        with open(mapping_path, "r") as f:
            return json.load(f)

    def _get_mapping_path(self) -> str:
        """Get the absolute path to the mapping file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        mapping_path = os.path.join(data_dir, "landmark_mapping.json")
        abs_mapping_path = os.path.abspath(mapping_path)

        if not os.path.exists(abs_mapping_path):
            raise FileNotFoundError(f"Landmark mapping file not found at {abs_mapping_path}")

        return abs_mapping_path

    def get_indices(self, feature: str) -> list[int]:
        """Get landmark indices for a given facial feature."""
        return self.mapping.get(feature, [])

    def get_features(self) -> list[str]:
        """Get list of all facial features."""
        return list(self.mapping.keys())


class FacialLandmarker:
    """Handles facial landmark detection using MediaPipe."""

    def __init__(self, max_faces: int):
        """Initialize the facial landmark detector."""
        model_path = self._get_model_path()
        self.options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=max_faces)
        self.detector = FaceLandmarker.create_from_options(self.options)
        self.mapping = LandmarkMapping()

    def _get_model_path(self):
        """Get the absolute path to the model file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "mediapipe_landmarker.task")
        abs_model_path = os.path.abspath(model_path)

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"MediaPipe model not found at {abs_model_path}")

        return abs_model_path

    def detect(self, image_path: str, image_width: int, image_height: int) -> list[Box]:
        """Detect facial landmarks in an image."""
        result = self.detector.detect(mediapipe.Image.create_from_file(image_path))
        if not result.face_landmarks:
            return []

        points = [(int(round(point.x * image_width)), int(round(point.y * image_height))) for point in result.face_landmarks[0]]

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
        "isodata": skimage.filters.threshold_isodata,
        "li": skimage.filters.threshold_li,
        "local": skimage.filters.threshold_local,
        "mean": skimage.filters.threshold_mean,
        "minimum": skimage.filters.threshold_minimum,
        "multiotsu": skimage.filters.threshold_multiotsu,
        "niblack": skimage.filters.threshold_niblack,
        "otsu": skimage.filters.threshold_otsu,
        "sauvola": skimage.filters.threshold_sauvola,
        "triangle": skimage.filters.threshold_triangle,
        "yen": skimage.filters.threshold_yen,
    }

    def __init__(self, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod):
        """Initialize the activation map generator."""
        self.cam_method = self.CAM_METHODS[cam_method]
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = self.THRESHOLD_METHODS[threshold_method]

    @staticmethod
    def _modify_model(m: tf.keras.Model) -> None:
        """Modify model for activation map generation."""
        m.layers[-1].activation = tf.keras.activations.linear

    @staticmethod
    def _score_function(output: tf.Tensor, target_class: Gender) -> tf.Tensor:
        """Score function for class activation mapping."""
        return output[0][target_class]

    def generate_heatmap(self, model: keras.Model, image: np.ndarray, target_class: Gender) -> np.ndarray:
        """Generate class activation map for an image."""
        visualizer = self.cam_method(model, model_modifier=self._modify_model, clone=True)
        image = np.expand_dims(image, axis=-1) if image.ndim == 2 else image  # Always set to three channels
        return visualizer(lambda output: self._score_function(output, target_class), image, penultimate_layer=-1)[0]

    def process_heatmap(self, heatmap: np.ndarray) -> list[Box]:
        """Process heatmap into activation boxes."""
        heatmap_copy = heatmap.copy()
        heatmap_copy[heatmap_copy < np.percentile(heatmap_copy, self.cutoff_percentile)] = 0
        binary = heatmap_copy > self.threshold_method(heatmap_copy)
        return [Box(min_col, min_row, max_col, max_row) for region in regionprops(label(binary)) for min_row, min_col, max_row, max_col in [region.bbox]]


class VisualExplainer:
    """Handles image explanation generation using activation maps and facial landmarks."""

    def __init__(
        self,
        max_faces: int,
        cam_method: CAMMethod,
        cutoff_percentile: int,
        threshold_method: ThresholdMethod,
        overlap_threshold: float,
        distance_metric: DistanceMetric,
        activation_maps_path: str,
    ):
        self.landmarker = FacialLandmarker(max_faces=max_faces)
        self.activation_mapper = ClassActivationMapper(
            cam_method=cam_method,
            cutoff_percentile=cutoff_percentile,
            threshold_method=threshold_method,
        )
        self.overlap_threshold = overlap_threshold
        self.distance_metric = distance_metric
        self.activation_maps_path = activation_maps_path

    def _save_activation_map(self, activation_map: np.ndarray, image_path: str) -> str:
        """Generate path for saving activation map."""
        basename = os.path.splitext(os.path.basename(image_path))[0]
        sanitized_name = "".join(c for c in basename if c.isalnum() or c in ("_", "-"))
        activation_map_path = os.path.join(self.activation_maps_path, f"{sanitized_name}.npz")
        os.makedirs(os.path.dirname(activation_map_path), exist_ok=True)
        np.savez_compressed(activation_map_path, activation_map=activation_map)
        return activation_map_path

    def _match_landmarks(self, activation_boxes: list[Box], landmark_boxes: list[Box]) -> list[Box]:
        """Match activation regions with facial landmarks."""
        if not activation_boxes or not landmark_boxes:
            return activation_boxes

        matched_boxes = []
        for a_box in activation_boxes:
            nearest = min(
                landmark_boxes,
                key=lambda l: cdist([l.center], [a_box.center], metric=self.distance_metric)[0],
            )
            overlap_width = max(0, min(a_box.max_x, nearest.max_x) - max(a_box.min_x, nearest.min_x))
            overlap_height = max(0, min(a_box.max_y, nearest.max_y) - max(a_box.min_y, nearest.min_y))
            overlap_area = overlap_width * overlap_height
            a_box.feature = nearest.feature if overlap_area / a_box.area >= self.overlap_threshold else None
            matched_boxes.append(a_box)
        return matched_boxes

    def explain_image(
        self,
        image_path: str,
        model: ClassificationModel,
        true_gender: Gender,
    ) -> tuple[list[Box], list[Box], str]:
        """Generate an explanation for a single image."""
        image = model.preprocess_image(image_path)
        activation_map = self.activation_mapper.generate_heatmap(model.model, image, true_gender)
        activation_boxes = self.activation_mapper.process_heatmap(activation_map)
        landmark_boxes = self.landmarker.detect(image_path, model.image_width, model.image_height)
        labeled_boxes = self._match_landmarks(activation_boxes, landmark_boxes)
        activation_map_path = self._save_activation_map(activation_map, image_path)
        return labeled_boxes, landmark_boxes, activation_map_path
