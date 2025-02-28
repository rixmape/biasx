"""Provides classes for generating and processing visual explanations of model decisions."""

import json
from typing import List, Tuple

import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from .config import configurable
from .models import Model
from .types import Box, CAMMethod, DistanceMetric, FacialFeature, Gender, LandmarkerMetadata, LandmarkerSource, ThresholdMethod
from .utils import download_resource, get_module_data_path, load_json_config


@configurable("landmarker")
class FacialLandmarker:
    """Detects facial landmarks using MediaPipe."""

    def __init__(self, source: LandmarkerSource, max_faces: int, **kwargs):
        """Initialize the facial landmark detector."""
        self.source = source
        self.landmarker_info = self._load_landmarker_metadata()
        self.model_path = self._download_model()
        self.landmark_mapping = self._load_landmark_mapping()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            num_faces=max_faces,
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def _load_landmarker_metadata(self) -> LandmarkerMetadata:
        """Load landmarker metadata from configuration file."""
        config = load_json_config(__file__, "landmarker_config.json")

        if self.source.value not in config:
            raise ValueError(f"Landmarker source {self.source.value} not found in configuration")

        return LandmarkerMetadata(**config[self.source.value])

    def _download_model(self) -> str:
        """Download the facial landmark model from HuggingFace."""
        return download_resource(
            repo_id=self.landmarker_info.repo_id,
            filename=self.landmarker_info.filename,
            repo_type=self.landmarker_info.repo_type,
        )

    def _load_landmark_mapping(self) -> dict:
        """Load facial landmark mapping from JSON configuration file."""
        mapping_path = get_module_data_path(__file__, "landmark_mapping.json")

        with open(str(mapping_path), "r") as f:
            mapping_data = json.load(f)

        landmark_mapping = {}
        for feature_name, indices in mapping_data.items():
            feature_enum = FacialFeature(feature_name)
            landmark_mapping[feature_enum] = indices

        return landmark_mapping

    def detect(self, image: Image.Image) -> List[Box]:
        """Detect facial landmarks in an image."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return []

        image_width, image_height = image.size
        points = [(int(round(point.x * image_width)), int(round(point.y * image_height))) for point in result.face_landmarks[0]]

        boxes = []
        for feature, indices in self.landmark_mapping.items():
            feature_points = [points[i] for i in indices]
            box = Box(
                min_x=min(x for x, _ in feature_points),
                min_y=min(y for _, y in feature_points),
                max_x=max(x for x, _ in feature_points),
                max_y=max(y for _, y in feature_points),
                feature=feature,
            )
            boxes.append(box)

        return boxes


@configurable("explainer")
class ClassActivationMapper:
    """Generates and processes class activation maps."""

    def __init__(self, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod, **kwargs):
        """Initialize the activation map generator."""
        self.cam_method = cam_method.get_implementation()
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = threshold_method.get_implementation()

    def generate_heatmap(
        self,
        model: tf.keras.Model,
        preprocessed_image: np.ndarray,
        target_class: Gender,
    ) -> np.ndarray:
        """Generate class activation map for a preprocessed image."""
        visualizer = self.cam_method(model, model_modifier=self._modify_model, clone=True)
        image = self._prepare_image_for_cam(preprocessed_image)
        heatmap = visualizer(lambda output: self._score_function(output, target_class), image, penultimate_layer=-1)[0]

        return heatmap

    def process_heatmap(self, heatmap: np.ndarray) -> List[Box]:
        """Process heatmap into bounding boxes of activated regions."""
        filtered_heatmap = heatmap.copy()
        filtered_heatmap[filtered_heatmap < np.percentile(filtered_heatmap, self.cutoff_percentile)] = 0

        binary = filtered_heatmap > self.threshold_method(filtered_heatmap)

        regions = regionprops(label(binary))

        boxes = []
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            box = Box(min_x=min_col, min_y=min_row, max_x=max_col, max_y=max_row)
            boxes.append(box)

        return boxes

    @staticmethod
    def _modify_model(model: tf.keras.Model) -> None:
        """Modify model for activation map generation."""
        model.layers[-1].activation = tf.keras.activations.linear

    @staticmethod
    def _score_function(output: tf.Tensor, target_class: Gender) -> tf.Tensor:
        """Score function for class activation mapping."""
        return output[0][target_class]

    @staticmethod
    def _prepare_image_for_cam(image: np.ndarray) -> np.ndarray:
        """Prepare image array for CAM processing."""
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        return image


@configurable("explainer")
class Explainer:
    """Coordinates generation of visual explanations for model decisions."""

    def __init__(self, landmarker_source: LandmarkerSource, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod, overlap_threshold: float, distance_metric: DistanceMetric, **kwargs):
        """Initialize the visual explainer."""
        self.landmarker = FacialLandmarker(source=landmarker_source)
        self.activation_mapper = ClassActivationMapper(cam_method=cam_method, cutoff_percentile=cutoff_percentile, threshold_method=threshold_method)
        self.overlap_threshold = overlap_threshold
        self.distance_metric = distance_metric.value

    def explain_image(
        self,
        pil_image: Image.Image,
        preprocessed_image: np.ndarray,
        model: Model,
        target_class: Gender,
    ) -> Tuple[np.ndarray, List[Box], List[Box]]:
        """Generate visual explanation for a single image."""
        activation_map = self.activation_mapper.generate_heatmap(model.model, preprocessed_image, target_class)
        activation_boxes = self.activation_mapper.process_heatmap(activation_map)
        landmark_boxes = self.landmarker.detect(pil_image)
        labeled_boxes = self._match_landmarks(activation_boxes, landmark_boxes)

        return activation_map, labeled_boxes, landmark_boxes

    def _match_landmarks(self, activation_boxes: List[Box], landmark_boxes: List[Box]) -> List[Box]:
        """Match activation regions with facial landmarks."""
        if not activation_boxes or not landmark_boxes:
            return activation_boxes

        matched_boxes = []
        for a_box in activation_boxes:
            nearest = min(
                landmark_boxes,
                key=lambda l: cdist([l.center], [a_box.center], metric=self.distance_metric)[0][0],
            )

            overlap_width = max(0, min(a_box.max_x, nearest.max_x) - max(a_box.min_x, nearest.min_x))
            overlap_height = max(0, min(a_box.max_y, nearest.max_y) - max(a_box.min_y, nearest.min_y))
            overlap_area = overlap_width * overlap_height

            if overlap_area / a_box.area >= self.overlap_threshold:
                a_box.feature = nearest.feature

            matched_boxes.append(a_box)

        return matched_boxes
