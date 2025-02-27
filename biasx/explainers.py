"""
Explanation module for BiasX.
Provides classes for generating and processing visual explanations of model decisions.
"""

import json
import pathlib

import mediapipe as mp
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu, threshold_sauvola, threshold_triangle
from skimage.measure import label, regionprops
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam

from .models import ClassificationModel
from .types import Box, CAMMethod, DistanceMetric, FacialFeature, Gender, ThresholdMethod


class FacialLandmarker:
    """Detects facial landmarks using MediaPipe."""

    MAX_FACES = 1

    def __init__(self):
        """Initialize the facial landmark detector."""
        model_path = self._download_model()
        self.landmark_mapping = self._load_landmark_mapping()
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), num_faces=self.MAX_FACES)
        self.detector = FaceLandmarker.create_from_options(options)

    def _download_model(self) -> str:
        """Download the MediaPipe facial landmark model from HuggingFace."""
        model_filename = "mediapipe_landmarker.task"
        repo_id = "rixmape/biasx-models"

        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, repo_type="model")

        return model_path

    def _load_landmark_mapping(self) -> dict:
        """Load facial landmark mapping from JSON configuration file."""
        module_dir = pathlib.Path(__file__).parent
        mapping_path = module_dir / "data" / "landmark_mapping.json"

        if not mapping_path.exists():
            raise FileNotFoundError(f"Landmark mapping file not found at {mapping_path}")

        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)

        landmark_mapping = {}
        for feature_name, indices in mapping_data.items():
            try:
                feature_enum = FacialFeature(feature_name)
                landmark_mapping[feature_enum] = indices
            except ValueError:
                continue

        return landmark_mapping

    def detect(self, image: Image) -> list[Box]:
        """Detect facial landmarks in an image."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return []

        image_width, image_height = image.size
        points = [
            (int(round(point.x * image_width)), int(round(point.y * image_height)))
            for point in result.face_landmarks[0]
        ]

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


class ClassActivationMapper:
    """Generates and processes class activation maps."""

    CAM_METHODS = {
        CAMMethod.GRADCAM: Gradcam,
        CAMMethod.GRADCAM_PLUS_PLUS: GradcamPlusPlus,
        CAMMethod.SCORECAM: Scorecam,
    }

    THRESHOLD_METHODS = {
        ThresholdMethod.OTSU: threshold_otsu,
        ThresholdMethod.SAUVOLA: threshold_sauvola,
        ThresholdMethod.TRIANGLE: threshold_triangle,
    }

    def __init__(
        self,
        cam_method: CAMMethod = CAMMethod.GRADCAM_PLUS_PLUS,
        cutoff_percentile: int = 90,
        threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
    ):
        """Initialize the activation map generator."""
        self.cam_method = self.CAM_METHODS[cam_method]
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = self.THRESHOLD_METHODS[threshold_method]

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

    def process_heatmap(self, heatmap: np.ndarray) -> list[Box]:
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


class VisualExplainer:
    """Coordinates generation of visual explanations for model decisions."""

    def __init__(
        self,
        cam_method: CAMMethod = CAMMethod.GRADCAM_PLUS_PLUS,
        cutoff_percentile: int = 90,
        threshold_method: ThresholdMethod = ThresholdMethod.OTSU,
        overlap_threshold: float = 0.2,
        distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ):
        """Initialize the visual explainer."""
        self.landmarker = FacialLandmarker()

        self.activation_mapper = ClassActivationMapper(
            cam_method=cam_method,
            cutoff_percentile=cutoff_percentile,
            threshold_method=threshold_method,
        )

        self.overlap_threshold = overlap_threshold
        self.distance_metric = distance_metric.value

    def explain_image(
        self,
        pil_image: Image.Image,
        preprocessed_image: np.ndarray,
        model: ClassificationModel,
        target_class: Gender,
    ) -> tuple[np.ndarray, list[Box], list[Box]]:
        """Generate visual explanation for a single image."""
        activation_map = self.activation_mapper.generate_heatmap(model.model, preprocessed_image, target_class)
        activation_boxes = self.activation_mapper.process_heatmap(activation_map)
        landmark_boxes = self.landmarker.detect(pil_image)
        labeled_boxes = self._match_landmarks(activation_boxes, landmark_boxes)

        return activation_map, labeled_boxes, landmark_boxes

    def _match_landmarks(self, activation_boxes: list[Box], landmark_boxes: list[Box]) -> list[Box]:
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
