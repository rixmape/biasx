"""Provides classes for generating and processing visual explanations of model decisions."""

import json
from typing import Dict, List, Tuple, Union

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
from .types import Box, CAMMethod, DistanceMetric, FacialFeature, Gender, LandmarkerSource, ResourceMetadata, ThresholdMethod
from .utils import get_file_path, get_json_config, get_resource_path


class FacialLandmarker:
    """Detects facial landmarks using MediaPipe."""

    def __init__(self, source: LandmarkerSource, max_faces: int):
        """Initialize the facial landmark detector."""
        self.source = source
        self.max_faces = max_faces
        self._load_resources()

        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=self.model_path), num_faces=self.max_faces)
        self.detector = FaceLandmarker.create_from_options(options)

    def _load_resources(self) -> None:
        """Load landmarker resources from configuration."""
        config = get_json_config(__file__, "landmarker_config.json")

        if self.source.value not in config:
            raise ValueError(f"Landmarker source {self.source.value} not found in configuration")

        metadata_dict = config[self.source.value]
        self.landmarker_info = ResourceMetadata(**metadata_dict)
        self.model_path = get_resource_path(repo_id=self.landmarker_info.repo_id, filename=self.landmarker_info.filename, repo_type=self.landmarker_info.repo_type)

        mapping_path = get_file_path(__file__, "data/landmark_mapping.json")
        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)

        self.landmark_mapping = {}
        for feature_name, indices in mapping_data.items():
            feature_enum = FacialFeature(feature_name)
            self.landmark_mapping[feature_enum] = indices

    def detect(self, images: Union[Image.Image, List[Image.Image]]) -> List[List[Box]]:
        """Detect facial landmarks in images."""
        if isinstance(images, Image.Image):
            images = [images]

        results = []
        for image in images:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
            result = self.detector.detect(mp_image)

            if not result.face_landmarks:
                results.append([])
                continue

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

            results.append(boxes)

        return results


class ClassActivationMapper:
    """Generates and processes class activation maps."""

    def __init__(self, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod):
        """Initialize the activation map generator."""
        self.cam_method = cam_method.get_implementation()
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = threshold_method.get_implementation()

    def generate_heatmap(self, model: tf.keras.Model, preprocessed_images: Union[np.ndarray, List[np.ndarray]], target_classes: Union[Gender, List[Gender]]) -> List[np.ndarray]:
        """Generate class activation maps for preprocessed images."""
        visualizer = self.cam_method(model, model_modifier=self._modify_model, clone=True)

        if isinstance(preprocessed_images, list):
            if not preprocessed_images:
                return []

            processed_batch = []
            for img in preprocessed_images:
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                processed_batch.append(img)

            images = np.stack(processed_batch)
        else:
            images = preprocessed_images
            if images.ndim == 2:
                images = np.expand_dims(images, axis=-1)
            if images.ndim == 3:
                images = np.expand_dims(images, axis=0)

        if isinstance(target_classes, Gender):
            target_classes = [target_classes] * len(images)

        heatmaps = []
        for i, image in enumerate(images):
            prepared_image = np.expand_dims(image, axis=0) if image.ndim == 3 else image
            heatmap = visualizer(lambda output: self._score_function(output, target_classes[i]), prepared_image, penultimate_layer=-1)[0]
            heatmaps.append(heatmap)

        return heatmaps

    def process_heatmap(self, heatmaps: Union[np.ndarray, List[np.ndarray]]) -> List[List[Box]]:
        """Process heatmaps into bounding boxes of activated regions."""
        if isinstance(heatmaps, np.ndarray):
            heatmaps = [heatmaps]

        results = []
        for heatmap in heatmaps:
            filtered_heatmap = heatmap.copy()
            filtered_heatmap[filtered_heatmap < np.percentile(filtered_heatmap, self.cutoff_percentile)] = 0

            binary = filtered_heatmap > self.threshold_method(filtered_heatmap)
            regions = regionprops(label(binary))

            boxes = [Box(min_x=region.bbox[1], min_y=region.bbox[0], max_x=region.bbox[3], max_y=region.bbox[2]) for region in regions]
            results.append(boxes)

        return results

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

    def __init__(self, landmarker_source: LandmarkerSource, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod, overlap_threshold: float, distance_metric: DistanceMetric, max_faces: int, batch_size: int, **kwargs):
        """Initialize the visual explainer."""
        self.landmarker = FacialLandmarker(source=landmarker_source, max_faces=max_faces)
        self.activation_mapper = ClassActivationMapper(cam_method=cam_method, cutoff_percentile=cutoff_percentile, threshold_method=threshold_method)
        self.overlap_threshold = overlap_threshold
        self.distance_metric = distance_metric.value
        self.batch_size = batch_size

    def explain_batch(self, pil_images: List[Image.Image], preprocessed_images: List[np.ndarray], model: Model, target_classes: List[Gender]) -> Tuple[List[np.ndarray], List[List[Box]], List[List[Box]]]:
        """Generate visual explanations for a batch of images."""
        if not pil_images:
            return [], [], []

        activation_maps = self.activation_mapper.generate_heatmap(model.model, preprocessed_images, target_classes)
        activation_boxes = self.activation_mapper.process_heatmap(activation_maps)
        landmark_boxes = self.landmarker.detect(pil_images)
        labeled_boxes = [self._match_landmarks(a_boxes, l_boxes) for a_boxes, l_boxes in zip(activation_boxes, landmark_boxes)]

        return activation_maps, labeled_boxes, landmark_boxes

    def explain_image(self, pil_image: Image.Image, preprocessed_image: np.ndarray, model: Model, target_class: Gender) -> Tuple[np.ndarray, List[Box], List[Box]]:
        """Generate visual explanation for a single image."""
        activation_maps, labeled_boxes, landmark_boxes = self.explain_batch([pil_image], [preprocessed_image], model, [target_class])
        return activation_maps[0], labeled_boxes[0], landmark_boxes[0]

    def _match_landmarks(self, activation_boxes: List[Box], landmark_boxes: List[Box]) -> List[Box]:
        """Match activation regions with facial landmarks."""
        if not activation_boxes or not landmark_boxes:
            return activation_boxes

        matched_boxes = []
        for a_box in activation_boxes:
            nearest = min(landmark_boxes, key=lambda l: cdist([l.center], [a_box.center], metric=self.distance_metric)[0][0])
            overlap_width = max(0, min(a_box.max_x, nearest.max_x) - max(a_box.min_x, nearest.min_x))
            overlap_height = max(0, min(a_box.max_y, nearest.max_y) - max(a_box.min_y, nearest.min_y))
            overlap_area = overlap_width * overlap_height

            if overlap_area / a_box.area >= self.overlap_threshold:
                a_box.feature = nearest.feature

            matched_boxes.append(a_box)

        return matched_boxes
