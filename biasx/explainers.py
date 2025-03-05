"""Provides classes for generating and processing visual explanations of model decisions."""

import json
from typing import List, Tuple, Union

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

    def __init__(self, source: LandmarkerSource):
        """Initialize the facial landmark detector."""
        self.source = source
        self._load_resources()

        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=self.model_path))
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

        if not isinstance(preprocessed_images, np.ndarray) and len(preprocessed_images) == 0:
            return []

        images = self._prepare_images_for_cam(preprocessed_images)

        if isinstance(target_classes, Gender):
            target_classes = [target_classes] * len(images)

        heatmaps = []
        for i, target_class in enumerate(target_classes):
            score_function = lambda output: output[0][target_class]
            image_batch = np.expand_dims(images[i], axis=0) if images[i].ndim == 3 else images[i]
            heatmap = visualizer(score_function, image_batch, penultimate_layer=-1)[0]
            heatmaps.append(heatmap)

        return heatmaps

    def process_heatmap(self, heatmaps: Union[np.ndarray, List[np.ndarray]]) -> List[List[Box]]:
        """Process heatmaps into bounding boxes of activated regions."""
        if isinstance(heatmaps, np.ndarray) and heatmaps.ndim <= 2:
            heatmaps = [heatmaps]

        results = []
        for heatmap in heatmaps:
            threshold_value = np.percentile(heatmap, self.cutoff_percentile)
            filtered_heatmap = np.where(heatmap < threshold_value, 0, heatmap)

            binary = filtered_heatmap > self.threshold_method(filtered_heatmap)
            regions = regionprops(label(binary))

            boxes = [Box(min_x=region.bbox[1], min_y=region.bbox[0], max_x=region.bbox[3], max_y=region.bbox[2]) for region in regions]
            results.append(boxes)

        return results

    def _prepare_images_for_cam(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Prepare image array for CAM processing."""
        if isinstance(images, list):
            processed_batch = []
            for img in images:
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                processed_batch.append(img)
            return np.stack(processed_batch)

        if images.ndim == 2:
            images = np.expand_dims(images, axis=-1)
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        return images

    @staticmethod
    def _modify_model(model: tf.keras.Model) -> None:
        """Modify model for activation map generation."""
        model.layers[-1].activation = tf.keras.activations.linear


@configurable("explainer")
class Explainer:
    """Coordinates generation of visual explanations for model decisions."""

    def __init__(self, landmarker_source: LandmarkerSource, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod, overlap_threshold: float, distance_metric: DistanceMetric, batch_size: int, **kwargs):
        """Initialize the visual explainer."""
        self.landmarker = FacialLandmarker(source=landmarker_source)
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

        labeled_boxes = []
        for a_boxes, l_boxes in zip(activation_boxes, landmark_boxes):
            if not a_boxes or not l_boxes:
                labeled_boxes.append(a_boxes)
                continue

            a_centers = np.array([box.center for box in a_boxes])
            l_centers = np.array([box.center for box in l_boxes])
            distances = cdist(a_centers, l_centers, metric=self.distance_metric)

            nearest_indices = np.argmin(distances, axis=1)

            for i, (a_box, nearest_idx) in enumerate(zip(a_boxes, nearest_indices)):
                l_box = l_boxes[nearest_idx]

                overlap_width = max(0, min(a_box.max_x, l_box.max_x) - max(a_box.min_x, l_box.min_x))
                overlap_height = max(0, min(a_box.max_y, l_box.max_y) - max(a_box.min_y, l_box.min_y))
                overlap_area = overlap_width * overlap_height

                if overlap_area / a_box.area >= self.overlap_threshold:
                    a_box.feature = l_box.feature

            labeled_boxes.append(a_boxes)

        return activation_maps, labeled_boxes, landmark_boxes
