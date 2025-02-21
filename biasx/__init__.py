"""
This module implements a comprehensive pipeline for analyzing gender bias in face classification models.
It provides tools for model evaluation, visual explanation generation, and bias metric calculation.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import keras
import mediapipe
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from tf_keras_vis.gradcam import GradcamPlusPlus


@dataclass
class Box:
    """Represents a bounding box with optional feature label."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    feature: Optional[str] = None

    @property
    def center(self) -> tuple[float, float]:
        """Compute center coordinates of the box."""
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def area(self) -> float:
        """Compute area of the box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def to_dict(self) -> dict[str, Any]:
        """Convert box to dictionary format."""
        data = {"minX": self.min_x, "minY": self.min_y, "maxX": self.max_x, "maxY": self.max_y}
        if self.feature is not None:
            data["feature"] = self.feature
        return data


class FaceDataset:
    """Manages the facial image dataset used for bias analysis."""

    def __init__(self, dataset_path: str, max_samples: int = -1):
        """
        Initialize the dataset from a directory of facial images.

        Args:
            dataset_path: Path to directory containing facial images
            max_samples: Maximum number of samples to load (-1 for all)
        """
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

        self.image_paths = []
        self.genders = []

        paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
        if max_samples > 0:
            paths = paths[:max_samples]

        for path in paths:
            filename = os.path.basename(path).split(".")[0]
            gender = int(filename.split("_")[1])
            self.image_paths.append(path)
            self.genders.append(gender)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.image_paths[idx], self.genders[idx]


class ClassificationModel:
    """Handles loading and inference of the face classification model."""

    def __init__(self, model_path: str, input_shape: tuple[int, int, int], preprocessing: dict[str, Any]):
        """
        Initialize the classification model.

        Args:
            model_path: Path to saved model file
            input_shape: Expected shape of input images (height, width, channels)
            preprocessing: dictionary of preprocessing parameters
        """
        self.model = keras.models.load_model(model_path)
        self.input_shape = input_shape
        self.preprocessing = preprocessing

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess a single image for model input."""
        with Image.open(image_path) as image:
            image = image.convert(self.preprocessing["color_mode"])
            image = image.resize(self.preprocessing["target_size"])
            return np.array(image, dtype=np.float32) / 255.0

    def predict(self, image: np.ndarray) -> int:
        """Generate prediction for a single preprocessed image."""
        prediction = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        return int(np.argmax(prediction, axis=1)[0])


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


class BiasCalculator:
    """Computes bias metrics from analysis results."""

    def __init__(self, ndigits: Optional[int] = 3):
        """
        Initialize the bias calculator.

        Args:
            ndigits: Number of digits to round bias scores to
        """
        self.ndigits = ndigits

    def compute_feature_probabilities(self, results: list[dict[str, Any]], feature: str) -> dict[int, float]:
        """Compute feature activation probabilities per class."""
        misclassified = [r for r in results if r["predictedGender"] != r["trueGender"]]

        if not misclassified:
            return {0: 0.0, 1: 0.0}

        probs = {}
        for gender in (0, 1):
            gender_results = [r for r in misclassified if r["trueGender"] == gender]
            if not gender_results:
                probs[gender] = 0.0
                continue

            feature_count = sum(1 for r in gender_results for box in r["activationBoxes"] if box.feature == feature)
            probs[gender] = round(feature_count / len(gender_results), self.ndigits)

        return probs

    def compute_feature_bias(self, results: list[dict[str, Any]], feature: str) -> float:
        """Compute bias score for a specific feature."""
        probs = self.compute_feature_probabilities(results, feature)
        return round(abs(probs[1] - probs[0]), self.ndigits)

    def compute_overall_bias(self, results: list[dict[str, Any]], features: list[str]) -> float:
        """Compute overall bias score across all features."""
        scores = [self.compute_feature_bias(results, feature) for feature in features]
        return round(np.mean(scores), self.ndigits)


class AnalysisDataset:
    """Manages storage and serialization of analysis results."""

    def __init__(self):
        """Initialize empty analysis dataset."""
        self.bias_score = None
        self.feature_scores = {}
        self.feature_probabilities = {}
        self.explanations = []

    def add_explanation(self, image_path: str, true_gender: int, predicted_gender: int, activation_map: np.ndarray, activation_boxes: list[Box], landmark_boxes: list[Box]) -> None:
        """Add a single image analysis result."""
        self.explanations.append(
            {
                "imagePath": image_path,
                "trueGender": true_gender,
                "predictedGender": predicted_gender,
                "activationMap": activation_map.tolist(),
                "activationBoxes": [box.to_dict() for box in activation_boxes],
                "landmarkBoxes": [box.to_dict() for box in landmark_boxes],
            }
        )

    def set_bias_metrics(self, bias_score: float, feature_scores: dict[str, float], feature_probabilities: dict[str, dict[int, float]]) -> None:
        """Set computed bias metrics."""
        self.bias_score = bias_score
        self.feature_scores = feature_scores
        self.feature_probabilities = feature_probabilities

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary format."""
        return {"biasScore": self.bias_score, "featureScores": self.feature_scores, "featureProbabilities": self.feature_probabilities, "explanations": self.explanations}

    def save(self, output_path: str) -> None:
        """Save dataset to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BiasAnalyzer:
    """Orchestrates the complete bias analysis pipeline."""

    def __init__(self, model: ClassificationModel, explainer: VisualExplainer, calculator: BiasCalculator):
        """
        Initialize the bias analyzer.

        Args:
            model: Initialized classification model
            explainer: Initialized visual explainer
            calculator: Initialized bias calculator
        """
        self.model = model
        self.explainer = explainer
        self.calculator = calculator

    def analyze_image(self, image_path: str, true_gender: int) -> dict[str, Any]:
        """
        Perform complete analysis of a single image.

        Args:
            image_path: Path to the image file
            true_gender: Ground truth gender label

        Returns:
            dictionary containing all analysis results for the image
        """
        try:
            image = self.model.preprocess_image(image_path)
            predicted_gender = self.model.predict(image)

            activation_map = self.explainer.generate_heatmap(self.model.model, image, true_gender)
            activation_boxes = self.explainer.process_heatmap(activation_map)

            landmark_boxes = self.explainer.detect_landmarks(image_path)
            labeled_boxes = self.explainer.match_landmarks(activation_boxes, landmark_boxes)

            return {
                "imagePath": image_path,
                "trueGender": true_gender,
                "predictedGender": predicted_gender,
                "activationMap": activation_map,
                "activationBoxes": labeled_boxes,
                "landmarkBoxes": landmark_boxes,
            }

        except Exception as e:
            print(f"Error analyzing image {image_path}: {e}")
            return None

    def analyze(self, dataset: FaceDataset) -> AnalysisDataset:
        """
        Perform complete bias analysis on a dataset.

        Args:
            dataset: FaceDataset instance containing images to analyze
            progress_bar: Whether to show progress bar during analysis

        Returns:
            AnalysisDataset containing complete analysis results
        """
        results = AnalysisDataset()

        image_results = []

        for image_path, true_gender in dataset:
            result = self.analyze_image(image_path, true_gender)
            if result is not None:
                image_results.append(result)
                results.add_explanation(
                    image_path=result["imagePath"],
                    true_gender=result["trueGender"],
                    predicted_gender=result["predictedGender"],
                    activation_map=result["activationMap"],
                    activation_boxes=result["activationBoxes"],
                    landmark_boxes=result["landmarkBoxes"],
                )

        features = list(self.explainer.landmark_map.keys())
        feature_scores = {feature: self.calculator.compute_feature_bias(image_results, feature) for feature in features}
        feature_probabilities = {feature: self.calculator.compute_feature_probabilities(image_results, feature) for feature in features}
        bias_score = self.calculator.compute_overall_bias(image_results, features)

        results.set_bias_metrics(bias_score=bias_score, feature_scores=feature_scores, feature_probabilities=feature_probabilities)
        return results
