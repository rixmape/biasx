from typing import Any

from .datasets import FaceDataset, AnalysisDataset
from .models import ClassificationModel
from .explainers import VisualExplainer
from .calculators import BiasCalculator


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
