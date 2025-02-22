from typing import Optional
from .calculators import BiasCalculator
from .datasets import AnalysisDataset, FaceDataset
from .explainers import VisualExplainer
from .models import ClassificationModel
from .types import Explanation


class BiasAnalyzer:
    """Orchestrates the complete bias analysis pipeline."""

    def __init__(self, model: ClassificationModel, explainer: VisualExplainer, calculator: BiasCalculator):
        """Initialize the bias analyzer."""
        self.model = model
        self.explainer = explainer
        self.calculator = calculator

    def analyze_image(self, image_path: str, true_gender: int) -> Optional[Explanation]:
        """Perform analysis on a single image."""
        try:
            image = self.model.preprocess_image(image_path)
            predicted_gender = self.model.predict(image)

            activation_map = self.explainer.generate_heatmap(self.model.model, image, true_gender)
            activation_boxes = self.explainer.process_heatmap(activation_map)
            landmark_boxes = self.explainer.detect_landmarks(image_path, self.model.target_size)
            labeled_boxes = self.explainer.match_landmarks(activation_boxes, landmark_boxes)

            return Explanation(image_path, true_gender, predicted_gender, activation_map, labeled_boxes, landmark_boxes)
        except Exception:
            return None

    def analyze(self, dataset: FaceDataset, return_explanations: bool = True) -> AnalysisDataset:
        """Perform complete bias analysis on a dataset."""
        results = AnalysisDataset()
        explanations = [result for image_path, true_gender in dataset if (result := self.analyze_image(image_path, true_gender))]

        if return_explanations:
            results.explanations = explanations

        features = self.explainer.get_features()
        results.set_bias_metrics(
            bias_score=self.calculator.compute_overall_bias(explanations, features),
            feature_scores={feature: self.calculator.compute_feature_bias(explanations, feature) for feature in features},
            feature_probabilities={feature: self.calculator.compute_feature_probabilities(explanations, feature) for feature in features},
        )

        return results
