import os
from typing import Optional

from .calculators import BiasCalculator
from .datasets import AnalysisDataset, FaceDataset
from .explainers import VisualExplainer
from .models import ClassificationModel
from .types import Explanation


class BiasAnalyzer:
    """Orchestrates the complete bias analysis pipeline with simplified setup."""

    def __init__(
        self,
        model_path: str,
        target_size: Optional[tuple[int, int]] = (128, 128),
        color_mode: Optional[str] = "L",
        single_channel: Optional[bool] = False,
        explainer: Optional[VisualExplainer] = None,
        calculator: Optional[BiasCalculator] = None,
    ):
        """Initialize the bias analyzer with automatic model, explainer, and calculator setup."""
        self.model = ClassificationModel(model_path, target_size, color_mode, single_channel)
        self.explainer = explainer or VisualExplainer()
        self.calculator = calculator or BiasCalculator()

    def analyze_image(self, image_path: str, true_gender: int) -> Optional[Explanation]:
        """Perform bias analysis on a single image."""
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

    def analyze(
        self,
        dataset_path: str,
        max_samples: Optional[int] = -1,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 69,
        return_explanations: Optional[bool] = True,
        output_path: Optional[str] = None,
    ) -> AnalysisDataset:
        """Perform complete bias analysis on a dataset with optional automatic saving."""
        dataset = FaceDataset(dataset_path, max_samples, shuffle, seed)
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

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results.save(output_path)

        return results
