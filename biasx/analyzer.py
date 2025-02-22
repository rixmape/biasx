from typing import Optional

from biasx.explainers import VisualExplainer

from .calculators import BiasCalculator
from .datasets import AnalysisDataset, FaceDataset
from .models import ClassificationModel
from .types import Explanation


class BiasAnalyzer:
    """Orchestrates the bias analysis pipeline."""

    def __init__(
        self,
        model_path: str,
        target_size: tuple[int, int] = (128, 128),
        color_mode: str = "L",
        single_channel: bool = False,
        visual_explainer: Optional[VisualExplainer] = None,
        calculator: Optional[BiasCalculator] = None,
    ):
        self.model = ClassificationModel(model_path, target_size, color_mode, single_channel)
        self.visual_explainer = visual_explainer or VisualExplainer()
        self.calculator = calculator or BiasCalculator()
        self.target_size = target_size

    def analyze_image(self, image_path: str, true_gender: int) -> Optional[Explanation]:
        """Analyze a single image and generate an explanation."""
        image = self.model.preprocess_image(image_path)
        predicted_gender, prediction_confidence = self.model.predict(image)

        activation_boxes, landmark_boxes, activation_map = self.visual_explainer.explain_image(image_path, self.model, true_gender, self.target_size)

        return Explanation(
            image_path=image_path,
            true_gender=true_gender,
            predicted_gender=predicted_gender,
            prediction_confidence=prediction_confidence,
            activation_map=activation_map,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes,
        )

    def analyze(
        self,
        dataset_path: str,
        max_samples: int = -1,
        shuffle: bool = True,
        seed: int = 69,
        return_explanations: bool = True,
        output_path: Optional[str] = None,
    ) -> AnalysisDataset:
        """Analyze a dataset of facial images and compute bias metrics."""
        dataset = FaceDataset(dataset_path, max_samples, shuffle, seed)
        results = AnalysisDataset()

        explanations = [result for image_path, true_gender in dataset if (result := self.analyze_image(image_path, true_gender))]

        if return_explanations:
            results.explanations = explanations

        features = self.visual_explainer.landmarker.mapping.get_features()
        results.set_bias_metrics(
            bias_score=self.calculator.compute_overall_bias(explanations, features),
            feature_scores={feature: self.calculator.compute_feature_bias(explanations, feature) for feature in features},
            feature_probabilities={feature: self.calculator.compute_feature_probabilities(explanations, feature) for feature in features},
        )

        if output_path:
            results.save(output_path)

        return results
