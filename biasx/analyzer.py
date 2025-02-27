"""
Analysis orchestration module for the BiasX library.
Coordinates the bias analysis pipeline and manages result aggregation.
"""

from typing import Optional

from .calculators import BiasCalculator
from .config import Config
from .datasets import FaceDataset
from .explainers import VisualExplainer
from .models import ClassificationModel
from .types import AnalysisResult, Explanation


class BiasAnalyzer:
    """Orchestrates the bias analysis pipeline."""

    def __init__(self, config: Config):
        """Initialize analyzer components from configuration."""
        self.config = config if isinstance(config, Config) else Config.create(config)

        self.model = ClassificationModel(**vars(self.config.model_config))
        self.dataset = FaceDataset(**vars(self.config.dataset_config))
        self.explainer = VisualExplainer(**vars(self.config.explainer_config))
        self.calculator = BiasCalculator(**vars(self.config.calculator_config))

    def analyze_image(self, image_data) -> Optional[Explanation]:
        """Analyze a single image through the pipeline."""
        predicted_gender, confidence = self.model.predict(image_data.preprocessed_image)

        activation_map, activation_boxes, landmark_boxes = self.explainer.explain_image(
            pil_image=image_data.pil_image,
            preprocessed_image=image_data.preprocessed_image,
            model=self.model,
            target_class=predicted_gender,
        )

        explanation = Explanation(
            image_data=image_data,
            predicted_gender=predicted_gender,
            prediction_confidence=confidence,
            activation_map=activation_map,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes,
        )

        return explanation

    def analyze(self) -> AnalysisResult:
        """Run the full analysis pipeline on the dataset."""
        explanations = []
        for image_data in self.dataset:
            explanation = self.analyze_image(image_data)
            if explanation:
                explanations.append(explanation)

        if not explanations:
            return AnalysisResult()

        feature_analyses = self.calculator.calculate_feature_biases(explanations)
        disparity_scores = self.calculator.calculate_disparities(feature_analyses, explanations)

        return AnalysisResult(
            explanations=explanations,
            feature_analyses=feature_analyses,
            disparity_scores=disparity_scores,
        )
