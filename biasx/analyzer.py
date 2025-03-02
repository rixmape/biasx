"""Coordinates the bias analysis pipeline and manages result aggregation."""

from typing import Dict, Union

from .calculators import Calculator
from .config import Config, configurable
from .datasets import Dataset
from .explainers import Explainer
from .models import Model
from .types import AnalysisResult, Explanation, ImageData


@configurable("analyzer")
class BiasAnalyzer:
    """Orchestrates the bias analysis pipeline."""

    def __init__(self, config: Union[Config, Dict, None] = None, **kwargs):
        """Initialize analyzer components from configuration."""
        if config is None:
            config = {}

        self.config = config if isinstance(config, Config) else Config.create(config)
        self.model = Model(**self.config.model)
        self.dataset = Dataset(**self.config.dataset)
        self.explainer = Explainer(**self.config.explainer)
        self.calculator = Calculator(**self.config.calculator)

    def analyze_image(self, image_data: ImageData) -> Explanation:
        """Analyze a single image through the pipeline."""
        predicted_gender, confidence = self.model.predict(image_data.preprocessed_image)

        activation_map, activation_boxes, landmark_boxes = self.explainer.explain_image(
            pil_image=image_data.pil_image,
            preprocessed_image=image_data.preprocessed_image,
            model=self.model,
            target_class=predicted_gender,
        )

        return Explanation(
            image_data=image_data,
            predicted_gender=predicted_gender,
            prediction_confidence=confidence,
            activation_map=activation_map,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes,
        )

    def analyze(self) -> AnalysisResult:
        """Run the full analysis pipeline on the dataset."""
        explanations = [exp for exp in (self.analyze_image(img_data) for img_data in self.dataset) if exp]

        if not explanations:
            return AnalysisResult()

        feature_analyses = self.calculator.calculate_feature_biases(explanations)
        disparity_scores = self.calculator.calculate_disparities(feature_analyses, explanations)

        return AnalysisResult(
            explanations=explanations,
            feature_analyses=feature_analyses,
            disparity_scores=disparity_scores,
        )

    @classmethod
    def from_file(cls, config_file_path: str) -> "BiasAnalyzer":
        """Create a BiasAnalyzer from a configuration file."""
        return cls(config=Config.from_file(config_file_path))
