"""Coordinates the bias analysis pipeline and manages result aggregation."""

from typing import Dict, List, Union

from .calculators import Calculator
from .config import Config, configurable
from .datasets import Dataset
from .explainers import Explainer
from .models import Model
from .types import AnalysisResult, Explanation, ImageData


@configurable("analyzer")
class BiasAnalyzer:
    """Orchestrates the bias analysis pipeline."""

    def __init__(self, config: Union[Config, Dict, None] = None, batch_size: int = 32, **kwargs):
        """Initialize analyzer components from configuration."""
        if config is None:
            config = {}

        self.config = config if isinstance(config, Config) else Config.create(config)
        self.model = Model(**self.config.model)
        self.dataset = Dataset(**self.config.dataset)
        self.explainer = Explainer(**self.config.explainer)
        self.calculator = Calculator(**self.config.calculator)
        self.batch_size = batch_size

    def analyze_batch(self, image_data_batch: List[ImageData]) -> List[Explanation]:
        """Analyze a batch of images through the pipeline."""
        if not image_data_batch:
            return []

        preprocessed_images = [img.preprocessed_image for img in image_data_batch]
        pil_images = [img.pil_image for img in image_data_batch]

        predictions = self.model.predict(preprocessed_images)
        predicted_genders = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]

        activation_maps, activation_boxes, landmark_boxes = self.explainer.explain_batch(
            pil_images=pil_images,
            preprocessed_images=preprocessed_images,
            model=self.model,
            target_classes=predicted_genders,
        )

        return [
            Explanation(
                image_data=image_data,
                predicted_gender=predicted_gender,
                prediction_confidence=confidence,
                activation_map=activation_map,
                activation_boxes=act_boxes,
                landmark_boxes=land_boxes,
            )
            for image_data, predicted_gender, confidence, activation_map, act_boxes, land_boxes in zip(image_data_batch, predicted_genders, confidences, activation_maps, activation_boxes, landmark_boxes)
        ]

    def analyze_image(self, image_data: ImageData) -> Explanation:
        """Analyze a single image through the pipeline."""
        explanations = self.analyze_batch([image_data])
        return explanations[0] if explanations else None

    def analyze(self) -> AnalysisResult:
        """Run the full analysis pipeline on the dataset."""
        all_explanations = []

        for batch in self.dataset:
            batch_explanations = self.analyze_batch(batch)
            all_explanations.extend(batch_explanations)

        if not all_explanations:
            return AnalysisResult()

        feature_analyses = self.calculator.calculate_feature_biases(all_explanations)
        disparity_scores = self.calculator.calculate_disparities(feature_analyses, all_explanations)

        return AnalysisResult(
            explanations=all_explanations,
            feature_analyses=feature_analyses,
            disparity_scores=disparity_scores,
        )

    @classmethod
    def from_file(cls, config_file_path: str) -> "BiasAnalyzer":
        """Create a BiasAnalyzer from a configuration file."""
        return cls(config=Config.from_file(config_file_path))
