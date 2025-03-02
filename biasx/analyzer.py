"""Coordinates the bias analysis pipeline and manages result aggregation."""

from typing import Dict, List, Union

import numpy as np

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

        batch_size = len(image_data_batch)
        if batch_size == 1:
            preprocessed_images = [image_data_batch[0].preprocessed_image]
            pil_images = [image_data_batch[0].pil_image]
        else:
            preprocessed_images = np.stack([img.preprocessed_image for img in image_data_batch])
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

        explanations = []
        for i in range(batch_size):
            explanation = Explanation(
                image_data=image_data_batch[i],
                predicted_gender=predicted_genders[i],
                prediction_confidence=confidences[i],
                activation_map=activation_maps[i],
                activation_boxes=activation_boxes[i],
                landmark_boxes=landmark_boxes[i],
            )
            explanations.append(explanation)

        return explanations

    def analyze(self) -> AnalysisResult:
        """Run the full analysis pipeline on the dataset."""
        batch_count = 0
        explanations_buffer = []
        total_explanations = []

        buffer_size = max(100, self.batch_size * 2)

        for batch in self.dataset:
            batch_explanations = self.analyze_batch(batch)
            explanations_buffer.extend(batch_explanations)
            batch_count += 1

            if len(explanations_buffer) >= buffer_size:
                total_explanations.extend(explanations_buffer)
                explanations_buffer = []

        if explanations_buffer:
            total_explanations.extend(explanations_buffer)

        if not total_explanations:
            return AnalysisResult()

        feature_analyses = self.calculator.calculate_feature_biases(total_explanations)
        disparity_scores = self.calculator.calculate_disparities(feature_analyses, total_explanations)

        return AnalysisResult(
            explanations=total_explanations,
            feature_analyses=feature_analyses,
            disparity_scores=disparity_scores,
        )

    @classmethod
    def from_file(cls, config_file_path: str) -> "BiasAnalyzer":
        """Create a BiasAnalyzer from a configuration file."""
        return cls(config=Config.from_file(config_file_path))
