from typing import Optional, Union

from biasx.explainers import VisualExplainer

from .calculators import BiasCalculator
from .config import Config
from .datasets import AnalysisDataset, FaceDataset
from .models import ClassificationModel
from .types import Explanation, Gender


class BiasAnalyzer:
    """Orchestrates the bias analysis pipeline."""

    def __init__(self, config: Union[str, dict, Config]):
        """Initialize analyzer with configuration"""
        self.config = config if isinstance(config, Config) else Config.create(config)

        self.model = ClassificationModel(model_path=self.config.model_path, **self.config.model_config)
        self.dataset = FaceDataset(dataset_path=self.config.dataset_path, **self.config.dataset_config)
        self.explainer = VisualExplainer(**self.config.explainer_config)
        self.calculator = BiasCalculator(**self.config.calculator_config)

    def analyze_image(self, image_path: str, true_gender: Gender) -> Optional[Explanation]:
        """Analyze a single image and generate an explanation."""
        image = self.model.preprocess_image(image_path)
        predicted_gender, prediction_confidence = self.model.predict(image)

        activation_boxes, landmark_boxes, activation_map_path = self.explainer.explain_image(
            image_path=image_path,
            model=self.model,
            true_gender=true_gender,
        )

        return Explanation(
            image_path=image_path,
            true_gender=true_gender,
            predicted_gender=predicted_gender,
            prediction_confidence=prediction_confidence,
            activation_map_path=activation_map_path,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes,
        )

    def analyze(self, output_path: Optional[str] = None) -> AnalysisDataset:
        """Analyze a dataset of facial images and compute bias metrics."""
        results = AnalysisDataset()
        results.explanations = [result for image_path, true_gender in self.dataset if (result := self.analyze_image(image_path, true_gender))]

        features = self.explainer.landmarker.mapping.get_features()

        fairness_scores = self.calculator.compute_all_fairness_scores(results.explanations, features)
        feature_scores = {feature: self.calculator.compute_feature_scores(results.explanations, feature) for feature in features}
        feature_probabilities = {feature: self.calculator.compute_feature_probs(results.explanations, feature) for feature in features}

        results.set_bias_metrics(feature_scores=feature_scores, feature_probabilities=feature_probabilities, fairness_scores=fairness_scores)

        if output_path:
            results.save(output_path)

        return results
