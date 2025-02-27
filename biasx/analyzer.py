from typing import Optional, Union

from .calculators import BiasCalculator
from .config import Config
from .datasets import AnalysisDataset, FaceDataset
from .explainers import VisualExplainer
from .models import ClassificationModel
from .types import Age, Explanation, Gender, Race


class BiasAnalyzer:
    """Orchestrates the bias analysis pipeline."""

    def __init__(self, config: Union[dict, Config]):
        """Initialize analyzer with configuration"""
        self.config = config if isinstance(config, Config) else Config.create(config)

        self.model = ClassificationModel(**self.config.model_config.to_dict())
        self.dataset = FaceDataset(**self.config.dataset_config.to_dict())
        self.explainer = VisualExplainer(**self.config.explainer_config.to_dict())
        self.calculator = BiasCalculator(**self.config.calculator_config.to_dict())

    def analyze_image(self, image_path: str, image_id: str, true_gender: Gender, age: Age, race: Race) -> Optional[Explanation]:
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
            image_id=image_id,
            true_gender=true_gender,
            predicted_gender=predicted_gender,
            age=age,
            race=race,
            prediction_confidence=prediction_confidence,
            activation_map_path=activation_map_path,
            activation_boxes=activation_boxes,
            landmark_boxes=landmark_boxes,
        )

    def analyze(self, output_path: Optional[str] = None) -> AnalysisDataset:
        """Analyze a dataset of facial images and compute bias metrics."""
        results = AnalysisDataset()

        results.explanations = [
            result
            for image_path, image_id, true_gender, age, race in self.dataset
            if (
                result := self.analyze_image(
                    image_path=image_path,
                    image_id=image_id,
                    true_gender=true_gender,
                    age=age,
                    race=race,
                )
            )
        ]

        features = self.explainer.landmarker.mapping.get_features()

        fairness_scores = self.calculator.compute_all_fairness_scores(results.explanations, features)
        feature_scores = {feature: self.calculator.compute_feature_scores(results.explanations, feature) for feature in features}
        feature_probabilities = {feature: self.calculator.compute_feature_probs(results.explanations, feature) for feature in features}

        results.set_bias_metrics(feature_scores=feature_scores, feature_probabilities=feature_probabilities, fairness_scores=fairness_scores)

        if output_path:
            results.save(output_path)

        return results
