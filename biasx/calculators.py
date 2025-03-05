"""Bias calculation module for BiasX."""

from typing import Dict, List, Tuple

from sklearn.metrics import confusion_matrix

from .config import configurable
from .types import DisparityScores, Explanation, FacialFeature, FeatureAnalysis, Gender


@configurable("calculator")
class Calculator:
    """Calculates metrics related to bias in facial classification models."""

    def __init__(self, precision: int, **kwargs):
        """Initialize the bias calculator."""
        self.precision = precision

    def calculate_feature_biases(self, explanations: List[Explanation]) -> Dict[FacialFeature, FeatureAnalysis]:
        """Calculate bias metrics for each facial feature."""
        feature_map = self._get_feature_activation_map(explanations)

        return {
            feature: FeatureAnalysis(
                feature=feature,
                bias_score=round(abs(probs[Gender.MALE] - probs[Gender.FEMALE]), self.precision),
                male_probability=probs[Gender.MALE],
                female_probability=probs[Gender.FEMALE],
            )
            for feature, probs in feature_map.items()
            if feature is not None
        }

    def calculate_disparities(self, feature_analyses: Dict[FacialFeature, FeatureAnalysis], explanations: List[Explanation]) -> DisparityScores:
        """Calculate overall disparity scores based on feature analyses and model performance."""
        if not feature_analyses:
            return DisparityScores()

        bias_scores = [analysis.bias_score for analysis in feature_analyses.values()]
        biasx_score = round(sum(bias_scores) / len(bias_scores), self.precision)
        equalized_odds_score = self._calculate_equalized_odds_score(explanations)

        return DisparityScores(biasx=biasx_score, equalized_odds=equalized_odds_score)

    def _calculate_equalized_odds_score(self, explanations: List[Explanation]) -> float:
        """Calculate the equalized odds score."""
        y_true_female, y_pred_female = self._get_gender_predictions(explanations, Gender.FEMALE)
        y_true_male, y_pred_male = self._get_gender_predictions(explanations, Gender.MALE)

        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true_female, y_pred_female).ravel()
        tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_true_male, y_pred_male).ravel()

        tpr_female = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        fpr_female = fp_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0

        tpr_male = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
        fpr_male = fp_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0

        tpr_disparity = abs(tpr_female - tpr_male)
        fpr_disparity = abs(fpr_female - fpr_male)

        return round(max(tpr_disparity, fpr_disparity), self.precision)

    def _get_gender_predictions(self, explanations: List[Explanation], gender: Gender) -> Tuple[List[int], List[int]]:
        """Extract binary predictions and ground truth for a specific gender."""
        y_true = []
        y_pred = []

        for explanation in explanations:
            is_target_gender = explanation.image_data.gender == gender
            y_true.append(1 if is_target_gender else 0)
            y_pred.append(1 if explanation.predicted_gender == gender else 0)

        return y_true, y_pred

    def _get_feature_activation_map(self, explanations: List[Explanation]) -> Dict[FacialFeature, Dict[Gender, float]]:
        """Create mapping of features to activation probabilities by gender."""
        feature_activations = {feature: {Gender.MALE: 0, Gender.FEMALE: 0} for feature in FacialFeature}
        misclassified_counts = {Gender.MALE: 0, Gender.FEMALE: 0}

        for explanation in explanations:
            true_gender = explanation.image_data.gender
            predicted_gender = explanation.predicted_gender

            if true_gender != predicted_gender:
                misclassified_counts[true_gender] += 1
                activated_features = {box.feature for box in explanation.activation_boxes if box.feature}

                for feature in activated_features:
                    feature_activations[feature][true_gender] += 1

        # Calculate probabilities for each feature
        result = {}
        for feature, counts in feature_activations.items():
            probs = {}
            for gender in [Gender.MALE, Gender.FEMALE]:
                if misclassified_counts[gender] == 0:
                    probs[gender] = 0.0
                else:
                    probs[gender] = round(counts[gender] / misclassified_counts[gender], self.precision)

            if any(probs.values()):  # Only include features with activations
                result[feature] = probs

        return result
