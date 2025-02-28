"""Bias calculation module for BiasX."""

from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from .types import DisparityScores, Explanation, FacialFeature, FeatureAnalysis, Gender


class ConfusionMetrics:
    """Helper class for computing confusion matrix metrics."""

    @staticmethod
    def get_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[np.int64, np.int64, np.int64, np.int64]:
        """Get confusion matrix elements (TN, FP, FN, TP)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn, fp, fn, tp

    @staticmethod
    def tpr(tn: np.int64, fp: np.int64, fn: np.int64, tp: np.int64) -> float:
        """Calculate true positive rate (sensitivity/recall)."""
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def fpr(tn: np.int64, fp: np.int64, fn: np.int64, tp: np.int64) -> float:
        """Calculate false positive rate."""
        return fp / (tn + fp) if (tn + fp) > 0 else 0.0


class Calculator:
    """Calculates metrics related to bias in facial classification models."""

    def __init__(self, precision: int):
        """Initialize the bias calculator."""
        self.precision = precision
        self.metrics = ConfusionMetrics()

    def calculate_feature_biases(self, explanations: List[Explanation]) -> Dict[FacialFeature, FeatureAnalysis]:
        """Calculate bias metrics for each facial feature."""
        feature_analyses = {}
        feature_map = self._get_feature_activation_map(explanations)

        for feature in feature_map:
            if feature is None:
                continue

            probs = self._calculate_feature_probs(explanations, feature)

            bias_score = round(abs(probs[Gender.MALE] - probs[Gender.FEMALE]), self.precision)

            feature_analyses[feature] = FeatureAnalysis(
                feature=feature,
                bias_score=bias_score,
                male_probability=probs[Gender.MALE],
                female_probability=probs[Gender.FEMALE],
            )

        return feature_analyses

    def calculate_disparities(
        self,
        feature_analyses: Dict[FacialFeature, FeatureAnalysis],
        explanations: List[Explanation],
    ) -> DisparityScores:
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

        cm_female = self.metrics.get_confusion_matrix(y_true_female, y_pred_female)
        cm_male = self.metrics.get_confusion_matrix(y_true_male, y_pred_male)

        tpr_female = self.metrics.tpr(*cm_female)
        fpr_female = self.metrics.fpr(*cm_female)

        tpr_male = self.metrics.tpr(*cm_male)
        fpr_male = self.metrics.fpr(*cm_male)

        tpr_disparity = abs(tpr_female - tpr_male)
        fpr_disparity = abs(fpr_female - fpr_male)

        return round(max(tpr_disparity, fpr_disparity), self.precision)

    def _get_gender_predictions(self, explanations: List[Explanation], gender: Gender) -> Tuple[List[int], List[int]]:
        """Extract binary predictions and ground truth for a specific gender."""
        y_true = []
        y_pred = []

        for explanation in explanations:
            if explanation.image_data.gender == gender:
                y_true.append(1)
                y_pred.append(1 if explanation.predicted_gender == gender else 0)
            else:
                y_true.append(0)
                y_pred.append(1 if explanation.predicted_gender == gender else 0)

        return y_true, y_pred

    def _get_feature_activation_map(self, explanations: List[Explanation]) -> DefaultDict[FacialFeature, Dict[Gender, int]]:
        """Create mapping of features to activation counts by gender."""
        feature_map = defaultdict(lambda: {Gender.MALE: 0, Gender.FEMALE: 0})

        for explanation in explanations:
            true_gender = explanation.image_data.gender
            predicted_gender = explanation.predicted_gender

            if true_gender != predicted_gender:
                seen_features = set()

                for box in explanation.activation_boxes:
                    feature = box.feature
                    if feature and feature not in seen_features:
                        feature_map[feature][true_gender] += 1
                        seen_features.add(feature)

        return feature_map

    def _calculate_feature_probs(self, explanations: List[Explanation], feature: FacialFeature) -> Dict[Gender, float]:
        """Calculate probability of feature activation in misclassifications by gender."""
        misclassified_counts = {Gender.MALE: 0, Gender.FEMALE: 0}

        for explanation in explanations:
            true_gender = explanation.image_data.gender
            predicted_gender = explanation.predicted_gender

            if true_gender != predicted_gender:
                misclassified_counts[true_gender] += 1

        feature_map = self._get_feature_activation_map(explanations)
        feature_counts = feature_map.get(feature, {})

        probs = {}
        for gender in [Gender.MALE, Gender.FEMALE]:
            if misclassified_counts[gender] == 0:
                probs[gender] = 0.0
            else:
                probs[gender] = round(feature_counts.get(gender, 0) / misclassified_counts[gender], self.precision)

        return probs
