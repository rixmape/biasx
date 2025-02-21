from typing import Any, Optional

import numpy as np


class BiasCalculator:
    """Computes bias metrics from analysis results."""

    def __init__(self, ndigits: Optional[int] = 3):
        """
        Initialize the bias calculator.

        Args:
            ndigits: Number of digits to round bias scores to
        """
        self.ndigits = ndigits

    def compute_feature_probabilities(self, results: list[dict[str, Any]], feature: str) -> dict[int, float]:
        """Compute feature activation probabilities per class."""
        misclassified = [r for r in results if r["predictedGender"] != r["trueGender"]]
        if not misclassified:
            return {0: 0.0, 1: 0.0}
        probs = {}
        for gender in (0, 1):
            gender_results = [r for r in misclassified if r["trueGender"] == gender]
            if not gender_results:
                probs[gender] = 0.0
                continue
            feature_count = sum(1 for r in gender_results for box in r["activationBoxes"] if box.feature == feature)
            probs[gender] = round(feature_count / len(gender_results), self.ndigits)
        return probs

    def compute_feature_bias(self, results: list[dict[str, Any]], feature: str) -> float:
        """Compute bias score for a specific feature."""
        probs = self.compute_feature_probabilities(results, feature)
        return round(abs(probs[1] - probs[0]), self.ndigits)

    def compute_overall_bias(self, results: list[dict[str, Any]], features: list[str]) -> float:
        """Compute overall bias score across all features."""
        scores = [self.compute_feature_bias(results, feature) for feature in features]
        return round(np.mean(scores), self.ndigits)
