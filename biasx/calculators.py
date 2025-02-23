import numpy as np

from .types import Explanation


class BiasCalculator:
    """Computes bias metrics from analysis results."""

    def __init__(self, ndigits: int):
        """Initialize the bias calculator."""
        self.ndigits = ndigits

    def compute_feature_probabilities(self, results: list[Explanation], feature: str) -> dict[int, float]:
        """Compute feature activation probabilities per class."""
        misclassified = [r for r in results if r.predicted_gender != r.true_gender]
        return {
            gender: (
                round(sum(1 for r in g_results for box in r.activation_boxes if box.feature == feature) / len(g_results), self.ndigits)
                if (g_results := [r for r in misclassified if r.true_gender == gender])
                else 0.0
            )
            for gender in (0, 1)
        }

    def compute_feature_bias(self, results: list[Explanation], feature: str) -> float:
        """Compute bias score for a specific feature."""
        probs = self.compute_feature_probabilities(results, feature)
        return round(abs(probs[1] - probs[0]), self.ndigits)

    def compute_overall_bias(self, results: list[Explanation], features: list[str]) -> float:
        """Compute overall bias score across all features."""
        return round(np.mean([self.compute_feature_bias(results, feature) for feature in features]), self.ndigits)
