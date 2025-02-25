from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

from .types import ConfusionMatrix, Explanation, FairnessScores, Gender, ProcessedResults


class BiasCalculator:
    """Calculates various fairness and bias metrics for face classification models."""

    def __init__(self, ndigits: int = 4):
        self.ndigits = ndigits
        self._cache = {}

    def preprocess_results(self, results: list[Explanation]) -> ProcessedResults:
        """Organize results by gender and classification status for efficient processing."""
        results_id = id(results)
        if results_id in self._cache and "preprocessed" in self._cache[results_id]:
            return self._cache[results_id]["preprocessed"]

        processed = {"by_gender": {0: [], 1: []}, "misclassified": [], "misclassified_by_gender": {0: [], 1: []}, "total_count": len(results)}

        for result in results:
            gender = result.true_gender
            is_misclassified = result.predicted_gender != gender
            processed["by_gender"][gender].append(result)
            if is_misclassified:
                processed["misclassified"].append(result)
                processed["misclassified_by_gender"][gender].append(result)

        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["preprocessed"] = processed
        return processed

    def compute_confusion_matrix(self, results: list[Explanation]) -> ConfusionMatrix:
        """Calculate confusion matrix statistics (TP, FP, TN, FN) for each gender."""
        results_id = id(results)
        if results_id in self._cache and "confusion_matrix" in self._cache[results_id]:
            return self._cache[results_id]["confusion_matrix"]

        matrix = {0: defaultdict(int), 1: defaultdict(int)}
        for result in results:
            true_gender = result.true_gender
            pred_gender = result.predicted_gender
            if true_gender == pred_gender:
                matrix[true_gender]["TP"] += 1
                matrix[1 - true_gender]["TN"] += 1
            else:
                matrix[true_gender]["FN"] += 1
                matrix[1 - true_gender]["FP"] += 1

        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["confusion_matrix"] = matrix
        return matrix

    def _get_feature_activation_map(self, results: list[Explanation]) -> Dict[str, Dict[int, int]]:
        """Create mapping of features to activation counts for misclassified instances by gender."""
        results_id = id(results)
        if results_id in self._cache and "feature_map" in self._cache[results_id]:
            return self._cache[results_id]["feature_map"]

        feature_map = defaultdict(lambda: {0: 0, 1: 0})
        processed = self.preprocess_results(results)

        for gender in (0, 1):
            for result in processed["misclassified_by_gender"][gender]:
                seen_features = set()
                for box in result.activation_boxes:
                    feature = box.feature
                    if feature not in seen_features:
                        feature_map[feature][gender] += 1
                        seen_features.add(feature)

        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["feature_map"] = feature_map
        return feature_map

    def compute_feature_probs(self, results: list[Explanation], feature: str) -> Dict[Gender, float]:
        """Calculate probability of feature presence in misclassifications for each gender."""
        processed = self.preprocess_results(results)
        feature_map = self._get_feature_activation_map(results)
        probs = {}

        for gender in (0, 1):
            misclassified_count = len(processed["misclassified_by_gender"][gender])
            if not misclassified_count:
                probs[gender] = 0.0
                continue
            feature_count = feature_map.get(feature, {}).get(gender, 0)
            probs[gender] = round(feature_count / misclassified_count, self.ndigits)
        return probs

    def compute_feature_scores(self, results: list[Explanation], feature: str) -> float:
        """Measure bias in a specific facial feature by comparing its effect across genders."""
        probs = self.compute_feature_probs(results, feature)
        return round(abs(probs[1] - probs[0]), self.ndigits)

    def compute_overall_bias(self, results: list[Explanation], features: list[str]) -> float:
        """Calculate average bias across all specified facial features."""
        feature_scores = [self.compute_feature_scores(results, feature) for feature in features]
        return round(np.mean(feature_scores), self.ndigits)

    def _compute_positive_rates(self, results: list[Explanation]) -> Dict[int, float]:
        """Determine rate of positive predictions for each gender group."""
        results_id = id(results)
        if results_id in self._cache and "positive_rates" in self._cache[results_id]:
            return self._cache[results_id]["positive_rates"]

        processed = self.preprocess_results(results)
        positive_rates = {}

        for gender in (0, 1):
            gender_results = processed["by_gender"][gender]
            if not gender_results:
                positive_rates[gender] = 0
                continue
            positive_count = sum(1 for result in gender_results if result.predicted_gender == 1)
            positive_rates[gender] = positive_count / len(gender_results)

        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["positive_rates"] = positive_rates
        return positive_rates

    def _compute_tpr_fpr(self, results: list[Explanation]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Calculate true positive and false positive rates for each gender."""
        results_id = id(results)
        if results_id in self._cache and "tpr_fpr" in self._cache[results_id]:
            return self._cache[results_id]["tpr_fpr"]

        matrix = self.compute_confusion_matrix(results)
        tpr, fpr = {}, {}

        for gender in (0, 1):
            tp, fn = matrix[gender]["TP"], matrix[gender]["FN"]
            denominator = tp + fn
            tpr[gender] = tp / denominator if denominator > 0 else 0

            fp, tn = matrix[gender]["FP"], matrix[gender]["TN"]
            denominator = fp + tn
            fpr[gender] = fp / denominator if denominator > 0 else 0

        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["tpr_fpr"] = (tpr, fpr)
        return tpr, fpr

    def compute_equalized_odds_score(self, results: list[Explanation]) -> float:
        """Measure maximum disparity in true positive or false positive rates between genders."""
        tpr, fpr = self._compute_tpr_fpr(results)
        tpr_diff = abs(tpr[0] - tpr[1])
        fpr_diff = abs(fpr[0] - fpr[1])
        return round(max(tpr_diff, fpr_diff), self.ndigits)

    # FIXME: Similar value with compute_equalized_odds_score
    def compute_demographic_parity_score(self, results: list[Explanation]) -> float:
        """Quantify difference in positive prediction rates between gender groups."""
        positive_rates = self._compute_positive_rates(results)
        return round(abs(positive_rates[0] - positive_rates[1]), self.ndigits)

    def compute_disparate_impact_score(self, results: list[Explanation]) -> float:
        """Calculate ratio of positive prediction rates between gender groups."""
        positive_rates = self._compute_positive_rates(results)
        if positive_rates[0] == 0 and positive_rates[1] == 0:
            return 1.0
        if positive_rates[0] == 0 or positive_rates[1] == 0:
            return 0.0
        ratio_0_to_1 = positive_rates[0] / positive_rates[1]
        ratio_1_to_0 = positive_rates[1] / positive_rates[0]
        return round(min(ratio_0_to_1, ratio_1_to_0), self.ndigits)

    def compute_predictive_parity_score(self, results: list[Explanation]) -> float:
        """Measure difference in precision (PPV) between gender groups."""
        matrix = self.compute_confusion_matrix(results)
        ppv = {}
        for gender in (0, 1):
            tp, fp = matrix[gender]["TP"], matrix[gender]["FP"]
            denominator = tp + fp
            ppv[gender] = tp / denominator if denominator > 0 else 0
        return round(abs(ppv[0] - ppv[1]), self.ndigits)

    # FIXME: Similar value with compute_equalized_odds_score
    def compute_equal_opportunity_score(self, results: list[Explanation]) -> float:
        """Quantify difference in true positive rates between gender groups."""
        tpr, _ = self._compute_tpr_fpr(results)
        return round(abs(tpr[0] - tpr[1]), self.ndigits)

    # FIXME: Similar value with compute_equalized_odds_score
    def compute_accuracy_parity_score(self, results: list[Explanation]) -> float:
        """Measure difference in overall accuracy between gender groups."""
        processed = self.preprocess_results(results)
        accuracy = {}
        for gender in (0, 1):
            gender_results = processed["by_gender"][gender]
            if not gender_results:
                accuracy[gender] = 0
                continue
            correct = sum(1 for result in gender_results if result.true_gender == result.predicted_gender)
            accuracy[gender] = correct / len(gender_results)
        return round(abs(accuracy[0] - accuracy[1]), self.ndigits)

    def compute_treatment_equality_score(self, results: list[Explanation], confusion_matrix: Optional[ConfusionMatrix] = None) -> float:
        """Calculate disparity in false negative to false positive ratios between genders."""
        matrix = confusion_matrix if confusion_matrix else self.compute_confusion_matrix(results)
        fn_fp_ratio = {}
        for gender in (0, 1):
            fn, fp = matrix[gender]["FN"], matrix[gender]["FP"]
            if fp == 0:
                fn_fp_ratio[gender] = 1.0 if fn == 0 else float("inf")
            else:
                fn_fp_ratio[gender] = fn / fp

        if fn_fp_ratio[0] == float("inf") and fn_fp_ratio[1] == float("inf"):
            return 0.0
        if fn_fp_ratio[0] == float("inf") or fn_fp_ratio[1] == float("inf"):
            return 1.0

        ratio = fn_fp_ratio[0] / fn_fp_ratio[1] if fn_fp_ratio[0] > fn_fp_ratio[1] else fn_fp_ratio[1] / fn_fp_ratio[0]
        return round(ratio - 1, self.ndigits)

    def compute_all_fairness_scores(self, results: list[Explanation], features: list[str]) -> FairnessScores:
        """Generate comprehensive report of all implemented fairness and bias metrics."""
        _ = self.preprocess_results(results)
        confusion_matrix = self.compute_confusion_matrix(results)
        _ = self._compute_positive_rates(results)
        _ = self._compute_tpr_fpr(results)

        return {
            "overall_bias": self.compute_overall_bias(results, features),
            "equalized_odds": self.compute_equalized_odds_score(results),
            "demographic_parity": self.compute_demographic_parity_score(results),
            "disparate_impact": self.compute_disparate_impact_score(results),
            "predictive_parity": self.compute_predictive_parity_score(results),
            "equal_opportunity": self.compute_equal_opportunity_score(results),
            "accuracy_parity": self.compute_accuracy_parity_score(results),
            "treatment_equality": self.compute_treatment_equality_score(results, confusion_matrix),
        }
