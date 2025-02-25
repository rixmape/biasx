from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

from .types import ConfusionMatrix, Explanation, FairnessScores, Gender, ProcessedResults


class BiasCalculator:
    """
    Calculates various fairness and bias metrics for face classification models.

    This class implements standard algorithmic fairness metrics including equalized odds,
    demographic parity, disparate impact, and predictive parity. It also implements
    feature-specific bias metrics that measure how facial features contribute
    to classification disparities across gender groups.
    """

    def __init__(self, ndigits: int = 4):
        """
        Initialize the BiasCalculator with specified rounding precision.
        """
        self.ndigits = ndigits
        self._cache = {}  # Cache to avoid redundant calculations across methods

    def preprocess_results(self, results: list[Explanation]) -> ProcessedResults:
        """
        Organize results by gender and classification status for efficient processing.

        This method structures the classification results into categories that facilitate
        computation of fairness metrics, particularly separating misclassifications by gender.
        """
        # Return cached results if available
        results_id = id(results)
        if results_id in self._cache and "preprocessed" in self._cache[results_id]:
            return self._cache[results_id]["preprocessed"]

        # Initialize structure for processed results
        processed = {
            "by_gender": {0: [], 1: []},  # All results split by gender
            "misclassified": [],  # All misclassified instances
            "misclassified_by_gender": {0: [], 1: []},  # Misclassified instances by gender
            "total_count": len(results),  # Total number of evaluated instances
        }

        # Categorize each result
        for result in results:
            gender = result.true_gender
            is_misclassified = result.predicted_gender != gender

            # Add to gender-specific collections
            processed["by_gender"][gender].append(result)

            # Track misclassifications
            if is_misclassified:
                processed["misclassified"].append(result)
                processed["misclassified_by_gender"][gender].append(result)

        # Cache the processed results
        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["preprocessed"] = processed
        return processed

    def compute_confusion_matrix(self, results: list[Explanation]) -> ConfusionMatrix:
        """
        Calculate confusion matrix statistics (TP, FP, TN, FN) for each gender.

        For binary gender classification, this computes the counts of true positives (TP),
        false positives (FP), true negatives (TN), and false negatives (FN) for each gender.
        Note that a correct prediction for one gender contributes to both TP for that gender
        and TN for the other gender.
        """
        # Return cached matrix if available
        results_id = id(results)
        if results_id in self._cache and "confusion_matrix" in self._cache[results_id]:
            return self._cache[results_id]["confusion_matrix"]

        # Initialize confusion matrix for each gender
        matrix = {0: defaultdict(int), 1: defaultdict(int)}

        # Calculate confusion matrix elements
        for result in results:
            true_gender = result.true_gender
            pred_gender = result.predicted_gender

            # Correct prediction (true positive for true gender, true negative for other gender)
            if true_gender == pred_gender:
                matrix[true_gender]["TP"] += 1
                matrix[1 - true_gender]["TN"] += 1
            # Incorrect prediction (false negative for true gender, false positive for other gender)
            else:
                matrix[true_gender]["FN"] += 1
                matrix[1 - true_gender]["FP"] += 1

        # Cache the confusion matrix
        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["confusion_matrix"] = matrix
        return matrix

    def _get_feature_activation_map(self, results: list[Explanation]) -> Dict[str, Dict[int, int]]:
        """
        Create mapping of features to activation counts for misclassified instances by gender.

        This internal method identifies which facial features are present in misclassifications
        and counts their occurrences for each gender. A feature is counted once per instance
        even if it appears multiple times in the same face.

        Mathematical representation:
        Count(feature f | gender g, misclassified) = Number of misclassified instances of
        gender g where feature f is present
        """
        # Return cached feature map if available
        results_id = id(results)
        if results_id in self._cache and "feature_map" in self._cache[results_id]:
            return self._cache[results_id]["feature_map"]

        # Initialize feature activation map
        feature_map = defaultdict(lambda: {0: 0, 1: 0})
        processed = self.preprocess_results(results)

        # Count feature occurrences in misclassifications by gender
        for gender in Gender.__args__:
            for result in processed["misclassified_by_gender"][gender]:
                # Track features seen in this instance to avoid double-counting
                seen_features = set()

                # Check each activation box in the explanation
                for box in result.activation_boxes:
                    feature = box.feature
                    # Count feature once per misclassified instance
                    if feature not in seen_features:
                        feature_map[feature][gender] += 1
                        seen_features.add(feature)

        # Cache the feature activation map
        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["feature_map"] = feature_map
        return feature_map

    def compute_feature_probs(self, results: list[Explanation], feature: str) -> Dict[Gender, float]:
        """
        Calculate probability of feature presence in misclassifications for each gender.

        This method computes P(feature = present | gender = g, misclassified) for each gender,
        which represents the probability that a specific facial feature appears in
        misclassification cases across gender groups.

        Mathematical definition:
        P_f^g = Pr(X_f = 1 | A = g, Ŷ ≠ Y)
        """
        processed = self.preprocess_results(results)
        feature_map = self._get_feature_activation_map(results)
        probs = {}

        # Calculate probability for each gender
        for gender in Gender.__args__:
            misclassified_count = len(processed["misclassified_by_gender"][gender])

            # Handle case with no misclassifications for this gender
            if not misclassified_count:
                probs[gender] = 0.0
                continue

            # Calculate feature probability: P(feature = present | gender, misclassified)
            feature_count = feature_map.get(feature, {}).get(gender, 0)
            probs[gender] = round(feature_count / misclassified_count, self.ndigits)

        return probs

    def compute_feature_scores(self, results: list[Explanation], feature: str) -> float:
        """
        Measure bias in a specific facial feature by comparing its effect across genders.

        This method quantifies how differently a facial feature contributes to model errors
        across genders through feature-specific bias. A score of 0 indicates the feature
        contributes equally to errors across genders, while higher values indicate greater
        disparity in how the feature influences misclassifications between gender groups.

        Mathematical definition:
        B_f = |P_f^0 - P_f^1|
        """
        # Calculate feature probabilities for each gender
        probs = self.compute_feature_probs(results, feature)

        # Calculate absolute difference in probabilities between genders
        # This measures the disparity in how this feature affects misclassifications
        return round(abs(probs[1] - probs[0]), self.ndigits)

    def compute_overall_bias(self, results: list[Explanation], features: list[str]) -> float:
        """
        Calculate average bias across all specified facial features.

        This method provides a comprehensive metric to assess the model's overall fairness
        across all facial features by aggregating individual feature-specific bias measurements.
        A score of 0 indicates no bias across all features, and 1 indicates maximum bias.

        Mathematical definition:
        B̄ = (1/|F|) * ∑_{f∈F} B_f
        """
        # Calculate bias score for each feature
        feature_scores = [self.compute_feature_scores(results, feature) for feature in features]

        # Calculate arithmetic mean of feature-specific bias scores
        return round(np.mean(feature_scores), self.ndigits)

    def _compute_positive_rates(self, results: list[Explanation]) -> Dict[int, float]:
        """
        Determine rate of positive predictions for each gender group.

        This internal method calculates the probability of a subject being classified as
        gender 1 for each true gender group. Used for demographic parity and disparate impact.

        Mathematical definition:
        P(Ŷ = 1 | A = g) = [count of predictions for gender 1] / [total count for gender g]
        """
        # Return cached rates if available
        results_id = id(results)
        if results_id in self._cache and "positive_rates" in self._cache[results_id]:
            return self._cache[results_id]["positive_rates"]

        processed = self.preprocess_results(results)
        positive_rates = {}

        # Calculate positive rate for each gender
        for gender in Gender.__args__:
            gender_results = processed["by_gender"][gender]

            # Handle case with no instances of this gender
            if not gender_results:
                positive_rates[gender] = 0
                continue

            # Count instances predicted as gender 1 (positive class)
            positive_count = sum(1 for result in gender_results if result.predicted_gender == 1)

            # Calculate positive rate: P(Ŷ = 1 | A = g)
            positive_rates[gender] = positive_count / len(gender_results)

        # Cache the positive rates
        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["positive_rates"] = positive_rates
        return positive_rates

    def _compute_tpr_fpr(self, results: list[Explanation]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Calculate true positive and false positive rates for each gender.

        This internal method computes true positive rate (TPR) and false positive rate (FPR)
        for each gender, which are fundamental to several fairness metrics.

        Mathematical definitions:
        TPR_g = P(Ŷ = 1 | Y = 1, A = g) = TP / (TP + FN)
        FPR_g = P(Ŷ = 1 | Y = 0, A = g) = FP / (FP + TN)
        """
        # Return cached rates if available
        results_id = id(results)
        if results_id in self._cache and "tpr_fpr" in self._cache[results_id]:
            return self._cache[results_id]["tpr_fpr"]

        matrix = self.compute_confusion_matrix(results)
        tpr, fpr = {}, {}

        # Calculate TPR and FPR for each gender
        for gender in Gender.__args__:
            # Calculate true positive rate: TPR = TP / (TP + FN)
            tp, fn = matrix[gender]["TP"], matrix[gender]["FN"]
            denominator = tp + fn
            tpr[gender] = tp / denominator if denominator > 0 else 0

            # Calculate false positive rate: FPR = FP / (FP + TN)
            fp, tn = matrix[gender]["FP"], matrix[gender]["TN"]
            denominator = fp + tn
            fpr[gender] = fp / denominator if denominator > 0 else 0

        # Cache the rates
        if results_id not in self._cache:
            self._cache[results_id] = {}
        self._cache[results_id]["tpr_fpr"] = (tpr, fpr)
        return tpr, fpr

    def compute_equalized_odds_score(self, results: list[Explanation]) -> float:
        """
        Measure maximum disparity in true positive or false positive rates between genders.

        Equalized odds requires equal TPR and FPR across gender groups. This method calculates
        the maximum violation of this requirement, providing a score where 0 indicates perfect
        adherence to equalized odds and higher values indicate greater disparities.

        Mathematical definition:
        EOscore = max{|TPR_0 - TPR_1|, |FPR_0 - FPR_1|}
        """
        # Get true positive and false positive rates for each gender
        tpr, fpr = self._compute_tpr_fpr(results)

        # Calculate absolute differences in rates between genders
        tpr_diff = abs(tpr[0] - tpr[1])
        fpr_diff = abs(fpr[0] - fpr[1])

        # Equalized odds violation is the maximum disparity in either TPR or FPR
        return round(max(tpr_diff, fpr_diff), self.ndigits)

    def compute_demographic_parity_score(self, results: list[Explanation]) -> float:
        """
        Quantify difference in positive prediction rates between gender groups.

        Demographic parity (statistical parity) requires equal probability of positive
        prediction across protected groups, regardless of true class. This score measures
        the absolute difference in these probabilities, where 0 indicates perfect parity.

        Mathematical definition:
        DP = |P(Ŷ = 1 | A = 0) - P(Ŷ = 1 | A = 1)|
        """
        # Get positive prediction rates for each gender
        positive_rates = self._compute_positive_rates(results)

        # Calculate absolute difference in positive rates between genders
        return round(abs(positive_rates[0] - positive_rates[1]), self.ndigits)

    def compute_disparate_impact_score(self, results: list[Explanation]) -> float:
        """
        Calculate ratio of positive prediction rates between gender groups.

        Disparate impact measures the ratio of positive prediction rates between groups,
        taking the minimum of ratios to ensure the score is between 0 and 1. A score of 1
        indicates perfect parity, while 0 indicates maximum disparity.

        Mathematical definition:
        DI = min(P(Ŷ = 1 | A = 0) / P(Ŷ = 1 | A = 1), P(Ŷ = 1 | A = 1) / P(Ŷ = 1 | A = 0))
        """
        # Get positive prediction rates for each gender
        positive_rates = self._compute_positive_rates(results)

        # Handle cases with zero positive rates
        if positive_rates[0] == 0 and positive_rates[1] == 0:
            return 1.0  # No positive predictions for either group means parity
        if positive_rates[0] == 0 or positive_rates[1] == 0:
            return 0.0  # One group has zero positive predictions means maximum disparity

        # Calculate ratios between gender positive rates
        ratio_0_to_1 = positive_rates[0] / positive_rates[1]
        ratio_1_to_0 = positive_rates[1] / positive_rates[0]

        # Take minimum ratio to ensure score is between 0 and 1
        return round(min(ratio_0_to_1, ratio_1_to_0), self.ndigits)

    def compute_predictive_parity_score(self, results: list[Explanation]) -> float:
        """
        Measure difference in precision (PPV) between gender groups.

        Predictive parity requires equal precision (positive predictive value) across groups.
        This score quantifies the absolute difference in precision, where 0 indicates perfect
        parity and higher values indicate greater disparity.

        Mathematical definition:
        PP = |P(Y = 1 | Ŷ = 1, A = 0) - P(Y = 1 | Ŷ = 1, A = 1)|
        """
        # Get confusion matrix
        matrix = self.compute_confusion_matrix(results)
        ppv = {}

        # Calculate precision (PPV) for each gender
        for gender in Gender.__args__:
            tp, fp = matrix[gender]["TP"], matrix[gender]["FP"]
            denominator = tp + fp

            # Precision = TP / (TP + FP)
            ppv[gender] = tp / denominator if denominator > 0 else 0

        # Calculate absolute difference in precision between genders
        return round(abs(ppv[0] - ppv[1]), self.ndigits)

    def compute_equal_opportunity_score(self, results: list[Explanation]) -> float:
        """
        Quantify difference in true positive rates between gender groups.

        Equal opportunity requires equal true positive rates across protected groups.
        This score measures the absolute difference in these rates, where 0 indicates
        perfect equality of opportunity and higher values indicate greater disparity.

        Mathematical definition:
        EO = |P(Ŷ = 1 | Y = 1, A = 0) - P(Ŷ = 1 | Y = 1, A = 1)|
        """
        # Get true positive rates for each gender
        tpr, _ = self._compute_tpr_fpr(results)

        # Calculate absolute difference in TPR between genders
        return round(abs(tpr[0] - tpr[1]), self.ndigits)

    def compute_accuracy_parity_score(self, results: list[Explanation]) -> float:
        """
        Measure difference in overall accuracy between gender groups.

        Accuracy parity requires equal overall prediction accuracy across protected groups.
        This score quantifies the absolute difference in accuracy, where 0 indicates
        perfect parity and higher values indicate greater disparity.

        Mathematical definition:
        AP = |P(Ŷ = Y | A = 0) - P(Ŷ = Y | A = 1)|
        """
        processed = self.preprocess_results(results)
        accuracy = {}

        # Calculate accuracy for each gender
        for gender in Gender.__args__:
            gender_results = processed["by_gender"][gender]

            # Handle case with no instances of this gender
            if not gender_results:
                accuracy[gender] = 0
                continue

            # Count correct predictions for this gender
            correct = sum(1 for result in gender_results if result.true_gender == result.predicted_gender)

            # Calculate accuracy: P(Ŷ = Y | A = g)
            accuracy[gender] = correct / len(gender_results)

        # Calculate absolute difference in accuracy between genders
        return round(abs(accuracy[0] - accuracy[1]), self.ndigits)

    def compute_treatment_equality_score(self, results: list[Explanation], confusion_matrix: Optional[ConfusionMatrix] = None) -> float:
        """
        Calculate disparity in false negative to false positive ratios between genders.

        Treatment equality requires equal ratios of false negatives to false positives
        across protected groups. This score quantifies the normalized difference in these
        ratios, where 0 indicates perfect equality and higher values indicate disparity.

        Mathematical definition:
        TE = |(FN₀/FP₀ - FN₁/FP₁)| / (FN₀/FP₀ + FN₁/FP₁) where FN₀/FP₀ > FN₁/FP₁
        """
        # Use provided confusion matrix or compute it
        matrix = confusion_matrix if confusion_matrix else self.compute_confusion_matrix(results)
        fn_fp_ratio = {}

        # Calculate FN/FP ratio for each gender
        for gender in Gender.__args__:
            fn, fp = matrix[gender]["FN"], matrix[gender]["FP"]

            # Handle division by zero cases
            if fp == 0:
                # If both FN and FP are zero, this indicates no errors for this group
                if fn == 0:
                    fn_fp_ratio[gender] = 1.0  # No errors is equivalent to equal error rates
                else:
                    # Small epsilon to avoid division by zero while preserving large ratio
                    fn_fp_ratio[gender] = fn / 0.0001
            else:
                fn_fp_ratio[gender] = fn / fp

        # If both groups have the same ratio, perfect equality (0.0)
        if fn_fp_ratio[0] == fn_fp_ratio[1]:
            return 0.0

        # Using normalized absolute difference to create a 0-1 bounded metric
        max_ratio = max(fn_fp_ratio[0], fn_fp_ratio[1])
        min_ratio = min(fn_fp_ratio[0], fn_fp_ratio[1])

        # This creates a 0-1 bounded score where 0 is perfect equality
        normalized_diff = (max_ratio - min_ratio) / (max_ratio + min_ratio)
        return round(normalized_diff, self.ndigits)

    def compute_all_fairness_scores(self, results: list[Explanation], features: list[str]) -> FairnessScores:
        """
        Generate comprehensive report of all implemented fairness and bias metrics.

        This method calculates all fairness metrics implemented in the class, including
        traditional group fairness metrics and feature-based bias metrics. Results are
        returned as a dictionary for easy analysis and comparison.
        """
        # Pre-compute common values used across multiple metrics to improve efficiency
        _ = self.preprocess_results(results)
        confusion_matrix = self.compute_confusion_matrix(results)
        _ = self._compute_positive_rates(results)
        _ = self._compute_tpr_fpr(results)

        # Return dictionary containing all fairness metrics
        return {
            # Feature-based bias metric
            "overall_bias": self.compute_overall_bias(results, features),
            # Traditional group fairness metrics
            "equalized_odds": self.compute_equalized_odds_score(results),
            "demographic_parity": self.compute_demographic_parity_score(results),
            "disparate_impact": self.compute_disparate_impact_score(results),
            "predictive_parity": self.compute_predictive_parity_score(results),
            "equal_opportunity": self.compute_equal_opportunity_score(results),
            "accuracy_parity": self.compute_accuracy_parity_score(results),
            "treatment_equality": self.compute_treatment_equality_score(results, confusion_matrix),
        }
