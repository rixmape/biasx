from collections import defaultdict

import numpy as np

# isort: off
from datatypes import FeatureBox, Gender
from utils import setup_logger

logger = setup_logger(name="experiment.analyzer")


class BiasAnalyzer:
    """Class that computes bias metrics by analyzing feature probabilities, confusion matrices, and gender-based performance statistics."""

    def __init__(self):
        logger.info("Initializing BiasAnalyzer")
        logger.debug("BiasAnalyzer will compute feature metrics, construct confusion matrices, and calculate gender-specific bias scores")

    def _compute_feature_metrics(self, labels: np.ndarray, features: list[list[FeatureBox]]) -> dict[str, dict[str, float]]:
        """Computes per-feature probabilities and overall feature bias by aggregating counts from the provided feature boxes."""
        logger.info("Computing feature importance metrics across gender groups")

        counts = defaultdict(lambda: defaultdict(int))
        for label, feature_boxes in zip(labels, features):
            for feature_box in feature_boxes:
                counts[feature_box.name][Gender(label).name] += 1

        male_count = (labels == Gender.MALE.value).sum()
        female_count = (labels == Gender.FEMALE.value).sum()

        analysis = {}
        total_bias = 0
        feature_count = 0

        for feature_name, count in counts.items():
            male_prob = count[Gender.MALE.name] / max(male_count, 1)
            female_prob = count[Gender.FEMALE.name] / max(female_count, 1)
            bias = abs(male_prob - female_prob)
            feature_count += 1

            logger.debug(f"Feature '{feature_name}': male_prob={male_prob:.3f}, female_prob={female_prob:.3f}, bias={bias:.3f}")

            analysis[feature_name] = {
                "male": male_prob,
                "female": female_prob,
                "bias": bias,
            }

            total_bias += bias

        feature_bias = total_bias / max(feature_count, 1)
        logger.info(f"Feature analysis complete: {feature_count} unique features identified with average bias score of {feature_bias:.3f}")

        if feature_count == 0:
            logger.warning("No features were found in the analysis. This may indicate issues with feature extraction or importance thresholding.")
        elif feature_bias > 0.3:
            logger.warning(f"High average feature bias detected ({feature_bias:.3f}). This suggests significant gender-based differences in feature importance.")

        return analysis, feature_bias

    def _compute_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, int]]:
        """Constructs a confusion matrix by comparing true labels with predictions for both male and female classes."""
        logger.info("Constructing gender-based confusion matrix")

        male_actual = labels == Gender.MALE.value
        female_actual = labels == Gender.FEMALE.value
        male_pred = preds == Gender.MALE.value
        female_pred = preds == Gender.FEMALE.value

        tp = int((male_actual & male_pred).sum())
        fn = int((male_actual & female_pred).sum())
        fp = int((female_actual & male_pred).sum())
        tn = int((female_actual & female_pred).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0

        logger.debug(f"Confusion matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        logger.debug(f"Overall accuracy: {accuracy:.3f} ({tp + tn}/{total} correct)")

        male_correct = tp / (tp + fn) if (tp + fn) > 0 else 0
        female_correct = tn / (tn + fp) if (tn + fp) > 0 else 0
        logger.debug(f"Class-wise accuracy: Male={male_correct:.3f} ({tp}/{tp+fn}), Female={female_correct:.3f} ({tn}/{tn+fp})")

        if abs(male_correct - female_correct) > 0.1:
            logger.warning(f"Significant accuracy gap between genders: Male={male_correct:.3f}, Female={female_correct:.3f}. This suggests potential bias in model performance.")

        return {
            "true_male": {
                "predicted_male": tp,
                "predicted_female": fn,
            },
            "true_female": {
                "predicted_male": fp,
                "predicted_female": tn,
            },
        }

    def _compute_gender_metrics(self, cm: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Calculates gender-specific performance metrics such as TPR, FPR, PPV, and FDR using the confusion matrix data."""
        logger.info("Computing gender-specific fairness metrics")

        tp_male = cm["true_male"]["predicted_male"]
        fn_male = cm["true_male"]["predicted_female"]
        fp_male = cm["true_female"]["predicted_male"]
        tn_male = cm["true_female"]["predicted_female"]

        male_tpr = tp_male / max(tp_male + fn_male, 1)
        male_fpr = fp_male / max(fp_male + tn_male, 1)
        male_ppv = tp_male / max(tp_male + fp_male, 1)
        male_fdr = fp_male / max(tp_male + fp_male, 1)

        female_tpr = tn_male / max(tn_male + fp_male, 1)
        female_fpr = fn_male / max(fn_male + tp_male, 1)
        female_ppv = tn_male / max(tn_male + fn_male, 1)
        female_fdr = fn_male / max(tn_male + fn_male, 1)

        logger.debug(f"Male metrics: TPR={male_tpr:.3f}, FPR={male_fpr:.3f}, PPV={male_ppv:.3f}, FDR={male_fdr:.3f}")
        logger.debug(f"Female metrics: TPR={female_tpr:.3f}, FPR={female_fpr:.3f}, PPV={female_ppv:.3f}, FDR={female_fdr:.3f}")

        tpr_gap = abs(male_tpr - female_tpr)
        fpr_gap = abs(male_fpr - female_fpr)
        ppv_gap = abs(male_ppv - female_ppv)

        if tpr_gap > 0.1:
            logger.warning(f"Large true positive rate gap between genders: {tpr_gap:.3f}. The model has different sensitivity for different genders.")

        if fpr_gap > 0.1:
            logger.warning(f"Large false positive rate gap between genders: {fpr_gap:.3f}. The model has different specificity for different genders.")

        if ppv_gap > 0.1:
            logger.warning(f"Large precision gap between genders: {ppv_gap:.3f}. The model has different reliability for different genders.")

        return (
            {
                "tpr": male_tpr,
                "fpr": male_fpr,
                "ppv": male_ppv,
                "fdr": male_fdr,
            },
            {
                "tpr": female_tpr,
                "fpr": female_fpr,
                "ppv": female_ppv,
                "fdr": female_fdr,
            },
        )

    def _compute_bias_scores(self, cm: dict, male_metrics: dict, female_metrics: dict, feature_bias: float) -> dict[str, float]:
        """Derives overall bias scores based on demographic parity, equalized odds, conditional use accuracy, treatment equality, and feature attention."""
        logger.info("Computing composite bias scores across multiple fairness criteria")

        tp_male = cm["true_male"]["predicted_male"]
        fn_male = cm["true_male"]["predicted_female"]
        fp_male = cm["true_female"]["predicted_male"]
        tn_male = cm["true_female"]["predicted_female"]

        male_count = tp_male + fn_male
        female_count = fp_male + tn_male

        male_selection_rate = tp_male / max(male_count, 1)
        female_selection_rate = fp_male / max(female_count, 1)
        demographic_parity = abs(male_selection_rate - female_selection_rate)

        logger.debug(f"Selection rates: Male={male_selection_rate:.3f}, Female={female_selection_rate:.3f}")
        logger.debug(f"Demographic parity difference: {demographic_parity:.3f}")

        tpr_diff = abs(male_metrics["tpr"] - female_metrics["tpr"])
        fpr_diff = abs(male_metrics["fpr"] - female_metrics["fpr"])
        equalized_odds = max(tpr_diff, fpr_diff)

        logger.debug(f"TPR difference: {tpr_diff:.3f}, FPR difference: {fpr_diff:.3f}")
        logger.debug(f"Equalized odds: {equalized_odds:.3f} (maximum of TPR and FPR differences)")

        # TODO: Clarify that CUAE requires equal PPV and NPV, but since male NPV is similar to female PPV and vice versa, we can just use PPV
        ppv_diff = abs(male_metrics["ppv"] - female_metrics["ppv"])
        conditional_use_accuracy_equality = ppv_diff

        logger.debug(f"Conditional use accuracy equality: {conditional_use_accuracy_equality:.3f}")
        logger.debug(f"Feature attention bias: {feature_bias:.3f}")

        bias_scores = {
            "demographic_parity": demographic_parity,
            "equalized_odds": equalized_odds,
            "conditional_use_accuracy_equality": conditional_use_accuracy_equality,
            "feature_attention": feature_bias,
        }

        average_bias = sum(bias_scores.values()) / len(bias_scores)
        logger.info(f"Overall average bias score: {average_bias:.3f}")

        if average_bias > 0.2:
            logger.warning(f"High overall bias detected ({average_bias:.3f}). Multiple fairness metrics suggest problematic gender disparities.")

        sorted_biases = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Fairness criteria ordered by bias magnitude: {sorted_biases}")

        return bias_scores

    def analyze(self, labels: np.ndarray, preds: np.ndarray, features: list[list[FeatureBox]]) -> dict:
        """Integrates feature metrics, confusion matrix, gender metrics, and bias scores to produce a comprehensive bias analysis."""
        logger.info(f"Starting comprehensive bias analysis on dataset with {len(labels)} samples")

        if len(labels) == 0:
            logger.error("Empty dataset provided for bias analysis")
            return {}

        if len(labels) != len(preds) or len(labels) != len(features):
            logger.error(f"Dimension mismatch: labels={len(labels)}, predictions={len(preds)}, features={len(features)}")
            return {}

        feature_probs, feature_bias = self._compute_feature_metrics(labels, features)
        confusion_matrix = self._compute_confusion_matrix(labels, preds)
        male_metrics, female_metrics = self._compute_gender_metrics(confusion_matrix)
        bias_scores = self._compute_bias_scores(confusion_matrix, male_metrics, female_metrics, feature_bias)

        logger.info("Bias analysis completed successfully")

        male_perf = male_metrics["tpr"]
        female_perf = female_metrics["tpr"]
        overall_accuracy = (confusion_matrix["true_male"]["predicted_male"] + confusion_matrix["true_female"]["predicted_female"]) / len(labels)

        logger.debug(f"Summary statistics - Overall accuracy: {overall_accuracy:.3f}, Male performance: {male_perf:.3f}, Female performance: {female_perf:.3f}")

        max_bias_metric = max(bias_scores.items(), key=lambda x: x[1])
        logger.debug(f"Highest bias detected in: {max_bias_metric[0]} = {max_bias_metric[1]:.3f}")

        return {
            "feature_probabilities": feature_probs,
            "confusion_matrix": confusion_matrix,
            "male_metrics": male_metrics,
            "female_metrics": female_metrics,
            "bias_scores": bias_scores,
        }
