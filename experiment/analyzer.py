from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix

# isort: off
from datatypes import FeatureBox, Gender
from utils import setup_logger


class BiasAnalyzer:
    """Class that computes bias metrics by analyzing feature probabilities, confusion matrices, and gender-based performance statistics."""

    def __init__(self, log_path: str):
        self.logger = setup_logger(name="bias_analyzer", log_path=log_path)

    def _compute_feature_metrics(self, labels: np.ndarray, features: list[list[FeatureBox]]) -> dict[str, dict[str, float]]:
        """Computes per-feature probabilities and overall feature bias by aggregating counts from the provided feature boxes."""
        self.logger.info("Computing feature occurence probabilities and attention bias")

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
            analysis[feature_name] = {"male": male_prob, "female": female_prob, "bias": bias}

            self.logger.debug(f"Feature '{feature_name}': male_prob={male_prob:.3f}, female_prob={female_prob:.3f}, bias={bias:.3f}")

            total_bias += bias
            feature_count += 1

        # TODO: Compute feature attention bias in `_compute_bias_scores`
        feature_attention_bias = total_bias / max(feature_count, 1)
        self.logger.info(f"Detected {feature_count} unique features with {feature_attention_bias:.3f} average attention bias")

        if feature_count == 0:
            self.logger.warning("No features were found during analysis. Potential issue during visual explanation.")

        return analysis, feature_attention_bias

    def _compute_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, int]]:
        """Constructs a confusion matrix by comparing true labels with predictions for both male and female classes."""
        self.logger.info("Constructing confusion matrix")

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

        self.logger.debug(f"Confusion matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        self.logger.debug(f"Overall accuracy: {accuracy:.3f} ({tp + tn}/{total} correct)")

        male_correct = tp / max(tp + fn, 1)
        female_correct = tn / max(tn + fp, 1)
        self.logger.debug(f"Gender-specific accuracy: male={male_correct:.3f} ({tp}/{tp+fn}), female={female_correct:.3f} ({tn}/{tn+fp})")

        return {
            "true_male": {"predicted_male": tp, "predicted_female": fn},
            "true_female": {"predicted_male": fp, "predicted_female": tn},
        }

    def _compute_gender_metrics(self, cm: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Calculates gender-specific performance metrics such as TPR, FPR, PPV, and FDR using the confusion matrix data."""
        self.logger.info("Computing gender-specific performance metrics")

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

        self.logger.debug(f"Male metrics: TPR={male_tpr:.3f}, FPR={male_fpr:.3f}, PPV={male_ppv:.3f}, FDR={male_fdr:.3f}")
        self.logger.debug(f"Female metrics: TPR={female_tpr:.3f}, FPR={female_fpr:.3f}, PPV={female_ppv:.3f}, FDR={female_fdr:.3f}")

        return (
            {"tpr": male_tpr, "fpr": male_fpr, "ppv": male_ppv, "fdr": male_fdr},
            {"tpr": female_tpr, "fpr": female_fpr, "ppv": female_ppv, "fdr": female_fdr},
        )

    def _compute_bias_scores(self, cm: dict, male_metrics: dict, female_metrics: dict) -> dict[str, float]:
        """Derives overall bias scores based on demographic parity, equalized odds, conditional use accuracy, treatment equality, and feature attention."""
        self.logger.info("Computing bias scores across multiple fairness criteria")

        tp_male = cm["true_male"]["predicted_male"]
        fn_male = cm["true_male"]["predicted_female"]
        fp_male = cm["true_female"]["predicted_male"]
        tn_male = cm["true_female"]["predicted_female"]

        male_count = tp_male + fn_male
        female_count = fp_male + tn_male

        male_selection_rate = tp_male / max(male_count, 1)
        female_selection_rate = fp_male / max(female_count, 1)
        demographic_parity = abs(male_selection_rate - female_selection_rate)

        self.logger.debug(f"Selection rates: male={male_selection_rate:.3f}, female={female_selection_rate:.3f}")
        self.logger.debug(f"Demographic parity difference: {demographic_parity:.3f}")

        tpr_diff = abs(male_metrics["tpr"] - female_metrics["tpr"])
        fpr_diff = abs(male_metrics["fpr"] - female_metrics["fpr"])
        equalized_odds = max(tpr_diff, fpr_diff)

        self.logger.debug(f"TPR difference: {tpr_diff:.3f}, FPR difference: {fpr_diff:.3f}")
        self.logger.debug(f"Equalized odds: {equalized_odds:.3f}")

        conditional_use_accuracy_equality = abs(male_metrics["ppv"] - female_metrics["ppv"])
        self.logger.debug(f"Conditional use accuracy equality: {conditional_use_accuracy_equality:.3f}")

        return {
            "demographic_parity": demographic_parity,
            "equalized_odds": equalized_odds,
            "conditional_use_accuracy_equality": conditional_use_accuracy_equality,
        }

    def analyze(self, labels: np.ndarray, preds: np.ndarray, features: list[list[FeatureBox]]) -> dict:
        """Integrates feature metrics, confusion matrix, gender metrics, and bias scores to produce a comprehensive bias analysis."""
        self.logger.info(f"Starting bias analysis on {len(labels)} samples")

        feature_probs, feature_bias = self._compute_feature_metrics(labels, features)
        confusion_matrix = self._compute_confusion_matrix(labels, preds)
        male_metrics, female_metrics = self._compute_gender_metrics(confusion_matrix)

        bias_scores = self._compute_bias_scores(confusion_matrix, male_metrics, female_metrics)
        bias_scores["feature_attention"] = feature_bias

        self.logger.info("Bias analysis completed successfully")

        return {
            "feature_probabilities": feature_probs,
            "confusion_matrix": confusion_matrix,
            "male_metrics": male_metrics,
            "female_metrics": female_metrics,
            "bias_scores": bias_scores,
        }
