from collections import defaultdict

import numpy as np

# isort: off
from datatypes import FacialFeature, Gender
from utils import setup_logger


class BiasAnalyzer:
    """Class that computes bias metrics by analyzing feature probabilities, confusion matrices, and gender-based performance statistics."""

    def __init__(self, log_path: str):
        self.logger = setup_logger(name="bias_analyzer", log_path=log_path)

    def _compute_feature_metrics(self, labels: np.ndarray, features: list[list[FacialFeature]]) -> dict[str, dict[str, float]]:
        """Computes per-feature probabilities by aggregating counts from the provided feature boxes."""
        counts = defaultdict(lambda: defaultdict(int))
        for label, feature_boxes in zip(labels, features):
            for feature_box in feature_boxes:
                counts[feature_box.name][Gender(label).name] += 1

        self.logger.info(f"Computing occurence probabilities for {len(counts)} features")

        male_count = (labels == Gender.MALE.value).sum()
        female_count = (labels == Gender.FEMALE.value).sum()

        analysis = {}
        for feature_name, count in counts.items():
            male_prob = count[Gender.MALE.name] / max(male_count, 1)
            female_prob = count[Gender.FEMALE.name] / max(female_count, 1)
            bias = abs(male_prob - female_prob)
            analysis[feature_name] = {"male": male_prob, "female": female_prob, "bias": bias}

            self.logger.debug(f"Feature '{feature_name}': male_prob={male_prob:.3f}, female_prob={female_prob:.3f}, bias={bias:.3f}")

        return analysis

    def _compute_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, int]]:
        """Constructs a confusion matrix by comparing true labels with predictions for both male and female classes."""
        self.logger.info("Computing confusion matrix")

        male_actual = labels == Gender.MALE.value
        female_actual = labels == Gender.FEMALE.value
        male_pred = preds == Gender.MALE.value
        female_pred = preds == Gender.FEMALE.value

        tp = int((male_actual & male_pred).sum())
        fn = int((male_actual & female_pred).sum())
        fp = int((female_actual & male_pred).sum())
        tn = int((female_actual & female_pred).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / max(total, 1)

        self.logger.debug(f"Confusion matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        self.logger.debug(f"Overall accuracy: {accuracy:.3f} ({tp + tn}/{total} correct)")

        return {
            "true_male": {"predicted_male": tp, "predicted_female": fn},
            "true_female": {"predicted_male": fp, "predicted_female": tn},
        }

    def _compute_gender_metrics(self, cm: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Calculates gender-specific performance metrics such as TPR, FPR, PPV, and FDR using the confusion matrix data."""
        self.logger.info("Computing gender-specific performance metrics")

        tp = cm["true_male"]["predicted_male"]
        fn = cm["true_male"]["predicted_female"]
        fp = cm["true_female"]["predicted_male"]
        tn = cm["true_female"]["predicted_female"]

        tpr_male = tp / max(tp + fn, 1)
        fpr_male = fp / max(fp + tn, 1)
        ppv_male = tp / max(tp + fp, 1)
        fdr_male = fp / max(tp + fp, 1)

        tpr_female = tn / max(tn + fp, 1)
        fpr_female = fn / max(fn + tp, 1)
        ppv_female = tn / max(tn + fn, 1)
        fdr_female = fn / max(tn + fn, 1)

        self.logger.debug(f"Male metrics: TPR={tpr_male:.3f}, FPR={fpr_male:.3f}, PPV={ppv_male:.3f}, FDR={fdr_male:.3f}")
        self.logger.debug(f"Female metrics: TPR={tpr_female:.3f}, FPR={fpr_female:.3f}, PPV={ppv_female:.3f}, FDR={fdr_female:.3f}")

        return (
            {"tpr": tpr_male, "fpr": fpr_male, "ppv": ppv_male, "fdr": fdr_male},
            {"tpr": tpr_female, "fpr": fpr_female, "ppv": ppv_female, "fdr": fdr_female},
        )

    def _compute_bias_scores(self, cm: dict, male_metrics: dict, female_metrics: dict, feature_probs: dict[str, dict[str, float]]) -> dict[str, float]:
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
        self.logger.debug(f"Selection rates: male={male_selection_rate:.3f}, female={female_selection_rate:.3f}")

        demographic_parity = abs(male_selection_rate - female_selection_rate)
        self.logger.debug(f"Demographic parity difference: {demographic_parity:.3f}")

        # TODO: `tpr_diff` and `fpr_diff` are always the same. Just use one of them.
        tpr_diff = abs(male_metrics["tpr"] - female_metrics["tpr"])
        fpr_diff = abs(male_metrics["fpr"] - female_metrics["fpr"])
        equalized_odds = max(tpr_diff, fpr_diff)
        self.logger.debug(f"Equalized odds: {equalized_odds:.3f}")

        conditional_use_accuracy_equality = abs(male_metrics["ppv"] - female_metrics["ppv"])
        self.logger.debug(f"Conditional use accuracy equality: {conditional_use_accuracy_equality:.3f}")

        feature_attention = np.mean([feature["bias"] for feature in feature_probs.values()])
        self.logger.debug(f"Average feature attention bias: {feature_attention:.3f}")

        return {
            "demographic_parity": demographic_parity,
            "equalized_odds": equalized_odds,
            "conditional_use_accuracy_equality": conditional_use_accuracy_equality,
            "feature_attention": feature_attention,
        }

    def analyze(self, labels: np.ndarray, preds: np.ndarray, features: list[list[FacialFeature]]) -> dict:
        """Integrates feature metrics, confusion matrix, gender metrics, and bias scores to produce a comprehensive bias analysis."""
        self.logger.info(f"Starting bias analysis on {len(labels)} samples")

        feature_probs = self._compute_feature_metrics(labels, features)
        confusion_matrix = self._compute_confusion_matrix(labels, preds)
        male_metrics, female_metrics = self._compute_gender_metrics(confusion_matrix)

        bias_scores = self._compute_bias_scores(confusion_matrix, male_metrics, female_metrics, feature_probs)

        self.logger.info("Bias analysis completed successfully")

        return {
            "feature_probabilities": feature_probs,
            "confusion_matrix": confusion_matrix,
            "male_metrics": male_metrics,
            "female_metrics": female_metrics,
            "bias_scores": bias_scores,
        }
