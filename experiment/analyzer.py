from collections import defaultdict
from typing import List

import numpy as np

# isort: off
from datatypes import AnalysisResult, BiasMetrics, Feature, FeatureDistribution, Gender, GenderPerformanceMetrics, Explanation
from utils import setup_logger


class BiasAnalyzer:

    def __init__(self, log_path: str, exp_id: str):
        self.logger = setup_logger(name="bias_analyzer", log_path=log_path, id=exp_id)
        self.logger.info("Completed bias analyzer initialization")

    def _compute_feature_distributions(
        self,
        image_details: List[Explanation],
    ) -> List[FeatureDistribution]:
        self.logger.info("Computing feature distributions based on key features.")
        feature_counts = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)

        for image_detail in image_details:
            true_label = image_detail.label
            gender_name = true_label.name
            label_counts[gender_name] += 1

            for feature_detail in image_detail.detected_features:
                if feature_detail.is_key_feature:
                    feature_counts[feature_detail.feature][gender_name] += 1

        male_total = label_counts[Gender.MALE.name]
        female_total = label_counts[Gender.FEMALE.name]
        distributions = []
        all_possible_features = set(Feature)

        for feature_enum in all_possible_features:
            gender_count = feature_counts[feature_enum]
            male_prob = gender_count[Gender.MALE.name] / max(male_total, 1)
            female_prob = gender_count[Gender.FEMALE.name] / max(female_total, 1)
            dist = FeatureDistribution(
                feature=feature_enum,
                male_distribution=male_prob,
                female_distribution=female_prob,
            )
            distributions.append(dist)

        return distributions

    def _compute_gender_performance_metrics(
        self,
        positive_class: Gender,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> GenderPerformanceMetrics:
        self.logger.info(f"Computing performance metrics for {positive_class.name} as Positive.")

        positive_val = positive_class.value

        is_positive_actual = true_labels == positive_val
        is_negative_actual = true_labels != positive_val
        is_positive_pred = predicted_labels == positive_val
        is_negative_pred = predicted_labels != positive_val

        tp = int(np.sum(is_positive_actual & is_positive_pred))
        fn = int(np.sum(is_positive_actual & is_negative_pred))
        fp = int(np.sum(is_negative_actual & is_positive_pred))
        tn = int(np.sum(is_negative_actual & is_negative_pred))

        metrics = GenderPerformanceMetrics(positive_class=positive_class, tp=tp, fp=fp, tn=tn, fn=fn)

        self.logger.debug(f"{positive_class.name} metrics: {metrics.model_dump()}")
        return metrics

    def _compute_bias_metrics(
        self,
        male_metrics: GenderPerformanceMetrics,
        female_metrics: GenderPerformanceMetrics,
        feature_distributions: List[FeatureDistribution],
    ) -> BiasMetrics:
        self.logger.info("Computing overall bias scores.")

        males_predicted_positive = male_metrics.tp + male_metrics.fp
        females_predicted_positive = female_metrics.tp + female_metrics.fp
        total_males = male_metrics.tp + male_metrics.fn
        total_females = female_metrics.tp + female_metrics.fn

        male_select_rate = males_predicted_positive / max(total_males, 1)
        female_select_rate = females_predicted_positive / max(total_females, 1)
        demographic_parity = abs(male_select_rate - female_select_rate)

        equalized_odds = abs(male_metrics.tpr - female_metrics.tpr)
        conditional_use_accuracy_equality = abs(male_metrics.ppv - female_metrics.ppv)
        mean_feature_distribution_bias = np.mean([dist.distribution_bias for dist in feature_distributions]) if feature_distributions else 0.0

        bias_metrics_result = BiasMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            conditional_use_accuracy_equality=conditional_use_accuracy_equality,
            mean_feature_distribution_bias=mean_feature_distribution_bias,
        )

        self.logger.debug(f"Overall bias metrics: {bias_metrics_result.model_dump()}")
        return bias_metrics_result

    def get_bias_analysis(
        self,
        image_details: List[Explanation],
    ) -> AnalysisResult:
        self.logger.info(f"Starting bias analysis on {len(image_details)} samples.")

        true_labels = np.array([img.label.value for img in image_details])
        predicted_labels = np.array([img.prediction.value for img in image_details])

        feature_distributions = self._compute_feature_distributions(image_details)
        male_perf_metrics = self._compute_gender_performance_metrics(Gender.MALE, true_labels, predicted_labels)
        female_perf_metrics = self._compute_gender_performance_metrics(Gender.FEMALE, true_labels, predicted_labels)
        bias_metrics = self._compute_bias_metrics(male_perf_metrics, female_perf_metrics, feature_distributions)

        self.logger.info("Bias analysis completed successfully.")

        return AnalysisResult(
            feature_distributions=feature_distributions,
            male_performance_metrics=male_perf_metrics,
            female_performance_metrics=female_perf_metrics,
            bias_metrics=bias_metrics,
            analyzed_images=image_details,
        )
