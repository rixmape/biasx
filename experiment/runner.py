import json
import os
import random
import warnings
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from analyzer import BiasAnalyzer
from config import Config
from dataset import DatasetGenerator
from datatypes import AnalysisResult, OutputLevel, ExperimentResult, Gender, Explanation, ModelHistory
from explainer import VisualExplainer
from masker import FeatureMasker
from model import ModelTrainer
from utils import create_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ExperimentRunner:
    """Manages the setup, execution, and analysis of a single bias analysis experiment."""

    def __init__(self, config: Config):
        """Initializes the experiment runner with configuration and sets up components."""
        self.config = config
        self.logger = create_logger(config)
        self._set_random_seeds()

        self.feature_masker = FeatureMasker(self.config, self.logger)
        self.dataset_generator = DatasetGenerator(self.config, self.logger, self.feature_masker)
        self.model_trainer = ModelTrainer(self.config, self.logger)
        self.visual_explainer = VisualExplainer(self.config, self.logger, self.feature_masker)
        self.bias_analyzer = BiasAnalyzer(self.config, self.logger)

        self.logger.info(f"Completed experiment runner initialization: id={self.config.experiment_id}")

    def _set_random_seeds(self) -> None:
        """Sets random seeds for Python, NumPy, and TensorFlow to ensure reproducibility."""
        seed = self.config.core.random_seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _get_batch_explanations(
        self,
        batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        model: tf.keras.Model,
        heatmap_generator: GradcamPlusPlus,
    ) -> List[Explanation]:
        """Generates detailed explanations for a single batch of data from the test set."""
        images, true_labels, image_ids, races, ages = batch
        batch_size = images.shape[0]
        details = []

        raw_predictions = model.predict(images, verbose=0)
        predicted_labels = raw_predictions.argmax(axis=1)

        for i in range(batch_size):
            image_np = images[i].numpy()
            true_label = int(true_labels[i].numpy())
            image_id = image_ids[i].numpy().decode("utf-8")
            race = races[i].numpy().decode("utf-8")
            age = int(ages[i].numpy())
            predicted_label = int(predicted_labels[i])
            confidence_scores = raw_predictions[i].tolist()

            detected_features, heatmap_path = self.visual_explainer.generate_explanation(
                heatmap_generator,
                model,
                image_np,
                true_label,
                image_id,
            )

            detail = Explanation(
                image_id=image_id,
                label=Gender(true_label),
                prediction=Gender(predicted_label),
                race=race,
                age=age,
                confidence_scores=confidence_scores,
                heatmap_path=heatmap_path,
                detected_features=detected_features,
            )
            details.append(detail)

        return details

    def _get_all_explanations(
        self,
        test_data: tf.data.Dataset,
        model: tf.keras.Model,
    ) -> List[Explanation]:
        """Generates explanations for all samples in the provided test dataset."""
        self.logger.info(f"Processing test data for explanations")

        heatmap_generator = self.visual_explainer.get_heatmap_generator(model)
        all_explanations = []
        processed = 0

        for batch in test_data:
            batch_explanations = self._get_batch_explanations(batch, model, heatmap_generator)
            all_explanations.extend(batch_explanations)
            processed += len(batch_explanations)

            if processed % (self.config.model.batch_size * 5) == 0:
                self.logger.info(f"Processed {processed} test images")

        self.logger.info(f"Completed processing {processed} test images")
        return all_explanations

    def _save_result(
        self,
        history: ModelHistory,
        analysis: AnalysisResult,
    ) -> ExperimentResult:
        """Saves the comprehensive experiment results (config, history, analysis) to a JSON file."""
        self.logger.info(f"Saving experiment results to JSON file")
        save_results = self.config.output.level in [
            OutputLevel.RESULTS_ONLY,
            OutputLevel.FULL,
        ]

        result = ExperimentResult(
            id=self.config.experiment_id,
            config=self.config.model_dump(mode="json"),
            history=history if save_results else None,
            analysis=analysis if save_results else None,
        )

        filename = f"{self.config.experiment_id}.json"
        path = os.path.join(self.config.output.base_path, filename)

        try:
            with open(path, "w") as f:
                json.dump(result.model_dump(mode="json"), f)
            self.logger.info(f"Saved results to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {path}: {e}", exc_info=True)

        return result

    def run_experiment(self) -> ExperimentResult:
        """Executes the full end-to-end experiment pipeline and returns the final results."""
        self.logger.info(f"Starting experiment run")

        splits = self.dataset_generator.prepare_datasets(self.config.core.random_seed)
        train_data, val_data, test_data = splits

        model, history = self.model_trainer.get_model_and_history(train_data, val_data)
        explanations = self._get_all_explanations(test_data, model)
        analysis = self.bias_analyzer.get_bias_analysis(explanations)
        result = self._save_result(history, analysis)

        self.logger.info(f"Completed experiment run")
        return result
