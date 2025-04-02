import hashlib
import json
import os
import random
import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from analyzer import BiasAnalyzer
from config import Config
from dataset import DatasetGenerator
from datatypes import AnalysisResult, ArtifactSavingLevel, ExperimentResult, Gender, Explanation, ModelHistory
from explainer import VisualExplainer
from masker import FeatureMasker
from model import ModelTrainer
from utils import setup_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ExperimentRunner:
    """Manages the setup, execution, and analysis of a single bias analysis experiment."""

    def __init__(self, config: Optional[Config] = None):
        """Initializes the experiment runner with configuration."""
        self.config = config or Config()
        self.exp_id = self._generate_experiment_id()
        self.output_path = self._create_output_directory()
        self.log_path = self._create_log_directory()
        self.logger = setup_logger(name="experiment_runner", log_path=self.log_path, id=self.exp_id)

        self._set_random_seeds()

        self.feature_masker = FeatureMasker(self.config, self.log_path, self.exp_id)
        self.dataset_generator = DatasetGenerator(self.config, self.feature_masker, self.log_path, self.exp_id)
        self.model_trainer = ModelTrainer(self.config, self.log_path, self.exp_id)
        self.visual_explainer = VisualExplainer(self.config, self.feature_masker, self.log_path, self.exp_id)
        self.bias_analyzer = BiasAnalyzer(self.log_path, self.exp_id)

        self.logger.info(f"Completed experiment runner initialization")

    def _generate_experiment_id(self) -> str:
        """Generates a unique experiment ID based on a hash of the configuration."""
        config_json = self.config.model_dump_json()
        hash_object = hashlib.sha256(config_json.encode())
        return hash_object.hexdigest()[:16]

    def _create_output_directory(self) -> str:
        """Creates the main output directory for the experiment."""
        output_dir = os.path.join(self.config.output.base_dir, self.exp_id)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _create_log_directory(self) -> str:
        """Creates a timestamped log directory for the experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_path = os.path.join(self.config.output.log_dir, f"exp_{self.exp_id}_{timestamp}")
        os.makedirs(log_dir_path, exist_ok=True)
        return log_dir_path

    def _set_random_seeds(self) -> None:
        """Sets random seeds for reproducibility across libraries."""
        seed = self.config.core.random_seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _get_batch_explanations(
        self,
        batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        model: tf.keras.Model,
        heatmap_generator: GradcamPlusPlus,
    ) -> List[Explanation]:
        """Generates explanations (predictions, confidence, heatmaps, features) for a batch of images."""
        images, true_labels, image_ids = batch
        batch_size = images.shape[0]
        details = []

        raw_predictions = model.predict(images, verbose=0)
        predicted_labels = raw_predictions.argmax(axis=1)

        for i in range(batch_size):
            image_np = images[i].numpy()
            true_label = int(true_labels[i].numpy())
            image_id = image_ids[i].numpy().decode("utf-8")
            predicted_label = int(predicted_labels[i])
            confidence_scores = raw_predictions[i].tolist()

            detected_features, heatmap_path = self.visual_explainer.generate_explanation(
                heatmap_generator,
                model,
                image_np,
                true_label,
                image_id,
                self.output_path,
            )

            detail = Explanation(
                image_id=image_id,
                label=Gender(true_label),
                prediction=Gender(predicted_label),
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
        """Generates explanations for all images in the test dataset."""
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
        """Saves the experiment configuration, training history, and bias analysis results to a JSON file."""
        self.logger.info(f"Saving experiment results to JSON file")
        save_results = self.config.output.artifact_level in [
            ArtifactSavingLevel.RESULTS_ONLY,
            ArtifactSavingLevel.FULL,
        ]

        result = ExperimentResult(
            id=self.exp_id,
            config=self.config.model_dump(mode="json"),
            history=history if save_results else None,
            analysis=analysis if save_results else None,
        )

        filename = f"results_{self.exp_id}.json"
        path = os.path.join(self.output_path, filename)

        try:
            with open(path, "w") as f:
                json.dump(result.model_dump(mode="json"), f)
            self.logger.info(f"Saved results to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {path}: {e}", exc_info=True)

        return result

    def run_experiment(self) -> ExperimentResult:
        """Executes the full experiment pipeline: data prep, model training, explanation, and bias analysis."""
        self.logger.info(f"Starting experiment run")

        splits = self.dataset_generator.get_data_splits(self.config.core.random_seed, self.exp_id)
        train_data, val_data, test_data = splits

        model, history = self.model_trainer.get_model_and_history(train_data, val_data)
        explanations = self._get_all_explanations(test_data, model)
        analysis = self.bias_analyzer.get_bias_analysis(explanations)
        result = self._save_result(history, analysis)

        self.logger.info(f"Completed experiment run")
        return result
