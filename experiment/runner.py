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
from config import ExperimentsConfig
from dataset import DatasetGenerator
from datatypes import ArtifactSavingLevel, ExperimentParameters, ExperimentResult, MaskDetails, Gender, ImageDetail, ReplicateResult, ModelTrainingHistory, AnalysisResult
from explainer import VisualExplainer
from masker import FeatureMasker
from model import ModelTrainer
from utils import setup_logger


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ExperimentRunner:

    def __init__(self, config: Optional[ExperimentsConfig] = None):
        self.config = config or ExperimentsConfig()
        self.exp_log_path = self._create_log_directory()
        self.logger = setup_logger(name="experiment_runner", log_path=self.exp_log_path)

        self.feature_masker = FeatureMasker(config=self.config, log_path=self.exp_log_path)
        self.dataset_generator = DatasetGenerator(config=self.config, feature_masker=self.feature_masker, log_path=self.exp_log_path)
        self.model_trainer = ModelTrainer(config=self.config, log_path=self.exp_log_path)
        self.visual_explainer = VisualExplainer(config=self.config, masker=self.feature_masker, log_path=self.exp_log_path)
        self.bias_analyzer = BiasAnalyzer(log_path=self.exp_log_path)

    def _create_log_directory(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.output.log_dir, f"experiment_{timestamp}")
        os.makedirs(path, exist_ok=True)
        return path

    def _set_random_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        self.logger.info(f"Random seeds set to {seed}.")

    def _initialize_replicate(
        self,
        replicate_index: int,
        exp_id: str,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
    ) -> Tuple[int, tf.keras.Model, ModelTrainingHistory]:
        self.logger.info(f"[{exp_id}] Starting replicate {replicate_index + 1}/{self.config.core.replicate_count}.")
        seed = self.config.core.base_random_seed + replicate_index
        self._set_random_seeds(seed)
        model, training_history = self.model_trainer.train_model(train_data, val_data)
        return seed, model, training_history

    def _process_test_batch(
        self,
        batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        model: tf.keras.Model,
        heatmap_generator: GradcamPlusPlus,
        heatmap_dir: str,
    ) -> List[ImageDetail]:
        images, true_labels_batch, ids_batch = batch
        batch_size = images.shape[0]
        batch_details = []

        batch_predictions_raw = model.predict(images, verbose=0)
        batch_predicted_labels = batch_predictions_raw.argmax(axis=1)
        batch_confidence_scores = batch_predictions_raw

        for i in range(batch_size):
            image_np = images[i].numpy()
            true_label = int(true_labels_batch[i].numpy())
            image_id = ids_batch[i].numpy().decode("utf-8")
            predicted_label = int(batch_predicted_labels[i])
            confidence_scores = batch_confidence_scores[i].tolist()

            heatmap_path, detected_features = self.visual_explainer.generate_explanation_for_image(heatmap_generator, image_np, true_label, image_id, heatmap_dir)

            img_details = ImageDetail(
                image_id=image_id,
                label=Gender(true_label),
                prediction=Gender(predicted_label),
                confidence_scores=confidence_scores,
                heatmap_path=heatmap_path,
                detected_features=detected_features,
            )
            batch_details.append(img_details)
        return batch_details

    def _create_replicate_result(
        self,
        seed: int,
        history: ModelTrainingHistory,
        analysis: AnalysisResult,
    ) -> ReplicateResult:
        save_results = self.config.output.artifact_level in [ArtifactSavingLevel.RESULTS_ONLY, ArtifactSavingLevel.FULL]
        return ReplicateResult(
            seed=seed,
            history=history if save_results else None,
            analysis=analysis if save_results else None,
        )

    def _run_single_replicate(
        self,
        replicate_index: int,
        data_splits: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
        exp_id: str,
        replicate_output_path: str,
    ) -> ReplicateResult:
        train_data, val_data, test_data = data_splits
        seed, model, history = self._initialize_replicate(replicate_index, exp_id, train_data, val_data)

        heatmap_generator = self.visual_explainer.setup_heatmap_generator(model)
        heatmap_dir = os.path.join(replicate_output_path, "heatmaps")
        all_image_details: List[ImageDetail] = []

        self.logger.info(f"[{exp_id}/rep{replicate_index+1}] Processing test data for predictions and explanations.")

        processed_count = 0
        for batch in test_data:
            batch_details = self._process_test_batch(batch, model, heatmap_generator, heatmap_dir)
            all_image_details.extend(batch_details)
            processed_count += len(batch_details)

            if processed_count % (self.config.model.batch_size * 5) == 0:
                self.logger.info(f"[{exp_id}/rep{replicate_index+1}] Processed {processed_count} test images...")

        self.logger.info(f"[{exp_id}/rep{replicate_index+1}] Completed processing {processed_count} test images.")

        analysis = self.bias_analyzer.analyze(all_image_details)
        result = self._create_replicate_result(seed, history, analysis)

        self.logger.info(f"[{exp_id}] Replicate {replicate_index+1} completed.")
        return result

    def run_single_experiment(
        self,
        target_male_proportion: float,
        feature_mask: Optional[MaskDetails],
    ) -> ExperimentResult:

        if feature_mask:
            features_str = "_".join(sorted([f.value.replace("_", "") for f in feature_mask.target_features]))
            gender_str = feature_mask.target_gender.name.lower()
            mask_str = f"mask_{gender_str}_{features_str}"
        else:
            mask_str = "mask_none"

        prop_str = str(int(target_male_proportion * 100)).zfill(3)
        exp_id = f"prop_{prop_str}_{mask_str}"

        self.logger.info(f"--- Starting Experiment: {exp_id} ---")
        self.logger.info(f"Parameters: male_prop={target_male_proportion}, feature_mask={feature_mask.model_dump_json() if feature_mask else 'None'}")

        data_splits = self.dataset_generator.prepare_datasets(target_male_proportion, feature_mask, self.config.core.base_random_seed, exp_id)

        replicate_results: List[ReplicateResult] = []
        for rep_idx in range(self.config.core.replicate_count):
            replicate_output_path = os.path.join(self.config.output.base_dir, exp_id, f"replicate_{rep_idx}")
            result = self._run_single_replicate(rep_idx, data_splits, exp_id, replicate_output_path)
            replicate_results.append(result)

        exp_params = ExperimentParameters(
            target_male_proportion=target_male_proportion,
            feature_mask=feature_mask,
        )

        experiment_summary = ExperimentResult(
            id=exp_id,
            parameters=exp_params,
            replicates=replicate_results,
        )

        self.logger.info(f"--- Experiment Completed: {exp_id} ---")
        return experiment_summary

    def _determine_experiment_setups(self) -> List[Tuple[float, Optional[MaskDetails]]]:
        valid_setups_tuples = []
        male_props = self.config.core.male_proportion_targets
        masking_configs = self.config.core.masking_configs or []

        for ratio in male_props:
            valid_setups_tuples.append((ratio, None))
            for mask_conf in masking_configs:
                valid_setups_tuples.append((ratio, mask_conf))

        seen_setups = set()
        unique_setups = []
        for setup in valid_setups_tuples:
            setup_key = (setup[0], setup[1].model_dump_json() if setup[1] else None)
            if setup_key not in seen_setups:
                unique_setups.append(setup)
                seen_setups.add(setup_key)

        self.logger.info(f"Unique experimental setups determined: {len(unique_setups)}")
        self.logger.debug(f"Unique setups: {unique_setups}")

        return unique_setups

    def run_all_experiments(self) -> None:
        self.logger.info("Starting execution of all defined experiments.")
        os.makedirs(self.config.output.base_dir, exist_ok=True)

        setups = self._determine_experiment_setups()
        total_experiments = len(setups)

        experiment_results: List[ExperimentResult] = []
        for i, (male_prop, feature_mask) in enumerate(setups):
            self.logger.info(f"*** Running Experiment {i+1}/{total_experiments} ***")
            result = self.run_single_experiment(male_prop, feature_mask)
            experiment_results.append(result)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result = f"experiment_results_{timestamp}.json"
            result_path = os.path.join(self.config.output.base_dir, result)

            results_list_dict = [exp_res.model_dump(mode="json") for exp_res in experiment_results]
            with open(result_path, "w") as f:
                json.dump(results_list_dict, f)
            self.logger.info(f"Saved cumulative results for {i+1} experiments to {result_path}")

        self.logger.info(f"*** All {total_experiments} experiments completed successfully. ***")
        final_results_filename = "experiment_results_final.json"
        final_results_path = os.path.join(self.config.output.base_dir, final_results_filename)

        final_results_list_dict = [exp_res.model_dump(mode="json") for exp_res in experiment_results]
        with open(final_results_path, "w") as f:
            json.dump(final_results_list_dict, f)
        self.logger.info(f"Saved final results to {final_results_path}")
