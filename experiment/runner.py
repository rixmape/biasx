import gc
import itertools
import json
import os
import random
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf

# isort: off
from analyzer import BiasAnalyzer
from config import Config
from dataset import DatasetGenerator
from explainer import VisualExplainer
from masker import FeatureMasker
from model import ModelTrainer
from datatypes import Gender
from utils import setup_logger


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ExperimentRunner:
    """Class that orchestrates the overall experiment workflow by integrating dataset preparation, model training, visual explanation, and bias analysis."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        self.exp_log_path = self._create_log_path()
        self.logger = setup_logger(name="experiment_runner", log_path=self.exp_log_path)

        self.feature_masker = FeatureMasker(self.config, self.exp_log_path)
        self.dataset_generator = DatasetGenerator(self.config, self.feature_masker, self.exp_log_path)
        self.model_trainer = ModelTrainer(self.config, self.exp_log_path)
        self.visual_explainer = VisualExplainer(self.config, self.feature_masker, self.exp_log_path)
        self.bias_analyzer = BiasAnalyzer(self.exp_log_path)

    def _create_log_path(self) -> str:
        """Creates a unique log directory for the current experiment run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.log_path, f"experiment_{timestamp}")
        os.makedirs(path, exist_ok=True)
        return path

    def _set_random_seeds(self, seed: int) -> None:
        """Sets the random seeds for reproducibility across various libraries and modules."""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _run_replicate(self, replicate: int, train_data: tf.data.Dataset, val_data: tf.data.Dataset, test_data: tf.data.Dataset) -> dict:
        """Runs a single experiment replicate by setting seeds, preparing data, training the model, and analyzing results."""
        self.logger.info(f"Running replicate {replicate + 1}/{self.config.replicate}")

        seed = self.config.base_seed + replicate
        self._set_random_seeds(seed)

        model = self.model_trainer.train_model(train_data, val_data)
        del train_data, val_data

        predictions, labels = self.model_trainer.predict(model, test_data)
        key_features = self.visual_explainer.explain(model, test_data)
        analysis = self.bias_analyzer.analyze(labels, predictions, key_features)

        return {"seed": seed, "analysis": analysis}

    def run_experiment(self, male_ratio: float, mask_gender: Optional[int], mask_features: Optional[list[str]]) -> dict:
        """Executes a single experiment with the specified parameters"""
        self.logger.info(f"Running experiment: male_ratio={male_ratio}, mask_gender={mask_gender}, mask_feature={mask_features}")

        features_str = "_".join([f.replace("_", "") for f in mask_features]) if mask_features is not None else "none"
        gender_str = Gender(mask_gender).name.lower() if mask_gender is not None else "none"
        exp_id = f"male_{int(male_ratio * 100)}_mask_{features_str}_of_{gender_str}"

        train_data, val_data, test_data = self.dataset_generator.prepare_data(male_ratio, mask_gender, mask_features, self.config.base_seed, exp_id)

        replicates = []
        for rep in range(self.config.replicate):
            replicate_result = self._run_replicate(rep, train_data, val_data, test_data)
            replicates.append(replicate_result)

            gc.collect()
            tf.keras.backend.clear_session()

        return {
            "id": exp_id,
            "parameters": {
                "male_ratio": male_ratio,
                "mask_gender": mask_gender,
                "mask_features": mask_features,
            },
            "replicates": replicates,
        }

    def run_all_experiments(self) -> None:
        """Executes multiple experiment replicates sequentially and aggregates their results."""
        self.logger.info("Starting all experiments")
        self.logger.debug(f"Experiments configuration: {self.config}")

        os.makedirs(self.config.output_path, exist_ok=True)

        mask_genders = self.config.mask_genders if self.config.mask_genders else [None]
        mask_features_set = self.config.mask_features if self.config.mask_features else [None]
        setups = list(itertools.product(self.config.male_ratios, mask_genders, mask_features_set))

        self.logger.info(f"Running {len(setups)} experiments with {self.config.replicate} replicates each")
        self.logger.debug(f"Experiments: {setups}")

        experiments = []
        for male_ratio, mask_gender, mask_features in setups:
            result = self.run_experiment(male_ratio, mask_gender, mask_features)
            experiments.append(result)

            timestamp = int(datetime.now().timestamp())
            path = os.path.join(self.config.output_path, f"experiments_{timestamp}.json")
            with open(path, "w") as f:
                json.dump(experiments, f)

            self.logger.info(f"Saved experiment results to {path}")

            gc.collect()
            tf.keras.backend.clear_session()

        self.logger.info("All experiments completed successfully")


def main() -> None:

    config = Config(
        replicate=1,
        male_ratios=[0.5],
        mask_features=None,
        mask_genders=None,
        feature_attention_threshold=0.3,
        epochs=3,
    )

    runner = ExperimentRunner(config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
