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
from datatypes import DatasetSplits, Gender
from utils import setup_logger

logger = setup_logger(name="experiment.runner")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ExperimentRunner:
    """Class that orchestrates the overall experiment workflow by integrating dataset preparation, model training, visual explanation, and bias analysis."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.feature_masker = FeatureMasker(self.config)
        self.dataset_generator = DatasetGenerator(self.config, self.feature_masker)
        self.model_trainer = ModelTrainer(self.config)
        self.visual_explainer = VisualExplainer(self.config, self.feature_masker)
        self.bias_analyzer = BiasAnalyzer()

    def _set_random_seeds(self, seed: int) -> None:
        """Sets the random seeds for reproducibility across various libraries and modules."""
        logger.info(f"Setting random seed: {seed}")

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _run_replicate(self, replicate: int, data_splits: DatasetSplits) -> dict:
        """Runs a single experiment replicate by setting seeds, preparing data, training the model, and analyzing results."""
        logger.info(f"Running replicate {replicate + 1}/{self.config.replicate}")

        seed = self.config.base_seed + replicate
        self._set_random_seeds(seed)

        model, predictions, test_labels = self.model_trainer.train_and_predict(data_splits)
        key_features = self.visual_explainer.explain(model, data_splits.test_dataset)
        analysis = self.bias_analyzer.analyze(test_labels, predictions, key_features)

        tf.keras.backend.clear_session()

        return {"seed": seed, "analysis": analysis}

    def run_experiment(self, male_ratio: float, mask_gender: int, mask_feature: str) -> dict:
        """Executes a single experiment with the specified parameters"""
        logger.info(f"Running experiment: male_ratio={male_ratio}, mask_gender={mask_gender}, mask_feature={mask_feature}")

        feature_str = mask_feature.replace("_", "") if mask_feature is not None else "none"
        gender_str = Gender(mask_gender).name.lower() if mask_gender is not None else "none"
        exp_id = f"male_{int(male_ratio * 100)}_mask_{feature_str}_of_{gender_str}"

        data_splits = self.dataset_generator.prepare_data(male_ratio, mask_gender, mask_feature, self.config.base_seed)

        replicates = []
        for rep in range(self.config.replicate):
            replicate_result = self._run_replicate(rep, data_splits)
            replicates.append(replicate_result)

        return {
            "id": exp_id,
            "parameters": {
                "male_ratio": male_ratio,
                "mask_gender": mask_gender,
                "mask_feature": mask_feature,
            },
            "replicates": replicates,
        }

    def run_all_experiments(self) -> None:
        """Executes multiple experiment replicates sequentially and aggregates their results."""
        os.makedirs(self.config.results_path, exist_ok=True)

        mask_genders = self.config.mask_genders if self.config.mask_genders else [None]
        mask_features = self.config.mask_features if self.config.mask_features else [None]
        setups = list(itertools.product(self.config.male_ratios, mask_genders, mask_features))

        logger.info(f"Running {len(setups)} experiments with {self.config.replicate} replicates each")
        logger.debug(f"Experiment setups: {setups}")

        experiments = []
        for male_ratio, mask_gender, mask_feature in setups:
            result = self.run_experiment(male_ratio, mask_gender, mask_feature)
            experiments.append(result)

            timestamp = int(datetime.now().timestamp())
            path = os.path.join(self.config.results_path, f"experiments_{timestamp}.json")
            with open(path, "w") as f:
                json.dump(experiments, f, indent=2)
            logger.info(f"Saved experiment results to {path}")

            gc.collect()
            tf.keras.backend.clear_session()

        logger.info("All experiments completed successfully")


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
