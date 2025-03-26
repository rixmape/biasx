import itertools
import json
import os

from classifier import ModelTrainer
from config import ClassifierConfig, DatasetConfig
from dataset import DatasetGenerator

from biasx import BiasAnalyzer

INPUT_SHAPE = (48, 48, 1)
NUM_REPLICATES = 2

MALE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASKED_GENDERS = [None]
MASKED_FEATURES = [None]

OUTPUT_DIR = "tmp/models"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "gender_ratio.json")


def create_experiment_metadata(ratio, masked_feature, masked_gender):
    male_ratio = int(ratio * 100)
    female_ratio = 100 - male_ratio
    return {
        "male_ratio": male_ratio,
        "female_ratio": female_ratio,
        "masked_feature": masked_feature,
        "masked_gender": masked_gender,
    }


def train_single_model(ratio, masked_feature, masked_gender):
    dataset_config = DatasetConfig(
        dataset_size=5000,
        gender_ratios={0: ratio, 1: 1.0 - ratio},
        masked_gender=masked_gender,
        masked_feature=masked_feature,
        padding=2,
        random_seed=42,
    )
    dataset = DatasetGenerator(dataset_config).create_dataset()

    model_config = ClassifierConfig(
        input_shape=INPUT_SHAPE,
        epochs=10,
        batch_size=32,
        val_split=0.2,
        test_split=0.1,
        random_seed=42,
    )
    model = ModelTrainer(model_config).run_training(dataset)

    model_path = os.path.join(OUTPUT_DIR, "model.keras")
    model.save(model_path)
    del model, dataset
    return model_path


def analyze_model_bias(model_path):
    biasx_config = {
        "model": {"path": model_path},
        "dataset": {
            "image_width": INPUT_SHAPE[0],
            "image_height": INPUT_SHAPE[1],
            "color_mode": "L",
            "single_channel": True,
            "max_samples": 500,
            "seed": 42,
        },
    }
    analysis_result = BiasAnalyzer(biasx_config).analyze()
    return extract_biasx_results(analysis_result)


def extract_biasx_results(analysis_result):
    return {
        "disparity_scores": {
            "biasx": analysis_result.disparity_scores.biasx,
            "equalized_odds": analysis_result.disparity_scores.equalized_odds,
        },
        "feature_analyses": {
            feature.value: {
                "bias_score": analysis.bias_score,
                "male_probability": analysis.male_probability,
                "female_probability": analysis.female_probability,
            }
            for feature, analysis in analysis_result.feature_analyses.items()
        },
    }


def run_experiment(ratio, masked_feature, masked_gender):
    metadata = create_experiment_metadata(ratio, masked_feature, masked_gender)
    bias_analyses = []
    for _ in range(1, NUM_REPLICATES + 1):
        model_path = train_single_model(ratio, masked_feature, masked_gender)
        while True:
            bias_results = analyze_model_bias(model_path)
            if bias_results["feature_analyses"]:
                break
        os.remove(model_path)
        bias_analyses.append(bias_results)
    return {"metadata": metadata, "bias_analyses": bias_analyses}


def run_all_experiments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)
    else:
        results = []

    for ratio, masked_feature, masked_gender in itertools.product(MALE_RATIOS, MASKED_FEATURES, MASKED_GENDERS):
        experiment_result = run_experiment(ratio, masked_feature, masked_gender)
        results.append(experiment_result)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_all_experiments()
