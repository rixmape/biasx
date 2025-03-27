import hashlib
import itertools
import json
import os

from classifier import ModelTrainer
from config import ClassifierConfig, DatasetConfig
from dataset import DatasetGenerator

from biasx import BiasAnalyzer

OUTPUT_DIR = "."
RESULTS_PATH = os.path.join(OUTPUT_DIR, "experiment_results.json")
INPUT_SHAPE = (48, 48, 1)
NUM_REPLICATES = 3

# Setup for demographic parity experiments
MALE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MASKED_GENDERS = []
MASKED_FEATURES = []

# Setup for feature attention experiments
# MALE_RATIOS = [0.5]  # Balanced gender distribution
# MASKED_GENDERS = [0, 1]  # 0: male, 1: female
# MASKED_FEATURES = ["left_eye", "right_eye", "nose", "lips", "left_cheek", "right_cheek", "chin", "forehead", "left_eyebrow", "right_eyebrow"]


def create_experiment_metadata(ratio, masked_feature, masked_gender):
    "Creates experiment metadata with id."
    male_ratio = int(ratio * 100)
    female_ratio = 100 - male_ratio
    id_str = f"{male_ratio}-{masked_feature}-{masked_gender}"
    exp_id = hashlib.md5(id_str.encode()).hexdigest()
    return {
        "id": exp_id,
        "male_ratio": male_ratio,
        "female_ratio": female_ratio,
        "masked_feature": masked_feature,
        "masked_gender": masked_gender,
    }


def train_single_model(ratio, masked_feature, masked_gender):
    "Trains a single model and saves it."
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
        batch_size=64,
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
    "Analyzes model bias using BiasAnalyzer."
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
    "Extracts bias analysis results."
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
    "Runs an experiment for given parameters."
    metadata = create_experiment_metadata(ratio, masked_feature, masked_gender)
    bias_analyses = []
    for _ in range(NUM_REPLICATES):
        model_path = train_single_model(ratio, masked_feature, masked_gender)
        while True:
            bias_results = analyze_model_bias(model_path)
            if bias_results["feature_analyses"]:
                break
        os.remove(model_path)
        bias_analyses.append(bias_results)
    return {"metadata": metadata, "bias_analyses": bias_analyses}


def load_results():
    "Loads existing results from file."
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return []


def save_results(results):
    "Saves results to file."
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def update_experiment_results(experiment_result):
    "Updates experiment results with new data."
    results = load_results()
    exp_id = experiment_result["metadata"]["id"]
    existing = next((entry for entry in results if entry["metadata"]["id"] == exp_id), None)
    if existing:
        existing["bias_analyses"].extend(experiment_result["bias_analyses"])
    else:
        results.append(experiment_result)
    save_results(results)


def run_all_experiments():
    "Iterates through all experiments and updates results."
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ratio, masked_feature, masked_gender in itertools.product(MALE_RATIOS, MASKED_FEATURES, MASKED_GENDERS):
        experiment_result = run_experiment(ratio, masked_feature, masked_gender)
        update_experiment_results(experiment_result)
        print(f"\033[92mCompleted experiment with ratio={ratio}, feature={masked_feature}, gender={masked_gender}\033[92m")


if __name__ == "__main__":
    run_all_experiments()
