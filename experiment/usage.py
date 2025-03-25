import itertools
import json
import os

from classifier import ModelTrainer
from config import ClassifierConfig, DatasetConfig
from dataset import DatasetGenerator

from biasx import BiasAnalyzer

INPUT_SHAPE = (48, 48, 1)
NUM_REPLICATES = 3

MALE_RATIOS = [0.5]
MASKED_GENDERS = ["male"]
MASKED_FEATURES = [
    None,
    "left_eye",
    "right_eye",
    "nose",
    "lips",
    "left_cheek",
    "right_cheek",
    "chin",
    "forehead",
    "left_eyebrow",
    "right_eyebrow",
]

OUTPUT_DIR = "tmp/models/feature_manipulation_2"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")


def create_experiment_name(ratio, feature, gender):
    """Generate a unique experiment name based on the configuration."""
    feature_str = feature if feature is not None else "none"
    gender_str = gender if gender is not None else "none"
    male_ratio = int(ratio * 100)
    female_ratio = 100 - male_ratio
    return f"male_{male_ratio}_female_{female_ratio}_feature_{feature_str}_target_{gender_str}"


def train_single_model(ratio, feature, gender, exp_name, replicate_idx):
    """Train one model replicate and save it."""
    dataset_config = DatasetConfig(
        dataset_size=5000,
        gender_ratios={0: ratio, 1: 1.0 - ratio},
        masked_gender=gender,
        masked_feature=feature,
    )
    dataset = DatasetGenerator(dataset_config).create_dataset()

    model_config = ClassifierConfig(
        input_shape=INPUT_SHAPE,
        epochs=10,
        batch_size=32,
    )
    model = ModelTrainer(model_config).run_training(dataset)

    model_filename = f"{exp_name}_replicate_{replicate_idx}.keras"
    model_path = os.path.join(OUTPUT_DIR, model_filename)
    model.save(model_path)

    del model, dataset
    return model_path


def analyze_model_bias(model_path):
    """Perform bias analysis for a saved model and return results."""
    biasx_config = {
        "model": {"path": model_path},
        "dataset": {
            "image_width": INPUT_SHAPE[0],
            "image_height": INPUT_SHAPE[1],
            "color_mode": "L",
            "single_channel": True,
            "max_samples": 500,
        },
    }
    analysis_result = BiasAnalyzer(biasx_config).analyze()
    return extract_biasx_results(analysis_result)


def extract_biasx_results(analysis_result):
    """Extract relevant bias metrics from analysis result."""
    return {
        "disparity_scores": {k: getattr(analysis_result.disparity_scores, k) for k in ["biasx", "equalized_odds"]},
        "feature_analyses": {feature.value: {k: getattr(analysis, k) for k in ["bias_score", "male_probability", "female_probability"]} for feature, analysis in analysis_result.feature_analyses.items()},
    }


def run_experiment(ratio, feature, gender, results):
    """Run training and analysis for a single experiment configuration."""
    exp_name = create_experiment_name(ratio, feature, gender)
    feature_str = feature if feature is not None else "none"
    gender_str = gender if gender is not None else "none"
    male_ratio = int(ratio * 100)
    female_ratio = 100 - male_ratio

    experiment_data = results.get(
        exp_name,
        {
            "metadata": {
                "male_ratio": male_ratio,
                "female_ratio": female_ratio,
                "masked_feature": feature_str,
                "masked_gender": gender_str,
            },
            "replicates": [],
        },
    )

    for replicate in range(1, NUM_REPLICATES + 1):
        print(f"Training replicate {replicate} for experiment {exp_name}...")
        model_path = train_single_model(ratio, feature, gender, exp_name, replicate)
        bias_results = analyze_model_bias(model_path)

        experiment_data["replicates"].append(
            {
                "replicate": replicate,
                "model_path": model_path,
                "bias_analysis": bias_results,
            }
        )
        os.remove(model_path)

    results[exp_name] = experiment_data


def run_all_experiments():
    """Generate all experiment configurations and process each one."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)

    for ratio, feature, gender in itertools.product(MALE_RATIOS, MASKED_FEATURES, MASKED_GENDERS):
        run_experiment(ratio, feature, gender, results)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_all_experiments()
