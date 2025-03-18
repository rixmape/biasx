import json
import os

from classifier import ModelTrainer
from config import ClassifierConfig, DatasetConfig
from dataset import DatasetGenerator

from biasx import BiasAnalyzer

INPUT_SHAPE = (48, 48, 1)


def setup_model(output_dir, ratio, feature, gender, exp_name):
    dataset_config = DatasetConfig(
        dataset_size=10000,
        gender_ratios={0: ratio, 1: 1.0 - ratio},
        target_gender=gender,
        target_feature=feature,
    )
    dataset = DatasetGenerator(dataset_config).create_dataset()

    model_config = ClassifierConfig(input_shape=INPUT_SHAPE)
    model = ModelTrainer(model_config).run_training(dataset)

    model_path = os.path.join(output_dir, f"{exp_name}.keras")
    model.save(model_path)
    return model_path


def extract_biasx_results(analysis_result) -> dict:
    """Extract key metrics from BiasX analysis result."""
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


def analyze_bias(model_path):
    """Perform bias analysis on trained model."""
    biasx_config = {
        "model": {
            "path": model_path,
        },
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


def main():
    """Run gender bias experiments with varying gender ratios and target settings."""
    output_dir = "outputs/gender_ratio/"
    os.makedirs(output_dir, exist_ok=True)

    male_ratios = [0.3, 0.5, 0.7]
    target_features = [None]
    target_genders = [None]

    setups = [(r, f, g) for r in male_ratios for f in target_features for g in target_genders]

    results = {}
    for ratio, feature, gender in setups:

        feature = feature if feature else "none"
        gender = gender if gender else "none"

        exp_name = f"male_{int(ratio * 100)}_female_{int((1 - ratio) * 100)}_feature_{feature}_target_{gender}"
        model_path = setup_model(output_dir, ratio, feature, gender, exp_name)
        analysis = analyze_bias(model_path)
        results[exp_name] = {"id": exp_name, "bias_analysis": analysis}

    summary_path = os.path.join(output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
