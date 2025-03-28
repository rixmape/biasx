"""
An experimental framework for investigating gender bias in facial classification models through controlled experiments,
feature masking, and explainability analysis. This module integrates dataset preparation with demographic controls,
neural network training, visual explanation of model decisions via GradCAM++, and multi-dimensional fairness analysis
to quantify disparities in model performance across gender groups.
"""

import itertools
import json
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Literal, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from charts import plot_bias_scores, plot_confusion_matrix, plot_dataset_distribution, plot_feature_probabilities, plot_image_grid, plot_training_history
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image
from sklearn.model_selection import train_test_split
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class Gender(Enum):
    """Enum for gender classification."""

    MALE = 0
    FEMALE = 1


@dataclass
class FeatureBox:
    """Data class for facial feature bounding box with importance score."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    name: str
    importance: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Dataclass holding configuration settings for experiments."""

    base_seed: int = 42

    dataset_name: Literal["utkface", "fairface"] = "utkface"
    dataset_size: int = 2000
    validation_ratio: float = 0.1
    testing_ratio: float = 0.2
    image_size: int = 48
    grayscale: bool = True

    male_ratios: list[float] = field(default_factory=lambda: [0.5])
    mask_genders: list[int] = None
    mask_features: list[str] = None
    activation_threshold: float = 0.5

    batch_size: int = 64
    epochs: int = 10

    exp_replicate: int = 1
    exp_output_dir: str = "tmp/experiments"


@dataclass
class DatasetSplits:
    """Dataclass holding train, validation, and test datasets with labels."""

    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class FeatureMasker:
    """Handles landmark detection and feature masking."""

    def __init__(self):
        self.landmarker = self.load_landmarker()
        self.feature_map = self.load_feature_map()

    @staticmethod
    def load_landmarker() -> FaceLandmarker:
        """Load landmark detector model."""
        model_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="mediapipe_landmarker.task", repo_type="model")
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path))
        return FaceLandmarker.create_from_options(options)

    @staticmethod
    def load_feature_map() -> dict[str, list[int]]:
        """Load feature to landmark indices mapping."""
        path = hf_hub_download(repo_id="rixmape/biasx-models", filename="landmark_map.json", repo_type="model")
        with open(path, "r") as f:
            return json.load(f)

    def _detect_landmarks(self, image: np.ndarray) -> Any:
        """Detect landmarks in an image."""
        rgb = image.copy()

        if image.shape[-1] == 1 or len(image.shape) == 2:
            rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        return result.face_landmarks[0] if result.face_landmarks else None

    def _to_pixel_coords(self, landmarks: list, img_shape: tuple[int, int]) -> list[tuple[int, int]]:
        """Convert normalized landmarks to pixel coordinates."""
        height, width = img_shape[:2]
        return [(int(pt.x * width), int(pt.y * height)) for pt in landmarks]

    def _get_bbox(self, landmarks: list[tuple[int, int]], feature: str, pad: int = 0) -> tuple[int, int, int, int]:
        """Get bounding box for a feature from landmarks."""
        pts = [landmarks[i] for i in self.feature_map[feature]]
        min_x = max(0, min(x for x, _ in pts) - pad)
        min_y = max(0, min(y for _, y in pts) - pad)
        max_x = max(x for x, _ in pts) + pad
        max_y = max(y for _, y in pts) + pad

        return int(min_x), int(min_y), int(max_x), int(max_y)

    def apply_mask(self, image: np.ndarray, feature: str) -> np.ndarray:
        """Apply mask to a specific feature in an image."""
        landmarks = self._detect_landmarks(image)

        if landmarks is None:
            return image

        pix_coords = self._to_pixel_coords(landmarks, image.shape)
        min_x, min_y, max_x, max_y = self._get_bbox(pix_coords, feature)
        result = image.copy()
        result[min_y:max_y, min_x:max_x] = 0

        return result

    def get_feature_boxes(self, image: np.ndarray) -> list[FeatureBox]:
        """Identify feature boxes in an image."""
        landmarks = self._detect_landmarks(image)

        if landmarks is None:
            return []

        pix_coords = self._to_pixel_coords(landmarks, image.shape)

        boxes = []
        for feature, indices in self.feature_map.items():
            pts = [pix_coords[i] for i in indices]
            min_x = max(0, min(x for x, _ in pts))
            min_y = max(0, min(y for _, y in pts))
            max_x = max(x for x, _ in pts)
            max_y = max(y for _, y in pts)
            boxes.append(FeatureBox(min_x, min_y, max_x, max_y, feature))

        return boxes


class DatasetGenerator:
    """Responsible for loading, preprocessing, and managing datasets."""

    def __init__(self, config: ExperimentConfig, feature_masker: FeatureMasker):
        self.config = config
        self.feature_masker = feature_masker
        self.dataset = None
        self.exp_dir = None

    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from Hugging Face repository."""
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset_name}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path, columns=["image", "gender", "race", "age"])
        df = df[df["age"] > 0]
        df["image"] = df["image"].apply(lambda x: np.array(Image.open(BytesIO(x["bytes"]))))
        return df

    def _sample_by_strata(self, df: pd.DataFrame, sample_size: int, seed: int) -> list[pd.DataFrame]:
        """Sample data while preserving demographic distributions."""
        df["strata"] = df["race"].astype(str) + "_" + df["age"].astype(str)

        samples = []
        for _, group in df.groupby("strata"):
            grp_size = len(group)
            grp_sample_size = round(sample_size * (grp_size / len(df)))
            if grp_sample_size > 0:
                sample = group.sample(n=grp_sample_size, random_state=seed, replace=(grp_size < grp_sample_size))
                samples.append(sample.drop(columns=["strata"]))

        return samples

    def _sample_by_gender(self, male_ratio: float, seed: int) -> pd.DataFrame:
        """Sample dataset with a specific male-to-female ratio."""
        ratios = {Gender.MALE: male_ratio, Gender.FEMALE: 1.0 - male_ratio}

        samples = []
        for gender, ratio in ratios.items():
            gender_sample_size = round(self.config.dataset_size * ratio)
            gender_df = self.dataset[self.dataset["gender"] == gender.value]
            strata_samples = self._sample_by_strata(gender_df, gender_sample_size, seed)
            samples.append(pd.concat(strata_samples))

        return pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for the neural network."""
        image_size = (self.config.image_size, self.config.image_size)
        pil_image = Image.fromarray(image).convert("L").resize(image_size)
        image_arr = np.array(pil_image, dtype=np.float32) / 255.0
        return np.expand_dims(image_arr, axis=-1) if self.config.grayscale else image_arr

    def split_dataset(self, seed: int):
        train_val, test = train_test_split(self.dataset, test_size=self.config.testing_ratio, random_state=seed, stratify=self.dataset["gender"])
        train, val = train_test_split(train_val, test_size=self.config.validation_ratio / (1 - self.config.testing_ratio), random_state=seed, stratify=train_val["gender"])
        return test, train, val

    def prepare_data(self, male_ratio: float, mask_gender: int, mask_feature: str, seed: int, exp_dir: str) -> DatasetSplits:
        """Prepare dataset for training and evaluation."""
        self.exp_dir = exp_dir
        self.dataset = self._load_dataset()
        self.dataset = self._sample_by_gender(male_ratio, seed)

        if mask_gender is not None and mask_feature is not None:
            gender_mask = self.dataset["gender"] == mask_gender
            self.dataset.loc[gender_mask, "image"] = self.dataset.loc[gender_mask, "image"].apply(lambda img: self.feature_masker.apply_mask(img, mask_feature))

        self.dataset["image"] = self.dataset["image"].apply(self._preprocess_image)

        test, train, val = self.split_dataset(seed)

        splits = DatasetSplits(
            train_images=np.stack(train["image"].values),
            train_labels=train["gender"].values,
            val_images=np.stack(val["image"].values),
            val_labels=val["gender"].values,
            test_images=np.stack(test["image"].values),
            test_labels=test["gender"].values,
        )

        plot_dataset_distribution(self.dataset, path=os.path.join(self.exp_dir, "dataset_distribution.png"))
        plot_image_grid(self.dataset["image"], path=os.path.join(self.exp_dir, "preprocessed_images.png"))

        return splits


class ModelTrainer:
    """Handles model building, training, and prediction."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.history = None
        self.predictions = None
        self.exp_dir = None

    def _build_model(self, image_shape: tuple[int, int, int]) -> tf.keras.Model:
        """Build a CNN model for facial gender classification."""
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Input(shape=image_shape, name="input"))

        for block, (filters, layers) in enumerate([(64, 2), (128, 2), (256, 3)], start=1):
            for i in range(layers):
                model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{block}_conv{i+1}"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{block}_pool"))

        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_1"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="dense_output"))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model

    def train_and_predict(self, splits: DatasetSplits, exp_dir: str) -> tuple[tf.keras.Model, np.ndarray]:
        """Train the model and generate predictions."""
        self.exp_dir = exp_dir
        self.data_splits = splits

        tf.keras.backend.clear_session()

        self.model = self._build_model(image_shape=splits.train_images.shape[1:])

        self.history = self.model.fit(
            splits.train_images,
            splits.train_labels,
            validation_data=(splits.val_images, splits.val_labels),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            verbose=1,
            shuffle=True,
        ).history

        self.model.save(os.path.join(exp_dir, "model.keras"))
        self.predictions = self.model.predict(splits.test_images).argmax(axis=1)

        plot_training_history(self.history, path=os.path.join(self.exp_dir, "training_history.png"))
        plot_confusion_matrix(self.data_splits.test_labels, self.predictions, path=os.path.join(self.exp_dir, "confusion_matrix.png"))

        return self.model, self.predictions


class VisualExplainer:
    """Handles model visualization and explainability."""

    def __init__(self, config: ExperimentConfig, masker: FeatureMasker):
        self.config = config
        self.masker = masker
        self.images = None
        self.labels = None
        self.heatmaps = None
        self.all_boxes = None
        self.key_features = None
        self.exp_dir = None

    def _get_heatmap(self, visualizer: Any, image: np.ndarray, true_label: int) -> np.ndarray:
        """Generate activation heatmap for an image."""
        score_fn = lambda output: output[0][true_label]
        expanded_img = image[np.newaxis, ...]
        return visualizer(score_fn, expanded_img, penultimate_layer="block3_conv3")[0]

    def _compute_feature_importance(self, box: FeatureBox, act_map: np.ndarray) -> float:
        """Compute importance of a feature using activation map."""
        roi = act_map[max(0, box.min_y) : min(act_map.shape[0], box.max_y), max(0, box.min_x) : min(act_map.shape[1], box.max_x)]
        return float(np.mean(roi)) if roi.size else 0.0

    def _filter_feature_boxes(self, boxes: list[FeatureBox], act_map: np.ndarray) -> list[FeatureBox]:
        """Filter features by importance threshold and sort by importance."""
        filtered = []
        for b in boxes:
            copied_box = FeatureBox(
                min_x=b.min_x,
                min_y=b.min_y,
                max_x=b.max_x,
                max_y=b.max_y,
                name=b.name,
                importance=self._compute_feature_importance(b, act_map),
            )

            if copied_box.importance > self.config.activation_threshold:
                filtered.append(copied_box)

        return sorted(filtered, key=lambda x: x.importance, reverse=True)

    def explain(self, model: tf.keras.Model, images: np.ndarray, labels: np.ndarray, exp_dir: str) -> list[list[FeatureBox]]:
        """Identify key features for classification decisions."""
        self.images = images
        self.labels = labels
        self.exp_dir = exp_dir

        modifier = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
        visualizer = GradcamPlusPlus(model, model_modifier=modifier)

        self.heatmaps = [self._get_heatmap(visualizer, img, label) for img, label in zip(images, labels)]
        self.all_boxes = [self.masker.get_feature_boxes(img) for img in images]
        self.key_features = [self._filter_feature_boxes(boxes, act_map) for boxes, act_map in zip(self.all_boxes, self.heatmaps)]

        plot_image_grid(self.images, heatmaps=self.heatmaps, path=os.path.join(self.exp_dir, "heatmaps.png"))
        plot_image_grid(self.images, boxes=self.all_boxes, path=os.path.join(self.exp_dir, "features.png"))
        plot_image_grid(self.images, heatmaps=self.heatmaps, boxes=self.key_features, path=os.path.join(self.exp_dir, "activated_features.png"))

        return self.key_features


class BiasAnalyzer:
    """Analyzes bias in model predictions and features with comprehensive metrics."""

    def __init__(self):
        self.labels = None
        self.preds = None
        self.features = None
        self.exp_dir = None

    def _compute_feature_metrics(self) -> dict[str, dict[str, float]]:
        """Computes feature probabilities between genders and feature attention bias."""
        counts = defaultdict(lambda: defaultdict(int))
        for label, features in zip(self.labels, self.features):
            for feature in features:
                counts[feature.name][Gender(label).name] += 1

        male_count = (self.labels == Gender.MALE.value).sum()
        female_count = (self.labels == Gender.FEMALE.value).sum()

        analysis = {}
        total_bias = 0

        for feature, count in counts.items():
            male_prob = count[Gender.MALE.name] / max(male_count, 1)
            female_prob = count[Gender.FEMALE.name] / max(female_count, 1)
            bias = abs(male_prob - female_prob)

            analysis[feature] = {
                "male": round(male_prob, 2),
                "female": round(female_prob, 2),
                "bias": round(bias, 2),
            }

            total_bias += bias

        feature_bias = round(total_bias / len(analysis) if analysis else 0, 2)

        return analysis, feature_bias

    def _compute_confusion_matrix(self) -> dict[str, dict[str, int]]:
        """Computes confusion matrix for binary gender classification."""
        male_actual = self.labels == Gender.MALE.value
        female_actual = self.labels == Gender.FEMALE.value
        male_pred = self.preds == Gender.MALE.value
        female_pred = self.preds == Gender.FEMALE.value

        tp = int((male_actual & male_pred).sum())
        fn = int((male_actual & female_pred).sum())
        fp = int((female_actual & male_pred).sum())
        tn = int((female_actual & female_pred).sum())

        return {
            "true_male": {
                "predicted_male": tp,
                "predicted_female": fn,
            },
            "true_female": {
                "predicted_male": fp,
                "predicted_female": tn,
            },
        }

    def _compute_gender_metrics(self, cm: dict) -> tuple[dict[str, float], dict[str, float]]:
        """Computes key metrics for male and female genders."""
        tp_male = cm["true_male"]["predicted_male"]
        fn_male = cm["true_male"]["predicted_female"]
        fp_male = cm["true_female"]["predicted_male"]
        tn_male = cm["true_female"]["predicted_female"]

        male_tpr = tp_male / max(tp_male + fn_male, 1)
        male_fpr = fp_male / max(fp_male + tn_male, 1)
        male_ppv = tp_male / max(tp_male + fp_male, 1)
        male_fdr = fp_male / max(tp_male + fp_male, 1)

        female_tpr = tn_male / max(tn_male + fp_male, 1)
        female_fpr = fn_male / max(fn_male + tp_male, 1)
        female_ppv = tn_male / max(tn_male + fn_male, 1)
        female_fdr = fn_male / max(tn_male + fn_male, 1)

        return (
            {
                "tpr": round(male_tpr, 2),
                "fpr": round(male_fpr, 2),
                "ppv": round(male_ppv, 2),
                "fdr": round(male_fdr, 2),
            },
            {
                "tpr": round(female_tpr, 2),
                "fpr": round(female_fpr, 2),
                "ppv": round(female_ppv, 2),
                "fdr": round(female_fdr, 2),
            },
        )

    def _compute_bias_scores(self, cm: dict, male_metrics: dict, female_metrics: dict, feature_bias: float) -> dict[str, float]:
        """Computes bias metrics based on confusion matrix and gender metrics."""
        tp_male = cm["true_male"]["predicted_male"]
        fn_male = cm["true_male"]["predicted_female"]
        fp_male = cm["true_female"]["predicted_male"]
        tn_male = cm["true_female"]["predicted_female"]

        male_count = tp_male + fn_male
        female_count = fp_male + tn_male

        male_selection_rate = tp_male / max(male_count, 1)
        female_selection_rate = fp_male / max(female_count, 1)
        demographic_parity = abs(male_selection_rate - female_selection_rate)

        tpr_diff = abs(male_metrics["tpr"] - female_metrics["tpr"])
        fpr_diff = abs(male_metrics["fpr"] - female_metrics["fpr"])
        equalized_odds = max(tpr_diff, fpr_diff)

        ppv_diff = abs(male_metrics["ppv"] - female_metrics["ppv"])
        conditional_use_accuracy_equality = ppv_diff

        male_ratio = fn_male / max(fp_male, 1)
        female_ratio = fp_male / max(fn_male, 1)
        treatment_equality = abs(male_ratio - female_ratio)

        return {
            "demographic_parity": round(demographic_parity, 2),
            "equalized_odds": round(equalized_odds, 2),
            "conditional_use_accuracy_equality": round(conditional_use_accuracy_equality, 2),
            "treatment_equality": round(treatment_equality, 2),
            "feature_attention": feature_bias,
        }

    def analyze(self, labels: np.ndarray, preds: np.ndarray, features: list[list[FeatureBox]], exp_dir: str) -> dict:
        """Analyzes bias in model predictions and feature importance."""
        self.labels, self.preds, self.features, self.exp_dir = labels, preds, features, exp_dir

        feature_probs, feature_bias = self._compute_feature_metrics()
        confusion_matrix = self._compute_confusion_matrix()
        male_metrics, female_metrics = self._compute_gender_metrics(confusion_matrix)
        bias_scores = self._compute_bias_scores(confusion_matrix, male_metrics, female_metrics, feature_bias)

        plot_feature_probabilities(feature_probs, path=os.path.join(exp_dir, "feature_probabilities.png"))
        plot_bias_scores(bias_scores, path=os.path.join(exp_dir, "bias_metrics.png"))

        return {
            "feature_probabilities": feature_probs,
            "confusion_matrix": confusion_matrix,
            "male_metrics": male_metrics,
            "female_metrics": female_metrics,
            "bias_scores": bias_scores,
        }


class ExperimentRunner:
    """Manages the execution of bias analysis experiments."""

    def __init__(self, setup: Literal["gender_bias", "attention_bias"], replicate: int):
        self.config = self._get_experiment_config(setup, replicate)
        self.feature_masker = FeatureMasker()
        self.dataset_generator = DatasetGenerator(self.config, self.feature_masker)
        self.model_trainer = ModelTrainer(self.config)
        self.visual_explainer = VisualExplainer(self.config, self.feature_masker)
        self.bias_analyzer = BiasAnalyzer()

    def _get_experiment_config(self, setup: str, replicate: int) -> ExperimentConfig:
        """Get experiment configuration based on the setup type."""
        config = ExperimentConfig()
        config.exp_replicate = replicate

        if setup == "gender_bias":
            config.male_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif setup == "attention_bias":
            config.male_ratios = [0.5]
            config.mask_features = FeatureMasker.load_feature_map().keys()
            config.mask_genders = [g.value for g in Gender]

        return config

    def _run_replicate(self, replicate: int, data_splits: DatasetSplits, exp_dir: str) -> dict:
        """Run a single experimental replicate."""
        seed = self.config.base_seed + replicate
        set_random_seeds(seed)

        rep_dir = os.path.join(exp_dir, f"{replicate:03d}")
        os.makedirs(rep_dir, exist_ok=True)

        model, preds = self.model_trainer.train_and_predict(data_splits, rep_dir)
        features = self.visual_explainer.explain(model, data_splits.test_images, data_splits.test_labels, rep_dir)
        analysis = self.bias_analyzer.analyze(data_splits.test_labels, preds, features, rep_dir)

        return {
            "seed": seed,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }

    def run_experiment(self, male_ratio: float, mask_gender: int, mask_feature: str) -> dict:
        """Run a complete experiment with multiple replicates."""
        exp_id = f"{int(male_ratio * 100)}_{mask_gender if mask_gender is not None else 'none'}_{mask_feature if mask_feature is not None else 'none'}"
        exp_dir = os.path.join(self.config.exp_output_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)

        data = self.dataset_generator.prepare_data(male_ratio, mask_gender, mask_feature, self.config.base_seed, exp_dir)

        replicates = [self._run_replicate(rep, data, exp_dir) for rep in range(self.config.exp_replicate)]

        return {
            "id": exp_id,
            "parameters": {
                "dataset_size": self.config.dataset_size,
                "male_ratio": male_ratio,
                "mask_gender": mask_gender,
                "mask_feature": mask_feature,
            },
            "replicates": replicates,
            "timestamp": datetime.now().isoformat(),
        }

    def run_all_experiments(self) -> None:
        """Run all configured experiments."""
        os.makedirs(self.config.exp_output_dir, exist_ok=True)

        mask_genders = [self.config.mask_genders] if self.config.mask_genders is None else self.config.mask_genders
        mask_features = [self.config.mask_features] if self.config.mask_features is None else self.config.mask_features
        setups = set(itertools.product(self.config.male_ratios, mask_genders, mask_features))

        experiments = []
        for male_ratio, mask_gender, mask_feature in setups:
            entry = self.run_experiment(male_ratio, mask_gender, mask_feature)
            experiments.append(entry)

            path = os.path.join(self.config.exp_output_dir, f"experiments_{int(datetime.now().timestamp())}.json")
            with open(path, "w") as f:
                json.dump(experiments, f, indent=2)


if __name__ == "__main__":
    """
    There are two setups for the experiment:

    1, `gender_bias` uses datasets with controlled gender ratios (e.g., 10:90 male-female) without feature masking.
    2. `attention_bias` uses balanced datasets with feature masking (e.g., masked nose on males).

    The `replicate` parameter controls the number of models trained for each experiment.
    """

    runner = ExperimentRunner(setup="gender_bias", replicate=3)
    runner.run_all_experiments()
