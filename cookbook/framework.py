import gc
import itertools
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from sklearn.model_selection import train_test_split
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus


def setup_logger(name="biasx_framework", log_dir="logs", console_level=logging.INFO, file_level=logging.DEBUG):
    """Configure and return a logger that writes to both console and a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized: file output to {log_filename}")

    return logger


logger = setup_logger()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class Gender(Enum):
    """Enum representing genders with predefined values for MALE and FEMALE."""

    MALE = 0
    FEMALE = 1


@dataclass
class FeatureBox:
    """Dataclass encapsulating the coordinates and name of a facial feature's bounding box, with an optional importance score."""

    min_x: int
    min_y: int
    max_x: int
    max_y: int
    name: str
    importance: Optional[float] = None

    def get_area(self) -> int:
        """Calculates and returns the area of the bounding box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)


@dataclass
class Config:
    """Dataclass containing experiment parameters, dataset configurations, and model training settings."""

    # Experiment parameters
    replicate: int
    male_ratios: list[float]
    mask_genders: list[int]
    mask_features: list[str]
    mask_padding: int = 0
    feature_attention_threshold: Optional[float] = 0.5
    results_path: Optional[str] = "outputs"
    base_seed: Optional[int] = 42

    # Dataset parameters
    dataset_name: Literal["utkface", "fairface"] = "utkface"
    dataset_size: int = 5000
    val_split: float = 0.1
    test_split: float = 0.2
    image_size: int = 48
    grayscale: bool = True

    # Model parameters
    batch_size: int = 64
    epochs: int = 10


@dataclass
class DatasetSplits:
    """Dataclass that holds the training, validation, and testing TensorFlow datasets."""

    train_dataset: tf.data.Dataset
    val_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset


class FeatureMasker:
    """Class responsible for detecting facial landmarks and applying masks to specific facial features in images."""

    def __init__(self, config: Config):
        self.config = config
        self.landmarker = self.load_landmarker()
        self.feature_map = self.load_feature_map()

    @staticmethod
    def load_landmarker() -> FaceLandmarker:
        """Loads and returns a pre-trained face landmarker model from the Hugging Face hub using specified options."""
        model_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="mediapipe_landmarker.task", repo_type="model")
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path))
        return FaceLandmarker.create_from_options(options)

    @staticmethod
    def load_feature_map() -> dict[str, list[int]]:
        """Retrieves and parses a JSON file mapping facial features to landmark indices from HuggingFace."""
        path = hf_hub_download(repo_id="rixmape/biasx-models", filename="landmark_map.json", repo_type="model")
        return json.load(open(path, "r"))

    def _detect_landmarks(self, image: np.ndarray) -> Any:
        """Converts the input image to RGB (if necessary) and detects facial landmarks using the loaded landmarker."""
        if image.shape[-1] == 1 or len(image.shape) == 2:
            rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            rgb = image

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        return result.face_landmarks[0] if result.face_landmarks else None

    def _to_pixel_coords(self, landmarks: list, img_shape: tuple[int, int]) -> list[tuple[int, int]]:
        """Converts normalized landmark coordinates to absolute pixel coordinates based on the image dimensions."""
        height, width = img_shape[:2]
        return [(int(pt.x * width), int(pt.y * height)) for pt in landmarks]

    def _get_landmarks_in_pixels(self, image: np.ndarray) -> Optional[list[tuple[int, int]]]:
        """Obtains facial landmarks in pixel coordinates by detecting landmarks and converting them using `_to_pixel_coords`."""
        landmarks = self._detect_landmarks(image)
        if landmarks is None:
            return None
        return self._to_pixel_coords(landmarks, image.shape)

    def _get_bbox(self, pix_coords: list[tuple[int, int]], feature: str, pad: int = 0) -> tuple[int, int, int, int]:
        """Computes the bounding box for a specified facial feature from pixel coordinates, incorporating optional padding."""
        pts = [pix_coords[i] for i in self.feature_map[feature]]
        min_x = max(0, min(x for x, _ in pts) - pad)
        min_y = max(0, min(y for _, y in pts) - pad)
        max_x = max(x for x, _ in pts) + pad
        max_y = max(y for _, y in pts) + pad
        return int(min_x), int(min_y), int(max_x), int(max_y)

    def apply_mask(self, image: np.ndarray, feature: str) -> np.ndarray:
        """Applies a mask by setting the pixel values to zero within the bounding box of the specified facial feature in the image."""
        pix_coords = self._get_landmarks_in_pixels(image)
        if pix_coords is None:
            return image

        min_x, min_y, max_x, max_y = self._get_bbox(pix_coords, feature, self.config.mask_padding)
        result = image.copy()
        result[min_y:max_y, min_x:max_x] = 0

        return result

    def get_feature_boxes(self, image: np.ndarray) -> list[FeatureBox]:
        """Returns a list of `FeatureBox` objects representing bounding boxes for all defined facial features in the image."""
        pix_coords = self._get_landmarks_in_pixels(image)
        if pix_coords is None:
            return []

        boxes = []
        for feature in self.feature_map:
            min_x, min_y, max_x, max_y = self._get_bbox(pix_coords, feature)
            boxes.append(FeatureBox(min_x, min_y, max_x, max_y, feature))

        return boxes


class DatasetGenerator:
    """Class that loads, processes, samples, and splits the dataset, then creates TensorFlow datasets for model training and evaluation."""

    def __init__(self, config: Config, feature_masker: FeatureMasker):
        self.config = config
        self.feature_masker = feature_masker
        logger.info(f"Initializing DatasetGenerator with dataset: {config.dataset_name}")
        logger.debug(f"Configuration details: size={config.dataset_size}, image_size={config.image_size}x{config.image_size}, grayscale={config.grayscale}, splits=[train={1-config.val_split-config.test_split:.2f}, val={config.val_split:.2f}, test={config.test_split:.2f}]")

    def _load_dataset(self) -> pd.DataFrame:
        """Downloads and loads the dataset from Hugging Face Hub, filters by valid age, and extracts image bytes and labels."""
        logger.info(f"Downloading {self.config.dataset_name} dataset from Hugging Face Hub")
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset_name}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        logger.debug(f"Dataset file successfully downloaded to: {path}")

        logger.info("Loading and processing dataset file")
        df = pd.read_parquet(path, columns=["image", "gender", "race", "age"])
        logger.debug(f"Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        initial_count = len(df)
        df = df[df["age"] > 0]
        filtered_count = len(df)
        logger.debug(f"Age filtering: removed {initial_count - filtered_count} invalid entries ({(initial_count - filtered_count)/initial_count:.1%} of data)")

        logger.debug("Extracting image bytes from dataset")
        df["image_bytes"] = df["image"].apply(lambda x: x["bytes"])

        male_count = (df["gender"] == Gender.MALE.value).sum()
        female_count = (df["gender"] == Gender.FEMALE.value).sum()
        logger.debug(f"Gender distribution in source data: Male={male_count} ({male_count/len(df):.1%}), Female={female_count} ({female_count/len(df):.1%})")

        return df.drop(columns=["image"])

    def _sample_by_strata(self, df: pd.DataFrame, sample_size: int, seed: int) -> list[pd.DataFrame]:
        """Stratifies the dataset by race and age, then samples each group proportionally based on the desired sample size."""
        logger.debug(f"Stratifying dataset by race and age, target sample size: {sample_size}")
        df["strata"] = df["race"].astype(str) + "_" + df["age"].astype(str)

        strata_counts = df.groupby("strata").size()
        logger.debug(f"Found {len(strata_counts)} unique strata groups")

        samples = []
        strata_sampled = 0
        strata_skipped = 0

        for strata_name, group in df.groupby("strata"):
            grp_size = len(group)
            grp_sample_size = round(sample_size * (grp_size / len(df)))

            if grp_sample_size > 0:
                replacement_needed = grp_size < grp_sample_size
                if replacement_needed:
                    logger.debug(f"Strata '{strata_name}': Using replacement sampling ({grp_size} available, {grp_sample_size} needed)")

                sample = group.sample(n=grp_sample_size, random_state=seed, replace=replacement_needed)
                samples.append(sample.drop(columns=["strata"]))
                logger.debug(f"Strata '{strata_name}': Sampled {grp_sample_size}/{grp_size} images ({grp_sample_size/sample_size:.1%} of target)")
                strata_sampled += 1
            else:
                logger.debug(f"Strata '{strata_name}': Skipped (size {grp_size} too small for proportional sampling)")
                strata_skipped += 1

        logger.debug(f"Stratified sampling complete: {strata_sampled} strata used, {strata_skipped} strata skipped")

        if strata_skipped > 0:
            logger.warning(f"{strata_skipped} strata groups were skipped during sampling. This may affect representation.")

        return samples

    def _sample_by_gender(self, male_ratio: float, seed: int, df: pd.DataFrame) -> pd.DataFrame:
        """Samples the dataset separately by gender using the specified male ratio and stratified sampling."""
        logger.info(f"Sampling dataset with target male ratio: {male_ratio:.2f}")
        ratios = {Gender.MALE: male_ratio, Gender.FEMALE: 1.0 - male_ratio}

        samples = []
        for gender, ratio in ratios.items():
            gender_sample_size = round(self.config.dataset_size * ratio)
            gender_df = df[df["gender"] == gender.value]

            logger.debug(f"{gender.name} sampling: Targeting {gender_sample_size} images ({ratio:.1%} of total) from {len(gender_df)} available")

            if len(gender_df) < gender_sample_size:
                logger.warning(f"Insufficient {gender.name.lower()} samples available ({len(gender_df)} < {gender_sample_size}). Will use replacement sampling which may affect model performance.")

            strata_samples = self._sample_by_strata(gender_df, gender_sample_size, seed)
            gender_sample = pd.concat(strata_samples)
            samples.append(gender_sample)
            logger.debug(f"{gender.name} sampling complete: {len(gender_sample)} images sampled")

        combined = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)

        actual_male_ratio = (combined["gender"] == Gender.MALE.value).mean()
        logger.info(f"Sampling complete: {len(combined)} total images with actual male ratio: {actual_male_ratio:.2f} (target: {male_ratio:.2f})")

        if abs(actual_male_ratio - male_ratio) > 0.01:
            logger.warning(f"Achieved male ratio ({actual_male_ratio:.2f}) differs from target ({male_ratio:.2f})")

        return combined

    def _decode_and_process_image(self, image_bytes: tf.Tensor, label: tf.Tensor, mask_gender: int, mask_feature: str) -> tuple[tf.Tensor, tf.Tensor]:
        """Decodes image bytes, resizes the image, applies masking based on gender and feature if needed, and normalizes pixel values."""

        def process(img_bytes, lbl):
            image = tf.io.decode_image(img_bytes, channels=3 if not self.config.grayscale else 1, expand_animations=False)
            image = tf.image.resize(image, [self.config.image_size, self.config.image_size])

            if mask_gender is not None and mask_feature is not None and lbl == mask_gender:
                image_np = image.numpy()
                masked_image = self.feature_masker.apply_mask(image_np, mask_feature)
                image = tf.convert_to_tensor(masked_image)

            if self.config.grayscale and image.shape[-1] != 1:
                image = tf.image.rgb_to_grayscale(image)

            return tf.cast(image, tf.float32) / 255.0, lbl

        result = tf.py_function(process, [image_bytes, label], [tf.float32, tf.int64])
        result[0].set_shape([self.config.image_size, self.config.image_size, 1 if self.config.grayscale else 3])
        result[1].set_shape([])

        return result

    def _create_dataset(self, df: pd.DataFrame, mask_gender: int, mask_feature: str, purpose: str) -> tf.data.Dataset:
        """Creates a TensorFlow dataset from image bytes and labels, applying the decoding and processing function with parallel mapping."""
        logger.debug(f"Creating {purpose} TensorFlow dataset from {len(df)} samples")
        image_bytes = df["image_bytes"].values
        labels = df["gender"].values

        male_count = (df["gender"] == Gender.MALE.value).sum()
        female_count = (df["gender"] == Gender.FEMALE.value).sum()
        logger.debug(f"{purpose} dataset gender distribution: Male={male_count} ({male_count/len(df):.1%}), Female={female_count} ({female_count/len(df):.1%})")

        if mask_gender is not None and mask_feature is not None:
            gender_name = Gender(mask_gender).name
            masked_count = (df["gender"] == mask_gender).sum()
            logger.debug(f"{purpose} dataset masking: Feature '{mask_feature}' will be masked for {gender_name} gender ({masked_count} images, {masked_count/len(df):.1%} of this split)")
        else:
            logger.debug(f"{purpose} dataset: No feature masking applied")

        dataset = tf.data.Dataset.from_tensor_slices((image_bytes, labels))
        dataset = dataset.map(lambda x, y: self._decode_and_process_image(x, y, mask_gender, mask_feature), num_parallel_calls=tf.data.AUTOTUNE)

        logger.debug(f"{purpose} dataset creation complete")
        return dataset.prefetch(tf.data.AUTOTUNE)

    def split_dataset(self, seed: int, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into training, validation, and test sets using stratified sampling based on gender."""
        logger.info(f"Splitting dataset of {len(df)} samples into train/validation/test sets")

        train_val, test = train_test_split(df, test_size=self.config.test_split, random_state=seed, stratify=df["gender"])
        effective_val_split = self.config.val_split / (1 - self.config.test_split)
        train, val = train_test_split(train_val, test_size=effective_val_split, random_state=seed, stratify=train_val["gender"])

        train_male_ratio = (train["gender"] == Gender.MALE.value).mean()
        val_male_ratio = (val["gender"] == Gender.MALE.value).mean()
        test_male_ratio = (test["gender"] == Gender.MALE.value).mean()

        logger.debug(f"Split sizes: Train={len(train)} ({train_male_ratio:.2f} male ratio), Validation={len(val)} ({val_male_ratio:.2f} male ratio), Test={len(test)} ({test_male_ratio:.2f} male ratio)")

        overall_male_ratio = (df["gender"] == Gender.MALE.value).mean()
        max_deviation = max(abs(train_male_ratio - overall_male_ratio), abs(val_male_ratio - overall_male_ratio), abs(test_male_ratio - overall_male_ratio))

        if max_deviation > 0.05:
            logger.warning(f"Dataset split gender ratios deviate by {max_deviation:.3f} from overall ratio ({overall_male_ratio:.2f}). This may indicate stratification issues.")

        return train, val, test

    def prepare_data(self, male_ratio: float, mask_gender: int, mask_feature: str, seed: int) -> DatasetSplits:
        """Loads, samples by gender, splits the dataset, and returns batched and cached TensorFlow datasets for training, validation, and testing."""
        experiment_desc = f"male_ratio={male_ratio:.2f}"
        if mask_gender is not None and mask_feature is not None:
            experiment_desc += f", masking '{mask_feature}' for {Gender(mask_gender).name} gender"

        logger.info(f"Starting data preparation for experiment: {experiment_desc}")
        logger.debug(f"Using random seed: {seed}")

        df = self._load_dataset()
        logger.info(f"Dataset loaded successfully with {len(df)} samples")

        df = self._sample_by_gender(male_ratio, seed, df)

        train_df, val_df, test_df = self.split_dataset(seed, df)

        logger.info("Creating TensorFlow datasets")
        batch_size = self.config.batch_size
        logger.debug(f"Using batch size: {batch_size}")

        train_dataset = self._create_dataset(train_df, mask_gender, mask_feature, "TRAINING").batch(batch_size).cache()
        logger.debug(f"Training dataset cached with {len(train_df)} samples ({len(train_df) // batch_size + 1} batches)")

        val_dataset = self._create_dataset(val_df, mask_gender, mask_feature, "VALIDATION").batch(batch_size).cache()
        logger.debug(f"Validation dataset cached with {len(val_df)} samples ({len(val_df) // batch_size + 1} batches)")

        test_dataset = self._create_dataset(test_df, mask_gender, mask_feature, "TEST").batch(batch_size)
        logger.debug(f"Test dataset created with {len(test_df)} samples ({len(test_df) // batch_size + 1} batches)")

        logger.info("Dataset preparation complete")
        return DatasetSplits(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)


class ModelTrainer:
    """Class that constructs a convolutional neural network model, trains it on the dataset, and produces predictions on test data."""

    def __init__(self, config: Config):
        self.config = config
        logger.info(f"Initializing ModelTrainer with image size: {config.image_size}x{config.image_size}")
        logger.debug(f"Training parameters: batch_size={config.batch_size}, epochs={config.epochs}, grayscale={config.grayscale}")

    def _build_model(self) -> tf.keras.Model:
        """Constructs and compiles a sequential convolutional neural network model based on the configuration settings."""
        logger.info("Building CNN model architecture")
        model = tf.keras.Sequential()

        input_shape = (self.config.image_size, self.config.image_size, 1 if self.config.grayscale else 3)
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))
        logger.debug(f"Model input shape: {input_shape}")

        for block, (filters, layers) in enumerate([(64, 2), (128, 2), (256, 3)], start=1):
            logger.debug(f"Adding convolutional block {block}: {layers} layers with {filters} filters each")
            for i in range(layers):
                model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{block}_conv{i+1}"))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{block}_pool"))

        model.add(tf.keras.layers.Flatten(name="flatten"))
        logger.debug("Adding classification head layers")
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense_1"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="dense_output"))

        logger.debug("Compiling model with Adam optimizer (lr=0.0001) and sparse categorical crossentropy loss")
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        total_params = model.count_params()
        logger.info(f"Model built successfully: {total_params:,} total parameters")

        return model

    def train_and_predict(self, splits: DatasetSplits) -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
        """Trains the model on the training dataset, evaluates it on the validation set, and generates predictions on the test data."""
        logger.info("Starting model training process")
        model = self._build_model()

        logger.info(f"Training model for {self.config.epochs} epochs")
        history = model.fit(splits.train_dataset, validation_data=splits.val_dataset, epochs=self.config.epochs, verbose=0)

        train_acc = history.history["accuracy"][-1]
        val_acc = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        logger.info(f"Training completed - Final metrics: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        logger.debug(f"Final loss values: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_acc < 0.6:
            logger.warning(f"Low validation accuracy ({val_acc:.4f}). Model may be underfitting or the task might be challenging.")

        if val_acc < train_acc - 0.1:
            logger.warning(f"Large gap between training and validation accuracy ({train_acc:.4f} vs {val_acc:.4f}). Possible overfitting.")

        logger.info("Generating predictions on test dataset")
        all_predictions = []
        all_labels = []

        batch_count = 0
        total_samples = 0

        for batch in splits.test_dataset:
            batch_count += 1
            images, labels = batch
            batch_size = len(labels)
            total_samples += batch_size

            logger.debug(f"Processing test batch {batch_count}: {batch_size} samples")
            batch_predictions = model.predict(images, verbose=0)
            all_predictions.append(batch_predictions)
            all_labels.append(labels)

        logger.debug(f"Processed {batch_count} test batches with {total_samples} total samples")

        predictions = np.vstack(all_predictions).argmax(axis=1)
        test_labels = np.concatenate(all_labels)

        return model, predictions, test_labels


class VisualExplainer:
    """Class that generates visual explanations by computing Grad-CAM heatmaps and extracting feature importance from masked image regions."""

    def __init__(self, config: Config, masker: FeatureMasker):
        self.config = config
        self.masker = masker
        logger.info(f"Initializing VisualExplainer with feature attention threshold: {config.feature_attention_threshold}")
        logger.debug(f"Configuration: mask_padding={config.mask_padding}, image_size={config.image_size}x{config.image_size}")

    def _get_heatmap(self, visualizer: Any, image: np.ndarray, true_label: int) -> np.ndarray:
        """Generates a Grad-CAM heatmap for an image based on its true label using the provided visualizer function."""
        logger.debug(f"Generating Grad-CAM heatmap for image with true label: {Gender(true_label).name}")
        score_fn = lambda output: output[0][true_label]
        expanded_img = np.expand_dims(image, axis=0)
        heatmap = visualizer(score_fn, expanded_img, penultimate_layer="block3_conv3")[0]

        logger.debug(f"Heatmap generated with shape: {heatmap.shape}, intensity range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        return heatmap

    def _compute_feature_importance(self, box: FeatureBox, heatmap: np.ndarray) -> float:
        """Computes the average intensity within a feature box region of the heatmap to determine its importance."""
        roi = heatmap[max(0, box.min_y) : min(heatmap.shape[0], box.max_y), max(0, box.min_x) : min(heatmap.shape[1], box.max_x)]
        return float(np.mean(roi))

    def _filter_feature_boxes(self, boxes: list[FeatureBox], heatmap: np.ndarray) -> list[FeatureBox]:
        """Filters and orders feature boxes based on whether their computed importance exceeds the configured threshold."""

        filtered = []
        for b in boxes:
            if b.get_area() == 0:
                logger.warning(f"Feature '{b.name}' has zero area and will be ignored")
                continue

            imp = self._compute_feature_importance(b, heatmap)
            copied_box = FeatureBox(min_x=b.min_x, min_y=b.min_y, max_x=b.max_x, max_y=b.max_y, name=b.name, importance=imp)

            if copied_box.importance > self.config.feature_attention_threshold:
                filtered.append(copied_box)
                logger.debug(f"Feature '{b.name}' selected: importance {imp:.4f} > threshold {self.config.feature_attention_threshold}")
            else:
                logger.debug(f"Feature '{b.name}' filtered out: importance {imp:.4f} <= threshold {self.config.feature_attention_threshold}")

        filtered_boxes = sorted(filtered, key=lambda x: x.importance, reverse=True)
        logger.debug(f"Kept {len(filtered_boxes)}/{len(boxes)} features after importance filtering")

        if not filtered_boxes:
            logger.warning("No features exceeded the importance threshold. Consider lowering the threshold value.")

        return filtered_boxes

    def explain(self, model: tf.keras.Model, test_dataset: tf.data.Dataset) -> list[list[FeatureBox]]:
        """Iterates over test dataset images to generate heatmaps, extract feature boxes, and compile key features for explanation."""
        logger.info("Starting visual explanation generation")

        logger.debug("Modifying model output layer for Grad-CAM visualization")
        modifier = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
        visualizer = GradcamPlusPlus(model, model_modifier=modifier)

        key_features = []
        batch_count = 0
        image_count = 0
        empty_feature_count = 0

        logger.info("Processing test batches for feature importance")
        for batch in test_dataset:
            batch_count += 1
            images, labels = batch
            batch_size = len(images)

            logger.debug(f"Processing batch {batch_count} with {batch_size} images")

            for i in range(batch_size):
                img = images[i].numpy()
                label = int(labels[i].numpy())
                image_count += 1

                logger.debug(f"Analyzing image {image_count} with gender label: {Gender(label).name}")

                heatmap = self._get_heatmap(visualizer, img, label)
                boxes = self.masker.get_feature_boxes(img)

                if not boxes:
                    logger.warning(f"No facial features detected in image {image_count}. This may indicate an issue with face detection.")
                    empty_feature_count += 1

                logger.debug(f"Filtering {len(boxes)} feature boxes with importance threshold: {self.config.feature_attention_threshold}")
                filtered_boxes = self._filter_feature_boxes(boxes, heatmap)
                key_features.append(filtered_boxes)

                if i % 10 == 0:
                    logger.debug(f"Processed {i}/{batch_size} images in current batch")

            logger.debug(f"Completed batch {batch_count}")

            if batch_count % 5 == 0:
                logger.info(f"Visual explanation progress: {image_count} images processed so far")

            tf.keras.backend.clear_session()
            gc.collect()

        logger.info(f"Visual explanation complete: analyzed {image_count} images across {batch_count} batches")

        if empty_feature_count > 0:
            logger.warning(f"{empty_feature_count}/{image_count} images ({empty_feature_count/image_count:.1%}) had no detected facial features")

        feature_counts = [len(features) for features in key_features]
        avg_features = sum(feature_counts) / max(len(feature_counts), 1)
        logger.info(f"Average number of important features identified per image: {avg_features:.2f}")

        empty_importance_count = sum(1 for features in key_features if not features)
        if empty_importance_count > 0:
            logger.warning(f"{empty_importance_count}/{image_count} images ({empty_importance_count/image_count:.1%}) had no features exceeding the importance threshold. Consider lowering the threshold value.")

        return key_features


class BiasAnalyzer:
    """Class that computes bias metrics by analyzing feature probabilities, confusion matrices, and gender-based performance statistics."""

    def __init__(self):
        logger.info("Initializing BiasAnalyzer")
        logger.debug("BiasAnalyzer will compute feature metrics, construct confusion matrices, and calculate gender-specific bias scores")

    def _compute_feature_metrics(self, labels: np.ndarray, features: list[list[FeatureBox]]) -> dict[str, dict[str, float]]:
        """Computes per-feature probabilities and overall feature bias by aggregating counts from the provided feature boxes."""
        logger.info("Computing feature importance metrics across gender groups")

        counts = defaultdict(lambda: defaultdict(int))
        for label, feature_boxes in zip(labels, features):
            for feature_box in feature_boxes:
                counts[feature_box.name][Gender(label).name] += 1

        male_count = (labels == Gender.MALE.value).sum()
        female_count = (labels == Gender.FEMALE.value).sum()

        analysis = {}
        total_bias = 0
        feature_count = 0

        for feature_name, count in counts.items():
            male_prob = count[Gender.MALE.name] / max(male_count, 1)
            female_prob = count[Gender.FEMALE.name] / max(female_count, 1)
            bias = abs(male_prob - female_prob)
            feature_count += 1

            logger.debug(f"Feature '{feature_name}': male_prob={male_prob:.3f}, female_prob={female_prob:.3f}, bias={bias:.3f}")

            analysis[feature_name] = {
                "male": male_prob,
                "female": female_prob,
                "bias": bias,
            }

            total_bias += bias

        feature_bias = total_bias / max(feature_count, 1)
        logger.info(f"Feature analysis complete: {feature_count} unique features identified with average bias score of {feature_bias:.3f}")

        if feature_count == 0:
            logger.warning("No features were found in the analysis. This may indicate issues with feature extraction or importance thresholding.")
        elif feature_bias > 0.3:
            logger.warning(f"High average feature bias detected ({feature_bias:.3f}). This suggests significant gender-based differences in feature importance.")

        return analysis, feature_bias

    def _compute_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray) -> dict[str, dict[str, int]]:
        """Constructs a confusion matrix by comparing true labels with predictions for both male and female classes."""
        logger.info("Constructing gender-based confusion matrix")

        male_actual = labels == Gender.MALE.value
        female_actual = labels == Gender.FEMALE.value
        male_pred = preds == Gender.MALE.value
        female_pred = preds == Gender.FEMALE.value

        tp = int((male_actual & male_pred).sum())
        fn = int((male_actual & female_pred).sum())
        fp = int((female_actual & male_pred).sum())
        tn = int((female_actual & female_pred).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0

        logger.debug(f"Confusion matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
        logger.debug(f"Overall accuracy: {accuracy:.3f} ({tp + tn}/{total} correct)")

        male_correct = tp / (tp + fn) if (tp + fn) > 0 else 0
        female_correct = tn / (tn + fp) if (tn + fp) > 0 else 0
        logger.debug(f"Class-wise accuracy: Male={male_correct:.3f} ({tp}/{tp+fn}), Female={female_correct:.3f} ({tn}/{tn+fp})")

        if abs(male_correct - female_correct) > 0.1:
            logger.warning(f"Significant accuracy gap between genders: Male={male_correct:.3f}, Female={female_correct:.3f}. This suggests potential bias in model performance.")

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
        """Calculates gender-specific performance metrics such as TPR, FPR, PPV, and FDR using the confusion matrix data."""
        logger.info("Computing gender-specific fairness metrics")

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

        logger.debug(f"Male metrics: TPR={male_tpr:.3f}, FPR={male_fpr:.3f}, PPV={male_ppv:.3f}, FDR={male_fdr:.3f}")
        logger.debug(f"Female metrics: TPR={female_tpr:.3f}, FPR={female_fpr:.3f}, PPV={female_ppv:.3f}, FDR={female_fdr:.3f}")

        tpr_gap = abs(male_tpr - female_tpr)
        fpr_gap = abs(male_fpr - female_fpr)
        ppv_gap = abs(male_ppv - female_ppv)

        if tpr_gap > 0.1:
            logger.warning(f"Large true positive rate gap between genders: {tpr_gap:.3f}. The model has different sensitivity for different genders.")

        if fpr_gap > 0.1:
            logger.warning(f"Large false positive rate gap between genders: {fpr_gap:.3f}. The model has different specificity for different genders.")

        if ppv_gap > 0.1:
            logger.warning(f"Large precision gap between genders: {ppv_gap:.3f}. The model has different reliability for different genders.")

        return (
            {
                "tpr": male_tpr,
                "fpr": male_fpr,
                "ppv": male_ppv,
                "fdr": male_fdr,
            },
            {
                "tpr": female_tpr,
                "fpr": female_fpr,
                "ppv": female_ppv,
                "fdr": female_fdr,
            },
        )

    def _compute_bias_scores(self, cm: dict, male_metrics: dict, female_metrics: dict, feature_bias: float) -> dict[str, float]:
        """Derives overall bias scores based on demographic parity, equalized odds, conditional use accuracy, treatment equality, and feature attention."""
        logger.info("Computing composite bias scores across multiple fairness criteria")

        tp_male = cm["true_male"]["predicted_male"]
        fn_male = cm["true_male"]["predicted_female"]
        fp_male = cm["true_female"]["predicted_male"]
        tn_male = cm["true_female"]["predicted_female"]

        male_count = tp_male + fn_male
        female_count = fp_male + tn_male

        male_selection_rate = tp_male / max(male_count, 1)
        female_selection_rate = fp_male / max(female_count, 1)
        demographic_parity = abs(male_selection_rate - female_selection_rate)

        logger.debug(f"Selection rates: Male={male_selection_rate:.3f}, Female={female_selection_rate:.3f}")
        logger.debug(f"Demographic parity difference: {demographic_parity:.3f}")

        tpr_diff = abs(male_metrics["tpr"] - female_metrics["tpr"])
        fpr_diff = abs(male_metrics["fpr"] - female_metrics["fpr"])
        equalized_odds = max(tpr_diff, fpr_diff)

        logger.debug(f"TPR difference: {tpr_diff:.3f}, FPR difference: {fpr_diff:.3f}")
        logger.debug(f"Equalized odds: {equalized_odds:.3f} (maximum of TPR and FPR differences)")

        # TODO: Clarify that CUAE requires equal PPV and NPV, but since male NPV is similar to female PPV and vice versa, we can just use PPV
        ppv_diff = abs(male_metrics["ppv"] - female_metrics["ppv"])
        conditional_use_accuracy_equality = ppv_diff

        logger.debug(f"Conditional use accuracy equality: {conditional_use_accuracy_equality:.3f}")
        logger.debug(f"Feature attention bias: {feature_bias:.3f}")

        bias_scores = {
            "demographic_parity": demographic_parity,
            "equalized_odds": equalized_odds,
            "conditional_use_accuracy_equality": conditional_use_accuracy_equality,
            "feature_attention": feature_bias,
        }

        average_bias = sum(bias_scores.values()) / len(bias_scores)
        logger.info(f"Overall average bias score: {average_bias:.3f}")

        if average_bias > 0.2:
            logger.warning(f"High overall bias detected ({average_bias:.3f}). Multiple fairness metrics suggest problematic gender disparities.")

        sorted_biases = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"Fairness criteria ordered by bias magnitude: {sorted_biases}")

        return bias_scores

    def analyze(self, labels: np.ndarray, preds: np.ndarray, features: list[list[FeatureBox]]) -> dict:
        """Integrates feature metrics, confusion matrix, gender metrics, and bias scores to produce a comprehensive bias analysis."""
        logger.info(f"Starting comprehensive bias analysis on dataset with {len(labels)} samples")

        if len(labels) == 0:
            logger.error("Empty dataset provided for bias analysis")
            return {}

        if len(labels) != len(preds) or len(labels) != len(features):
            logger.error(f"Dimension mismatch: labels={len(labels)}, predictions={len(preds)}, features={len(features)}")
            return {}

        feature_probs, feature_bias = self._compute_feature_metrics(labels, features)
        confusion_matrix = self._compute_confusion_matrix(labels, preds)
        male_metrics, female_metrics = self._compute_gender_metrics(confusion_matrix)
        bias_scores = self._compute_bias_scores(confusion_matrix, male_metrics, female_metrics, feature_bias)

        logger.info("Bias analysis completed successfully")

        male_perf = male_metrics["tpr"]
        female_perf = female_metrics["tpr"]
        overall_accuracy = (confusion_matrix["true_male"]["predicted_male"] + confusion_matrix["true_female"]["predicted_female"]) / len(labels)

        logger.debug(f"Summary statistics - Overall accuracy: {overall_accuracy:.3f}, Male performance: {male_perf:.3f}, Female performance: {female_perf:.3f}")

        max_bias_metric = max(bias_scores.items(), key=lambda x: x[1])
        logger.debug(f"Highest bias detected in: {max_bias_metric[0]} = {max_bias_metric[1]:.3f}")

        return {
            "feature_probabilities": feature_probs,
            "confusion_matrix": confusion_matrix,
            "male_metrics": male_metrics,
            "female_metrics": female_metrics,
            "bias_scores": bias_scores,
        }


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
