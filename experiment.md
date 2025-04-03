### File: `experiment/utils.py`

```py
import logging
import os
import sys

# isort: off
from config import Config


class CustomFormatter(logging.Formatter):
    """Applies level-specific colors to log messages using a provided format string."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.formats = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno, self.fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_logger(
    config: Config,
    console_level: int = logging.ERROR,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Creates a logger with both console and file handler."""
    id = config.experiment_id
    format = f"%(asctime)s -  %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    os.makedirs(config.output.log_path, exist_ok=True)
    filename = os.path.join(config.output.log_path, f"{id}.log")

    logger = logging.getLogger(id)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CustomFormatter(format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

```

### File: `experiment/masker.py`

```py
import json
import logging
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult

# isort: off
from config import Config
from datatypes import BoundingBox, Feature, FeatureDetails


class FeatureMasker:
    """Detects facial features, provides bounding boxes, and applies masks."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initializes the FeatureMasker with configuration and logger."""
        self.config = config
        self.logger = logger
        self.landmarker = self._load_landmarker()
        self.feature_indices_map = self._load_feature_indices_map()
        self.logger.info("Completed feature masker initialization")

    def _load_landmarker(self) -> FaceLandmarker:
        """Loads the MediaPipe face landmarker model from HuggingFace Hub."""
        self.logger.debug("Starting landmarker model loading")
        try:
            model_path = hf_hub_download(
                repo_id="rixmape/biasx-models",
                filename="mediapipe_landmarker.task",
                repo_type="model",
            )
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            landmarker = FaceLandmarker.create_from_options(options)
            self.logger.info("Completed loading landmark detection model")
            return landmarker
        except Exception as e:
            self.logger.error(f"Failed to load landmarker model: {e}", exc_info=True)
            raise

    def _load_feature_indices_map(self) -> Dict[Feature, List[int]]:
        """Loads the mapping from facial features to landmark indices from JSON."""
        self.logger.debug("Starting feature indices map loading")
        try:
            map_path = hf_hub_download(
                repo_id="rixmape/biasx-models",
                filename="landmark_map.json",
                repo_type="model",
            )
            with open(map_path, "r") as f:
                raw_map = json.load(f)

            feature_map = {Feature(key): value for key, value in raw_map.items()}
            self.logger.info("Completed loading feature indices map")
            return feature_map
        except FileNotFoundError:
            self.logger.error(f"Feature map file not found: path={map_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON: path={map_path}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Failed to load feature indices map: {e}", exc_info=True)
            raise

    def _get_pixel_landmarks(
        self,
        image_np: np.ndarray,
        image_id: str,
    ) -> List[Tuple[int, int]]:
        """Detects face landmarks in an image and returns pixel coordinates."""
        if image_np.dtype in [np.float32, np.float64]:
            image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        elif image_np.dtype == np.uint8:
            image_uint8 = image_np
        else:
            self.logger.error(f"[{image_id}] Unsupported image dtype: {image_np.dtype}")
            return []

        if len(image_uint8.shape) != 3 or image_uint8.shape[-1] != 3:
            self.logger.error(f"[{image_id}] Invalid image shape/channels: {image_uint8.shape}")
            return []

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
            detection_result: Optional[FaceLandmarkerResult] = self.landmarker.detect(mp_image)
        except Exception as e:
            self.logger.error(f"[{image_id}] MediaPipe landmark detection failed: {e}", exc_info=True)
            return []

        if not detection_result or not detection_result.face_landmarks:
            return []

        landmarks = detection_result.face_landmarks[0]
        img_size_h, img_size_w = image_np.shape[:2]
        pixel_coords = [(int(pt.x * img_size_w), int(pt.y * img_size_h)) for pt in landmarks]

        return pixel_coords

    def _get_feature_bbox(
        self,
        pixel_coords: List[Tuple[int, int]],
        feature: Feature,
        image_id: str,
        img_height: int,
        img_width: int,
    ) -> Optional[BoundingBox]:
        """Calculates the padded bounding box for a specific feature from landmarks."""
        indices = self.feature_indices_map.get(feature)
        if not indices:
            self.logger.warning(f"[{image_id}] No indices found: feature='{feature.value}'")
            return None
        if max(indices) >= len(pixel_coords):
            self.logger.warning(f"[{image_id}] Index out of bounds: feature='{feature.value}', max_index={max(indices)}, landmarks={len(pixel_coords)}")
            return None

        try:
            points = [pixel_coords[i] for i in indices]
            min_x = min(x for x, y in points)
            min_y = min(y for x, y in points)
            max_x = max(x for x, y in points)
            max_y = max(y for x, y in points)
        except IndexError:
            self.logger.error(
                f"[{image_id}] IndexError accessing coords: feature='{feature.value}', indices={indices}, total_coords={len(pixel_coords)}",
                exc_info=True,
            )
            return None

        pad = self.config.core.mask_pixel_padding
        min_x_pad = max(0, min_x - pad)
        min_y_pad = max(0, min_y - pad)
        max_x_pad = min(img_width, max_x + pad)
        max_y_pad = min(img_height, max_y + pad)

        if min_x_pad >= max_x_pad or min_y_pad >= max_y_pad:
            self.logger.warning(f"[{image_id}] Invalid bbox after padding: feature='{feature.value}', box=({min_x_pad}, {min_y_pad}, {max_x_pad}, {max_y_pad})")
            return None

        return BoundingBox(min_x=min_x_pad, min_y=min_y_pad, max_x=max_x_pad, max_y=max_y_pad)

    def apply_mask(self, image_np: np.ndarray, label: int, image_id: str) -> np.ndarray:
        """Applies configured masks (zeros out regions) if the label matches the target."""
        if not self.config.core.mask_gender or label != self.config.core.mask_gender.value:
            return image_np
        if not self.config.core.mask_features:
            self.logger.warning(f"[{image_id}] Masking requested but no features specified: mask_gender={self.config.core.mask_gender.name}")
            return image_np

        pixel_coords = self._get_pixel_landmarks(image_np, image_id)
        if not pixel_coords:
            self.logger.warning(f"[{image_id}] Skipping masking: no landmarks found")
            return image_np

        masked_image = image_np.copy()
        img_height, img_width = image_np.shape[:2]
        applied_mask_count = 0

        for feature in self.config.core.mask_features:
            bbox = self._get_feature_bbox(pixel_coords, feature, image_id, img_height, img_width)
            if bbox:
                masked_image[bbox.min_y : bbox.max_y, bbox.min_x : bbox.max_x] = 0
                applied_mask_count += 1

        if applied_mask_count == 0:
            self.logger.warning(f"[{image_id}] No valid features found to mask")

        return masked_image

    def get_features(
        self,
        image_np: np.ndarray,
        image_id: str,
    ) -> List[FeatureDetails]:
        """Detects all facial features and returns their details including bounding boxes."""
        pixel_coords = self._get_pixel_landmarks(image_np, image_id)
        if not pixel_coords:
            self.logger.warning(f"[{image_id}] Cannot get features: no landmarks found")
            return []

        img_height, img_width = image_np.shape[:2]
        detected_features: List[FeatureDetails] = []

        for feature in self.feature_indices_map.keys():
            bbox = self._get_feature_bbox(pixel_coords, feature, image_id, img_height, img_width)
            if bbox:
                feature_detail = FeatureDetails(feature=feature, bbox=bbox)
                detected_features.append(feature_detail)

        if not detected_features:
            self.logger.warning(f"[{image_id}] No valid feature bboxes generated")

        return detected_features

```

### File: `experiment/dataset.py`

```py
import logging
import os
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# isort: off
from datatypes import OutputLevel, DatasetSplit, Gender
from config import Config
from masker import FeatureMasker

class DatasetGenerator:

    def __init__(self, config: Config, logger: logging.Logger, feature_masker: FeatureMasker):
        self.config = config
        self.logger = logger
        self.feature_masker = feature_masker
        self.logger.info("Completed dataset generator initialization")

    def _load_raw_dataframe(self) -> pd.DataFrame:
        self.logger.info(f"Downloading {self.config.dataset.source_name.value} dataset.")
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset.source_name.value}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path, columns=["image_id", "image", "gender", "race", "age"])
        self.logger.info(f"Raw dataset loaded: {len(df)} rows.")
        return df

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["image_id"] = df["image_id"].astype(str).str[:16]
        df["image_bytes"] = df["image"].apply(lambda x: x["bytes"])
        df = df.drop(columns=["image"])

        df["age"] = df["age"].astype(int)
        infant_mask = df["age"] > 0
        df = df[infant_mask]

        self.logger.info(f"Processed dataset: {len(df)} rows remaining after filtering.")
        return df

    def _sample_by_strata(self, df: pd.DataFrame, target_sample_size: int, seed: int) -> pd.DataFrame:
        samples = []
        total_rows = len(df)
        if total_rows == 0:
            return pd.DataFrame(columns=df.columns)

        for strata_name, group in df.groupby("strata"):
            group_size = len(group)
            group_sample_size = max(1, round(target_sample_size * (group_size / total_rows)))
            replacement_needed = group_size < group_sample_size

            if replacement_needed:
                self.logger.debug(f"Strata '{strata_name}': Using replacement sampling ({group_size} available, {group_sample_size} needed).")

            sample = group.sample(n=group_sample_size, random_state=seed, replace=replacement_needed)
            samples.append(sample)

        return pd.concat(samples)

    def _get_sampled_gender_subset(self, df: pd.DataFrame, gender: Gender, gender_target_size: int, seed: int) -> pd.DataFrame:
        gender_df = df[df["gender"] == gender.value].copy()
        sample_size = len(gender_df)
        self.logger.info(f"Targeting {gender_target_size} from {sample_size} available {gender.name.lower()} images.")

        if sample_size < gender_target_size:
            self.logger.warning(f"Available {gender.name.lower()} samples ({sample_size}) is less than target size ({gender_target_size}). Using replacement sampling.")
            if sample_size == 0:
                return pd.DataFrame(columns=df.columns)

        strata_sample = self._sample_by_strata(gender_df, gender_target_size, seed)
        actual_size = len(strata_sample)

        if actual_size < gender_target_size:
            self.logger.warning(f"Actual {gender.name.lower()} sample size ({actual_size}) is less than target ({gender_target_size}) after stratified sampling.")

        return strata_sample

    def _sample_by_gender(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        df["strata"] = df["race"].astype(str) + "_" + df["age"].astype(str)
        self.logger.info(f"Found {df['strata'].nunique()} unique strata for sampling.")

        target_female_proportion = 1.0 - self.config.core.target_male_proportion
        target_male_size = round(self.config.dataset.target_size * self.config.core.target_male_proportion)
        target_female_size = round(self.config.dataset.target_size * target_female_proportion)

        male_samples = self._get_sampled_gender_subset(df, Gender.MALE, target_male_size, seed)
        female_samples = self._get_sampled_gender_subset(df, Gender.FEMALE, target_female_size, seed)

        combined_df = pd.concat([male_samples, female_samples])
        if combined_df.empty:
            self.logger.warning("Sampling resulted in an empty DataFrame.")
            return combined_df

        combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        combined_df = combined_df.drop(columns=["strata"])

        final_size = len(combined_df)
        self.logger.info(f"Sampling complete. Final dataset size: {final_size} ({final_size / max(self.config.dataset.target_size, 1):.1%} of target {self.config.dataset.target_size}).")
        if final_size < self.config.dataset.target_size:
            self.logger.warning(f"Final dataset size ({final_size}) is less than target size ({self.config.dataset.target_size}).")

        return combined_df

    def _preprocess_single_image(self, image_bytes: bytes, label: int, image_id: str, purpose: DatasetSplit) -> np.ndarray:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [self.config.dataset.image_size, self.config.dataset.image_size])
        image = tf.cast(image, tf.float32) / 255.0
        image_np = image.numpy()

        if purpose == DatasetSplit.TRAIN:
            image_np = self.feature_masker.apply_mask(image_np, label, image_id)

        if self.config.dataset.use_grayscale:
            image_np = tf.image.rgb_to_grayscale(image_np).numpy()

        return image_np.astype(np.float32)

    def _create_generator(self, df: pd.DataFrame, purpose: DatasetSplit) -> Generator[Tuple[np.ndarray, int, str], None, None]:
        for _, row in df.iterrows():
            image_bytes = row["image_bytes"]
            label = int(row["gender"])
            image_id = row["image_id"]

            processed_image = self._preprocess_single_image(image_bytes, label, image_id, purpose)
            yield processed_image, label, image_id

    def _create_tf_dataset(self, df: pd.DataFrame, purpose: DatasetSplit) -> tf.data.Dataset:
        output_shape = (
            self.config.dataset.image_size,
            self.config.dataset.image_size,
            1 if self.config.dataset.use_grayscale else 3,
        )

        output_signature = (
            tf.TensorSpec(shape=output_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )

        dataset = tf.data.Dataset.from_generator(lambda: self._create_generator(df, purpose), output_signature=output_signature)
        return dataset

    def _save_images_to_disk(self, dataset: tf.data.Dataset, purpose: DatasetSplit) -> None:
        path = os.path.join(
            self.config.output.base_path,
            self.config.experiment_id,
            f"{purpose.value.lower()}_images",
        )
        os.makedirs(path, exist_ok=True)

        image_count = 0
        for image, _, image_id_tensor in dataset:
            image_np = image.numpy()
            image_id = image_id_tensor.numpy().decode("utf-8")

            if image_np.dtype in [np.float32, np.float64]:
                image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
            if image_np.shape[-1] == 1:
                image_np = np.squeeze(image_np, axis=-1)

            filename = f"{purpose.value.lower()}_{image_id}.png"
            filepath = os.path.join(path, filename)
            tf.keras.utils.save_img(filepath, image_np)
            image_count += 1

        self.logger.info(f"Saved {image_count} images for {purpose} to {path}")

    def _build_dataset_split(
        self,
        df: pd.DataFrame,
        purpose: DatasetSplit,
    ) -> tf.data.Dataset:
        self.logger.info(f"Creating {purpose} from {len(df)} samples.")
        dataset = self._create_tf_dataset(df, purpose)

        if self.config.output.level == OutputLevel.FULL:
            self._save_images_to_disk(dataset, purpose)

        dataset = dataset.cache()
        dataset = dataset.batch(self.config.model.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.logger.info(f"{purpose} build complete.")
        return dataset

    def _split_dataframe(self, df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.logger.info("Splitting dataset into train, validation, and test sets.")

        train_val_df, test_df = train_test_split(df, test_size=self.config.dataset.test_ratio, random_state=seed, stratify=df["gender"])
        adjusted_val_ratio = self.config.dataset.validation_ratio / (1.0 - self.config.dataset.test_ratio)
        train_df, val_df = train_test_split(train_val_df, test_size=adjusted_val_ratio, random_state=seed, stratify=train_val_df["gender"])

        self.logger.info(f"Actual splits: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}.")
        return train_df, val_df, test_df

    def prepare_datasets(
        self,
        seed: int,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self.logger.info(f"Starting dataset preparation")

        raw_df = self._load_raw_dataframe()
        processed_df = self._process_dataframe(raw_df)
        sampled_df = self._sample_by_gender(processed_df, seed)
        train_df, val_df, test_df = self._split_dataframe(sampled_df, seed)

        train_data = self._build_dataset_split(train_df, DatasetSplit.TRAIN)
        val_data = self._build_dataset_split(val_df, DatasetSplit.VALIDATION)
        test_data = self._build_dataset_split(test_df, DatasetSplit.TEST)

        self.logger.info(f"Completed dataset preparation")
        return train_data, val_data, test_data

```

### File: `experiment/runner.py`

```py
import json
import os
import random
import warnings
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from analyzer import BiasAnalyzer
from config import Config
from dataset import DatasetGenerator
from datatypes import AnalysisResult, OutputLevel, ExperimentResult, Gender, Explanation, ModelHistory
from explainer import VisualExplainer
from masker import FeatureMasker
from model import ModelTrainer
from utils import create_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


class ExperimentRunner:
    """Manages the setup, execution, and analysis of a single bias analysis experiment."""

    def __init__(self, config: Config):
        """Initializes the experiment runner with configuration."""
        self.config = config
        self.logger = create_logger(config)
        self._set_random_seeds()

        self.feature_masker = FeatureMasker(self.config, self.logger)
        self.dataset_generator = DatasetGenerator(self.config, self.logger, self.feature_masker)
        self.model_trainer = ModelTrainer(self.config, self.logger)
        self.visual_explainer = VisualExplainer(self.config, self.logger, self.feature_masker)
        self.bias_analyzer = BiasAnalyzer(self.config, self.logger)

        self.logger.info(f"Completed experiment runner initialization: id={self.config.experiment_id}")

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
        save_results = self.config.output.level in [
            OutputLevel.RESULTS_ONLY,
            OutputLevel.FULL,
        ]

        result = ExperimentResult(
            id=self.config.experiment_id,
            config=self.config.model_dump(mode="json"),
            history=history if save_results else None,
            analysis=analysis if save_results else None,
        )

        filename = f"{self.config.experiment_id}.json"
        path = os.path.join(self.config.output.base_path, filename)

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

        splits = self.dataset_generator.prepare_datasets(self.config.core.random_seed)
        train_data, val_data, test_data = splits

        model, history = self.model_trainer.get_model_and_history(train_data, val_data)
        explanations = self._get_all_explanations(test_data, model)
        analysis = self.bias_analyzer.get_bias_analysis(explanations)
        result = self._save_result(history, analysis)

        self.logger.info(f"Completed experiment run")
        return result

```

### File: `experiment/config.py`

```py
import hashlib
from functools import cache, cached_property
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# isort: off
from datatypes import OutputLevel, DatasetSource, Feature, Gender


# TODO: Add `protected_attribute` field
class CoreConfig(BaseModel):
    target_male_proportion: float = Field(..., ge=0.0, le=1.0)
    mask_gender: Optional[Gender] = Field(default=None)
    mask_features: Optional[List[Feature]] = Field(default=None)
    mask_pixel_padding: int = Field(default=2, ge=0)
    key_feature_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    random_seed: int = Field(default=42, ge=0)


class DatasetConfig(BaseModel):
    source_name: DatasetSource = DatasetSource.UTKFACE
    target_size: int = Field(default=5000, gt=0)
    validation_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    image_size: int = Field(default=48, gt=0)
    use_grayscale: bool = False

    @model_validator(mode="after")
    def check_split_ratios_sum(self) -> "DatasetConfig":
        if self.validation_ratio + self.test_ratio >= 1.0:
            raise ValueError("Validation and test ratios must sum to less than 1.0")
        return self


class ModelConfig(BaseModel):
    batch_size: int = Field(default=64, gt=0)
    epochs: int = Field(default=10, gt=0)


# TODO: Add `console_level` and `file_level` fields
class OutputConfig(BaseModel):
    base_path: str = "outputs"
    log_path: str = "logs"
    level: OutputLevel = OutputLevel.FULL

    @field_validator("base_path", "log_path")
    @classmethod
    def check_paths(cls, v: str) -> str:
        if not v:
            raise ValueError("Directory names cannot be empty")
        return v


class Config(BaseModel):
    core: CoreConfig = Field(default_factory=CoreConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @cached_property
    def experiment_id(self) -> str:
        """Generates a unique experiment ID based on the current configuration."""
        config_json = self.model_dump_json()
        hash_object = hashlib.sha256(config_json.encode())
        return hash_object.hexdigest()[:16]

```

### File: `experiment/analyzer.py`

```py
import logging
from collections import defaultdict
from typing import List

import numpy as np

# isort: off
from datatypes import AnalysisResult, BiasMetrics, Feature, FeatureDistribution, Gender, GenderPerformanceMetrics, Explanation
from config import Config


class BiasAnalyzer:

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.logger.info("Completed bias analyzer initialization")

    def _compute_feature_distributions(
        self,
        image_details: List[Explanation],
    ) -> List[FeatureDistribution]:
        self.logger.info("Computing feature distributions based on key features.")
        feature_counts = defaultdict(lambda: defaultdict(int))
        label_counts = defaultdict(int)

        for image_detail in image_details:
            true_label = image_detail.label
            gender_name = true_label.name
            label_counts[gender_name] += 1

            for feature_detail in image_detail.detected_features:
                if feature_detail.is_key_feature:
                    feature_counts[feature_detail.feature][gender_name] += 1

        male_total = label_counts[Gender.MALE.name]
        female_total = label_counts[Gender.FEMALE.name]
        distributions = []
        all_possible_features = set(Feature)

        for feature_enum in all_possible_features:
            gender_count = feature_counts[feature_enum]
            male_prob = gender_count[Gender.MALE.name] / max(male_total, 1)
            female_prob = gender_count[Gender.FEMALE.name] / max(female_total, 1)
            dist = FeatureDistribution(
                feature=feature_enum,
                male_distribution=male_prob,
                female_distribution=female_prob,
            )
            distributions.append(dist)

        return distributions

    def _compute_gender_performance_metrics(
        self,
        positive_class: Gender,
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> GenderPerformanceMetrics:
        self.logger.info(f"Computing performance metrics for {positive_class.name} as Positive.")

        positive_val = positive_class.value

        is_positive_actual = true_labels == positive_val
        is_negative_actual = true_labels != positive_val
        is_positive_pred = predicted_labels == positive_val
        is_negative_pred = predicted_labels != positive_val

        tp = int(np.sum(is_positive_actual & is_positive_pred))
        fn = int(np.sum(is_positive_actual & is_negative_pred))
        fp = int(np.sum(is_negative_actual & is_positive_pred))
        tn = int(np.sum(is_negative_actual & is_negative_pred))

        metrics = GenderPerformanceMetrics(positive_class=positive_class, tp=tp, fp=fp, tn=tn, fn=fn)

        self.logger.debug(f"{positive_class.name} metrics: {metrics.model_dump()}")
        return metrics

    def _compute_bias_metrics(
        self,
        male_metrics: GenderPerformanceMetrics,
        female_metrics: GenderPerformanceMetrics,
        feature_distributions: List[FeatureDistribution],
    ) -> BiasMetrics:
        self.logger.info("Computing overall bias scores.")

        males_predicted_positive = male_metrics.tp + male_metrics.fp
        females_predicted_positive = female_metrics.tp + female_metrics.fp
        total_males = male_metrics.tp + male_metrics.fn
        total_females = female_metrics.tp + female_metrics.fn

        male_select_rate = males_predicted_positive / max(total_males, 1)
        female_select_rate = females_predicted_positive / max(total_females, 1)
        demographic_parity = abs(male_select_rate - female_select_rate)

        equalized_odds = abs(male_metrics.tpr - female_metrics.tpr)
        conditional_use_accuracy_equality = abs(male_metrics.ppv - female_metrics.ppv)
        mean_feature_distribution_bias = np.mean([dist.distribution_bias for dist in feature_distributions]) if feature_distributions else 0.0

        bias_metrics_result = BiasMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            conditional_use_accuracy_equality=conditional_use_accuracy_equality,
            mean_feature_distribution_bias=mean_feature_distribution_bias,
        )

        self.logger.debug(f"Overall bias metrics: {bias_metrics_result.model_dump()}")
        return bias_metrics_result

    def get_bias_analysis(
        self,
        image_details: List[Explanation],
    ) -> AnalysisResult:
        self.logger.info(f"Starting bias analysis on {len(image_details)} samples.")

        true_labels = np.array([img.label.value for img in image_details])
        predicted_labels = np.array([img.prediction.value for img in image_details])

        feature_distributions = self._compute_feature_distributions(image_details)
        male_perf_metrics = self._compute_gender_performance_metrics(Gender.MALE, true_labels, predicted_labels)
        female_perf_metrics = self._compute_gender_performance_metrics(Gender.FEMALE, true_labels, predicted_labels)
        bias_metrics = self._compute_bias_metrics(male_perf_metrics, female_perf_metrics, feature_distributions)

        self.logger.info("Bias analysis completed successfully.")

        return AnalysisResult(
            feature_distributions=feature_distributions,
            male_performance_metrics=male_perf_metrics,
            female_performance_metrics=female_perf_metrics,
            bias_metrics=bias_metrics,
            analyzed_images=image_details,
        )

```

### File: `experiment/explainer.py`

```py
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from config import Config
from datatypes import OutputLevel, FeatureDetails
from masker import FeatureMasker


class VisualExplainer:
    """Generates visual explanations (heatmaps) and calculates feature attention."""

    def __init__(self, config: Config, logger: logging.Logger, masker: FeatureMasker):
        """Initializes the VisualExplainer with configuration, masker, and logger."""
        self.config = config
        self.logger = logger
        self.masker = masker
        self.logger.info("Completed visual explainer initialization")

    def _calculate_heatmap(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        true_label_index: int,
        image_id: str,
    ) -> np.ndarray:
        """Calculates a normalized heatmap for a given image and label."""
        target_class = lambda output: output[0][true_label_index]
        image_batch = np.expand_dims(image_np.astype(np.float32), axis=0)
        target_layer = "block3_conv3"

        model_layers = [layer.name for layer in model.layers]
        if target_layer not in model_layers:
            self.logger.error(f"[{image_id}] Target layer '{target_layer}' not found: model_layers={model_layers}")
            return np.zeros(image_np.shape[:2], dtype=np.float32)

        try:
            heatmap = heatmap_generator(target_class, image_batch, penultimate_layer=target_layer)[0]
        except Exception as e:
            self.logger.error(f"[{image_id}] Heatmap generation via GradCAM++ failed: {e}", exc_info=True)
            return np.zeros(image_np.shape[:2], dtype=np.float32)

        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val <= min_val:
            self.logger.warning(f"[{image_id}] Heatmap range invalid (max <= min), returning zeros.")
            return np.zeros_like(heatmap, dtype=np.float32)

        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
        return normalized_heatmap.astype(np.float32)

    def _calculate_single_feature_attention(
        self,
        feature: FeatureDetails,
        heatmap: np.ndarray,
        image_id: str,
    ) -> float:
        """Calculates the mean attention score within a feature's bounding box."""
        heatmap_height, heatmap_width = heatmap.shape[:2]

        min_y, max_y = max(0, feature.bbox.min_y), min(heatmap_height, feature.bbox.max_y)
        min_x, max_x = max(0, feature.bbox.min_x), min(heatmap_width, feature.bbox.max_x)

        if min_y >= max_y or min_x >= max_x:
            self.logger.debug(f"[{image_id}] Invalid/Empty attention region after clamping for feature {feature.feature.name}: box=({min_x}, {min_y}, {max_x}, {max_y})")
            return 0.0

        feature_attention_region = heatmap[min_y:max_y, min_x:max_x]

        if feature_attention_region.size == 0:
            self.logger.debug(f"[{image_id}] Feature attention region is empty for feature {feature.feature.name}: box=({min_x}, {min_y}, {max_x}, {max_y})")
            return 0.0

        return float(np.mean(feature_attention_region))

    def _save_heatmap(
        self,
        heatmap: np.ndarray,
        image_id: str,
    ) -> Optional[str]:
        """Saves the heatmap array to disk if artifact saving is enabled."""
        if self.config.output.level != OutputLevel.FULL:
            self.logger.debug(f"[{image_id}] Skipping heatmap saving: artifact level is {self.config.output.level.name}")
            return None

        path = os.path.join(self.config.output.base_path, self.config.experiment_id, "heatmaps")
        os.makedirs(path, exist_ok=True)

        filename = f"{image_id}.npy"
        filepath = os.path.join(path, filename)

        try:
            np.save(filepath, heatmap.astype(np.float16))
            heatmap_rel_path = os.path.relpath(filepath, self.config.output.base_path)
            return heatmap_rel_path
        except Exception as e:
            self.logger.error(f"[{image_id}] Failed to save heatmap to {filepath}: {e}", exc_info=True)
            return None

    def _compute_feature_details(
        self,
        features: List[FeatureDetails],
        heatmap: np.ndarray,
        image_id: str,
    ) -> List[FeatureDetails]:
        """Computes and adds attention scores and key feature flags to feature details."""
        if not features:
            self.logger.warning(f"[{image_id}] No features provided for attention calculation.")
            return []

        if heatmap is None or heatmap.size == 0 or np.all(heatmap == 0):
            self.logger.warning(f"[{image_id}] Cannot compute feature attention: heatmap is invalid or all zeros. Setting scores to 0.")
            for feature_detail in features:
                feature_detail.attention_score = 0.0
                feature_detail.is_key_feature = False
            return features

        key_features_found = 0
        for feature_detail in features:
            attention_score = self._calculate_single_feature_attention(feature_detail, heatmap, image_id)
            is_key = attention_score >= self.config.core.key_feature_threshold

            feature_detail.attention_score = float(attention_score)
            feature_detail.is_key_feature = bool(is_key)

            if is_key:
                key_features_found += 1

        if key_features_found == 0:
            self.logger.info(f"[{image_id}] No key features above threshold {self.config.core.key_feature_threshold}.")

        return features

    def get_heatmap_generator(
        self,
        model: tf.keras.Model,
    ) -> GradcamPlusPlus:
        """Creates and returns a GradCAM++ heatmap generator for the given model."""
        try:
            replace_to_linear = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
            generator = GradcamPlusPlus(model, model_modifier=replace_to_linear, clone=True)
            self.logger.debug("Successfully created GradCAM++ heatmap generator.")
            return generator
        except Exception as e:
            self.logger.error(f"Failed to create GradCAM++ generator: {e}", exc_info=True)
            raise

    def generate_explanation(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        label: int,
        image_id: str,
    ) -> Tuple[List[FeatureDetails], Optional[str]]:
        """Generates feature details with attention and saves heatmap for a single image."""
        heatmap = self._calculate_heatmap(heatmap_generator, model, image_np, label, image_id)
        heatmap_path = self._save_heatmap(heatmap, image_id)

        detected_features = self.masker.get_features(image_np, image_id)
        if not detected_features:
            self.logger.warning(f"[{image_id}] No features detected by masker; returning empty feature list for explanation.")
            return [], heatmap_path

        feature_details_with_attention = self._compute_feature_details(detected_features, heatmap, image_id)

        return feature_details_with_attention, heatmap_path

```

### File: `experiment/model.py`

```py
import logging
from typing import Tuple

import tensorflow as tf

# isort: off
from config import Config
from datatypes import ModelHistory


class ModelTrainer:
    """Builds and trains the CNN model."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initializes the model trainer with configuration and logger."""
        self.config = config
        self.logger = logger

    def _build_model(self) -> tf.keras.Model:
        """Constructs the CNN model architecture using TensorFlow/Keras."""
        input_channels = 1 if self.config.dataset.use_grayscale else 3
        input_shape = (self.config.dataset.image_size, self.config.dataset.image_size, input_channels)

        self.logger.info(f"Starting model building: input_shape={input_shape}")

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))

        conv_blocks = [(64, 2), (128, 2), (256, 3)]
        for idx, (filters, layers_count) in enumerate(conv_blocks, start=1):
            for i in range(1, layers_count + 1):
                conv_layer = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{idx}_conv{i}")
                model.add(conv_layer)
            pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{idx}_pool")
            model.add(pooling_layer)

        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(2, activation="softmax", name="output"))

        model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        total_params = model.count_params()
        self.logger.info(f"Completed model building: parameters={total_params:,}")
        return model

    def get_model_and_history(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
    ) -> Tuple[tf.keras.Model, ModelHistory]:
        """Trains the model on the provided data and returns the model and training history."""

        train_data_fit = train_data.map(lambda image, label, _: (image, label))
        val_data_fit = val_data.map(lambda image, label, _: (image, label))

        model = self._build_model()
        self.logger.info(f"Starting model training: epochs={self.config.model.epochs}, batch_size={self.config.model.batch_size}.")

        history = model.fit(train_data_fit, validation_data=val_data_fit, epochs=self.config.model.epochs, verbose=0)
        history = ModelHistory(
            train_loss=history.history["loss"],
            train_accuracy=history.history["accuracy"],
            val_loss=history.history["val_loss"],
            val_accuracy=history.history["val_accuracy"],
        )

        self.logger.info(f"Completed model training: train_acc={history.train_accuracy[-1]:.4f}, val_acc={history.val_accuracy[-1]:.4f}")
        return model, history

```

### File: `experiment/datatypes.py`

```py
from enum import Enum, auto
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationInfo, computed_field, field_validator


class OutputLevel(Enum):
    NONE = auto()
    RESULTS_ONLY = auto()
    FULL = auto()


class DatasetSource(Enum):
    UTKFACE = "utkface"
    FAIRFACE = "fairface"


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Gender(Enum):
    MALE = 0
    FEMALE = 1


class Feature(Enum):
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    LIPS = "lips"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    CHIN = "chin"
    FOREHEAD = "forehead"
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"


class BoundingBox(BaseModel):
    min_x: int = Field(..., ge=0)
    min_y: int = Field(..., ge=0)
    max_x: int = Field(..., ge=0)
    max_y: int = Field(..., ge=0)

    @computed_field
    @property
    def area(self) -> int:
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        return max(0, width) * max(0, height)

    @field_validator("max_x")
    @classmethod
    def check_x_coords(cls, v: int, info: ValidationInfo) -> int:
        if "min_x" in info.data and v < info.data["min_x"]:
            raise ValueError("max_x must be greater than or equal to min_x")
        return v

    @field_validator("max_y")
    @classmethod
    def check_y_coords(cls, v: int, info: ValidationInfo) -> int:
        if "min_y" in info.data and v < info.data["min_y"]:
            raise ValueError("max_y must be greater than or equal to min_y")
        return v


class FeatureDetails(BaseModel):
    feature: Feature
    bbox: BoundingBox = Field(default_factory=BoundingBox)
    attention_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_key_feature: bool = Field(default=False)


class GenderPerformanceMetrics(BaseModel):
    positive_class: Gender
    tp: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    tn: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)

    @computed_field
    @property
    def tpr(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @computed_field
    @property
    def fpr(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @computed_field
    @property
    def tnr(self) -> float:
        return self.tn / max(self.tn + self.fp, 1)

    @computed_field
    @property
    def fnr(self) -> float:
        return self.fn / max(self.fn + self.tp, 1)

    @computed_field
    @property
    def ppv(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @computed_field
    @property
    def npv(self) -> float:
        return self.tn / max(self.tn + self.fn, 1)

    @computed_field
    @property
    def fdr(self) -> float:
        return self.fp / max(self.fp + self.tp, 1)

    @computed_field
    @property
    def _for(self) -> float:
        return self.fn / max(self.fn + self.tn, 1)


class BiasMetrics(BaseModel):
    demographic_parity: float = Field(..., ge=0.0)
    equalized_odds: float = Field(..., ge=0.0)
    conditional_use_accuracy_equality: float = Field(..., ge=0.0)
    mean_feature_distribution_bias: float = Field(..., ge=0.0)


class Explanation(BaseModel):
    image_id: str = Field(..., min_length=1)
    label: Gender
    prediction: Gender
    confidence_scores: List[float] = Field(default_factory=list)
    heatmap_path: Optional[str] = Field(default=None)
    detected_features: List[FeatureDetails] = Field(default_factory=list)

    @field_validator("confidence_scores")
    @classmethod
    def check_confidence_scores(cls, v: List[float]) -> List[float]:
        if len(v) != len(Gender):
            raise ValueError(f"confidence_scores must have length {len(Gender)}")
        if not all(0.0 <= score <= 1.0 for score in v):
            raise ValueError("All confidence scores must be between 0.0 and 1.0")
        return v


class FeatureDistribution(BaseModel):
    feature: Feature
    male_distribution: float = Field(..., ge=0.0, le=1.0)
    female_distribution: float = Field(..., ge=0.0, le=1.0)

    @computed_field
    @property
    def distribution_bias(self) -> float:
        return abs(self.male_distribution - self.female_distribution)


class AnalysisResult(BaseModel):
    feature_distributions: List[FeatureDistribution] = Field(default_factory=list)
    male_performance_metrics: Optional[GenderPerformanceMetrics] = Field(default=None)
    female_performance_metrics: Optional[GenderPerformanceMetrics] = Field(default=None)
    bias_metrics: Optional[BiasMetrics] = Field(default=None)
    analyzed_images: List[Explanation] = Field(default_factory=list)


class ModelHistory(BaseModel):
    train_loss: List[float] = Field(default_factory=list)
    train_accuracy: List[float] = Field(default_factory=list)
    val_loss: List[float] = Field(default_factory=list)
    val_accuracy: List[float] = Field(default_factory=list)

    @field_validator("train_accuracy", "val_accuracy")
    @classmethod
    def check_accuracy_values(cls, v: List[float]) -> List[float]:
        if not all(0.0 <= acc <= 1.0 for acc in v):
            raise ValueError("Accuracy values must be between 0.0 and 1.0")
        return v

    @field_validator("train_loss", "val_loss")
    @classmethod
    def check_loss_values(cls, v: List[float]) -> List[float]:
        if not all(loss >= 0.0 for loss in v):
            raise ValueError("Loss values must be non-negative")
        return v


class ExperimentResult(BaseModel):
    id: str = Field(..., min_length=1)
    config: dict = Field(default_factory=dict)
    history: Optional[ModelHistory] = Field(default=None)
    analysis: Optional[AnalysisResult] = Field(default=None)

```

