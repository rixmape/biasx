import os
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from config import Config
from datatypes import ArtifactSavingLevel, DatasetSplit, Gender
from huggingface_hub import hf_hub_download
from masker import FeatureMasker
from sklearn.model_selection import train_test_split
from utils import setup_logger


class DatasetGenerator:

    def __init__(self, config: Config, feature_masker: FeatureMasker, log_path: str, exp_id: str):
        self.config = config
        self.feature_masker = feature_masker
        self.logger = setup_logger(name="dataset_generator", log_path=log_path, id=exp_id)
        self.logger.info("Completed dataset generator initialization")

    def _load_raw_dataframe(self) -> pd.DataFrame:
        self.logger.info(f"Downloading {self.config.dataset.source_name.value} dataset.")
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset.source_name.value}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path, columns=["image_id", "image", "gender", "race", "age"])
        self.logger.info(f"Raw dataset loaded: {len(df)} rows.")
        return df

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df["image_id"] = df["image_id"].astype(str).str[:8]
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

    def _save_images_to_disk(self, dataset: tf.data.Dataset, purpose: DatasetSplit, base_output_directory: str) -> None:
        specific_output_directory = os.path.join(base_output_directory, f"{purpose.value.lower()}_images")
        os.makedirs(specific_output_directory, exist_ok=True)

        image_count = 0
        for image, _, image_id_tensor in dataset:
            image_np = image.numpy()
            image_id = image_id_tensor.numpy().decode("utf-8")

            if image_np.dtype in [np.float32, np.float64]:
                image_np = (image_np * 255.0).clip(0, 255).astype(np.uint8)
            if image_np.shape[-1] == 1:
                image_np = np.squeeze(image_np, axis=-1)

            filename = f"{purpose.value.lower()}_{image_id}.png"
            filepath = os.path.join(specific_output_directory, filename)
            tf.keras.utils.save_img(filepath, image_np)
            image_count += 1

        self.logger.info(f"Saved {image_count} images for {purpose} to {specific_output_directory}")

    def _build_dataset_split(
        self,
        df: pd.DataFrame,
        purpose: DatasetSplit,
        output_dir: str,
    ) -> tf.data.Dataset:
        self.logger.info(f"Creating {purpose} from {len(df)} samples.")
        dataset = self._create_tf_dataset(df, purpose)

        if self.config.output.artifact_level == ArtifactSavingLevel.FULL:
            self._save_images_to_disk(dataset, purpose, output_dir)

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

    def get_data_splits(
        self,
        seed: int,
        exp_id: str,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self.logger.info(f"Starting Dataset Preparation for Exp '{exp_id}'")

        raw_df = self._load_raw_dataframe()
        processed_df = self._process_dataframe(raw_df)
        sampled_df = self._sample_by_gender(processed_df, seed)
        train_df, val_df, test_df = self._split_dataframe(sampled_df, seed)

        output_dir = os.path.join(self.config.output.base_dir, exp_id)

        train_data = self._build_dataset_split(train_df, DatasetSplit.TRAIN, output_dir)
        val_data = self._build_dataset_split(val_df, DatasetSplit.VALIDATION, output_dir)
        test_data = self._build_dataset_split(test_df, DatasetSplit.TEST, output_dir)

        self.logger.info(f"Dataset Preparation Complete for Exp '{exp_id}'")
        return train_data, val_data, test_data
