import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# isort: off
from config import Config
from masker import FeatureMasker
from datatypes import Gender
from utils import setup_logger


class DatasetGenerator:
    """Class that loads, processes, samples, and splits the dataset, then creates TensorFlow datasets for model training and evaluation."""

    def __init__(self, config: Config, feature_masker: FeatureMasker, log_path: str):
        self.config = config
        self.logger = setup_logger(name="dataset_generator", log_path=log_path)
        self.feature_masker = feature_masker

    def _load_dataset(self) -> pd.DataFrame:
        """Downloads and loads the dataset from Hugging Face Hub, filters by valid age, and extracts image bytes and labels."""
        self.logger.info(f"Downloading {self.config.dataset_name} dataset")
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset_name}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        self.logger.debug(f"Dataset successfully downloaded")

        # TODO: Include image ID column to mention in log messages during image processing
        df = pd.read_parquet(path, columns=["image", "gender", "race", "age"])
        self.logger.debug(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        df["image_bytes"] = df["image"].apply(lambda x: x["bytes"])
        df = df.drop(columns=["image"])

        initial_count = len(df)
        df = df[df["age"] > 0]
        filtered_count = len(df)
        self.logger.debug(f"Removed 0-9 years old samples: {initial_count - filtered_count} ({(initial_count - filtered_count) / initial_count:.2%})")

        male_count = (df["gender"] == Gender.MALE.value).sum()
        female_count = (df["gender"] == Gender.FEMALE.value).sum()
        self.logger.debug(f"Gender distribution: male={male_count} ({male_count/len(df):.2%}), female={female_count} ({female_count/len(df):.2%})")

        self.logger.info(f"Processed dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        return df

    def _sample_by_strata(self, df: pd.DataFrame, sample_size: int, seed: int) -> list[pd.DataFrame]:
        """Stratifies the dataset by race and age, then samples each group proportionally based on the desired sample size."""
        samples = []
        strata_sampled = 0
        strata_skipped = 0

        for strata_name, group in df.groupby("strata"):
            grp_size = len(group)
            grp_sample_size = round(sample_size * (grp_size / len(df)))

            if grp_sample_size == 0:
                self.logger.warning(f"Strata '{strata_name}': Skipped (size {grp_size} too small for proportional sampling)")
                strata_skipped += 1
                continue

            replacement_needed = grp_size < grp_sample_size

            if replacement_needed:
                self.logger.debug(f"Strata '{strata_name}': Using replacement sampling ({grp_size} available, {grp_sample_size} needed)")

            sample = group.sample(n=grp_sample_size, random_state=seed, replace=replacement_needed)
            samples.append(sample)
            strata_sampled += 1

        return samples

    def _sample_by_gender(self, male_ratio: float, seed: int, df: pd.DataFrame) -> pd.DataFrame:
        """Samples the dataset separately by gender using the specified male ratio and stratified sampling."""
        ratios = {Gender.MALE: male_ratio, Gender.FEMALE: 1.0 - male_ratio}

        df["strata"] = df["race"].astype(str) + "_" + df["age"].astype(str)
        strata_counts = df.groupby("strata").size()
        self.logger.info(f"Found {len(strata_counts)} unique race-age groups for stratified sampling")

        samples = []
        for gender, target_ratio in ratios.items():
            gender_sample_size = round(self.config.dataset_size * target_ratio)
            gender_df = df[df["gender"] == gender.value]

            self.logger.info(f"Targetting {gender_sample_size} from {len(gender_df)} available {gender.name.lower()} images ({target_ratio:.2%} of total)")

            if len(gender_df) < gender_sample_size:
                self.logger.warning(f"Available {gender.name.lower()} samples ({len(gender_df)}) is less than target {gender_sample_size} samples")

            strata_samples = self._sample_by_strata(gender_df, gender_sample_size, seed)
            gender_sample = pd.concat(strata_samples)
            actual_ratio = (gender_sample["gender"] / gender.value).mean()
            samples.append(gender_sample)

            if actual_ratio < target_ratio:
                self.logger.warning(f"Actual {gender.name.lower()} ratio ({actual_ratio}) is less than target ratio ({target_ratio})")

        combined = pd.concat(samples).sample(frac=1, random_state=seed).reset_index(drop=True)
        combined = combined.drop(columns=["strata"])

        self.logger.info(f"Sampled {len(combined)} total images ({len(combined) / self.config.dataset_size:.2%} of target size)")

        if len(combined) < self.config.dataset_size:
            self.logger.warning(f"Final dataset size ({len(combined)}) is less than target size ({self.config.dataset_size})")

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
        self.logger.info(f"Creating {purpose} TensorFlow dataset from {len(df)} samples")

        male_count = (df["gender"] == Gender.MALE.value).sum()
        female_count = (df["gender"] == Gender.FEMALE.value).sum()
        self.logger.debug(f"{purpose} dataset gender distribution: Male={male_count} ({male_count/len(df):.2%}), Female={female_count} ({female_count/len(df):.2%})")

        if mask_gender is not None and mask_feature is not None:
            gender_name = Gender(mask_gender).name
            masked_count = (df["gender"] == mask_gender).sum()
            self.logger.debug(f"{purpose} dataset masking: Feature '{mask_feature}' will be masked for {gender_name} gender ({masked_count} images, {masked_count/len(df):.2%} of this split)")
        else:
            self.logger.debug(f"{purpose} dataset: No feature masking applied")

        dataset = tf.data.Dataset.from_tensor_slices((df["image_bytes"].values, df["gender"].values))
        dataset = dataset.map(lambda x, y: self._decode_and_process_image(x, y, mask_gender, mask_feature), num_parallel_calls=tf.data.AUTOTUNE)

        self.logger.debug(f"{purpose} dataset creation complete")
        return dataset.prefetch(tf.data.AUTOTUNE)

    def split_dataset(self, seed: int, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the dataset into training, validation, and test sets using stratified sampling based on gender."""
        self.logger.info("Splitting dataset into train, validation, and test sets")

        target_train_ratio = 1 - self.config.val_split - self.config.test_split
        target_val_ratio = self.config.val_split
        target_test_ratio = self.config.test_split
        self.logger.debug(f"Target splits: train={target_train_ratio:.2%}, validation={target_val_ratio:.2%}, test={target_test_ratio:.2%}")

        train_val, test = train_test_split(df, test_size=target_test_ratio, random_state=seed, stratify=df["gender"])
        effective_val_split = target_val_ratio / (1 - target_test_ratio)
        train, val = train_test_split(train_val, test_size=effective_val_split, random_state=seed, stratify=train_val["gender"])

        self.logger.debug(f"Actual splits: train={len(train) / len(df):.2%}, validation={len(val) / len(df):.2%}, test={len(test) / len(df):.2%}")

        return train, val, test

    def prepare_data(self, male_ratio: float, mask_gender: int, mask_feature: str, seed: int) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Loads, samples by gender, splits the dataset, and returns batched and cached TensorFlow datasets for training, validation, and testing."""
        self.logger.info(f"Preparing dataset: male_ratio={male_ratio}, mask_gender={mask_gender}, mask_feature={mask_feature}, seed={seed}")

        df = self._load_dataset()
        df = self._sample_by_gender(male_ratio, seed, df)

        train_df, val_df, test_df = self.split_dataset(seed, df)

        batch_size = self.config.batch_size

        train_data = self._create_dataset(train_df, mask_gender, mask_feature, "TRAINING").batch(batch_size).cache()
        self.logger.debug(f"Training dataset cached with {len(train_df)} samples ({len(train_df) // batch_size + 1} batches)")

        val_data = self._create_dataset(val_df, mask_gender, mask_feature, "VALIDATION").batch(batch_size).cache()
        self.logger.debug(f"Validation dataset cached with {len(val_df)} samples ({len(val_df) // batch_size + 1} batches)")

        test_data = self._create_dataset(test_df, mask_gender, mask_feature, "TEST").batch(batch_size)
        self.logger.debug(f"Test dataset created with {len(test_df)} samples ({len(test_df) // batch_size + 1} batches)")

        self.logger.info("Dataset preparation complete")

        return train_data, val_data, test_data
