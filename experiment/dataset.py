import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

# isort: off
from config import Config
from masker import FeatureMasker
from datatypes import DatasetSplits, Gender
from utils import setup_logger

logger = setup_logger(name="experiment.dataset")


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
