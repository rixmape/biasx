"""Handles loading and processing image datasets, and managing analysis results."""

from io import BytesIO
from typing import Iterator, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .config import configurable
from .types import Age, ColorMode, Gender, ImageData, Race, ResourceMetadata
from .utils import get_json_config, get_resource_path


@configurable("dataset")
class Dataset:
    """Manages facial image datasets and preprocessing for model input."""

    def __init__(self, source: str, image_width: int, image_height: int, color_mode: ColorMode, single_channel: bool, max_samples: int, shuffle: bool, seed: int, batch_size: int, **kwargs):
        """Initialize the dataset with preprocessing parameters."""
        self.source = source
        self.image_width = image_width
        self.image_height = image_height
        self.color_mode = color_mode
        self.single_channel = single_channel
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size

        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset configuration and data."""
        config = get_json_config(__file__, "dataset_config.json")

        if self.source.value not in config:
            raise ValueError(f"Dataset source {self.source.value} not found in configuration")

        metadata_dict = config[self.source.value]
        self.dataset_info = ResourceMetadata(**metadata_dict)
        self.dataset_path = get_resource_path(repo_id=self.dataset_info.repo_id, filename=self.dataset_info.filename, repo_type=self.dataset_info.repo_type)

        table = pq.read_table(self.dataset_path)
        self.dataframe = table.to_pandas()

        if self.max_samples > 0:
            if self.shuffle:
                self.dataframe = self.dataframe.sample(min(self.max_samples, len(self.dataframe)), random_state=self.seed)
            else:
                self.dataframe = self.dataframe.head(self.max_samples)
        elif self.shuffle:
            self.dataframe = self.dataframe.sample(frac=1, random_state=self.seed)

    def _preprocess_batch(self, pil_images: List[Image.Image]) -> np.ndarray:
        """Preprocess a batch of images into a single NumPy array."""
        if not pil_images:
            return np.empty((0, self.image_height, self.image_width, 1 if self.color_mode == ColorMode.GRAYSCALE else 3))

        processed_images = []
        for img in pil_images:
            if img.mode != self.color_mode.value:
                img = img.convert(self.color_mode.value)
            if (img.width, img.height) != (self.image_width, self.image_height):
                img = img.resize((self.image_width, self.image_height))
            processed_images.append(np.array(img, dtype=np.float32))

        batch_array = np.stack(processed_images)
        batch_array = batch_array / 255.0

        if self.color_mode == ColorMode.GRAYSCALE and (len(batch_array.shape) < 4 or batch_array.shape[3] != 1):
            batch_array = np.expand_dims(batch_array, axis=-1)

        return batch_array

    def _extract_batch_metadata(self, batch_df) -> Tuple[List[str], List[Gender], List[Age], List[Race]]:
        """Extract metadata for a batch of images."""
        image_ids = [str(row[self.dataset_info.image_id_col]) for _, row in batch_df.iterrows()]
        genders = [Gender(int(row[self.dataset_info.gender_col])) for _, row in batch_df.iterrows()]
        ages = [Age(int(row[self.dataset_info.age_col])) for _, row in batch_df.iterrows()]
        races = [Race(int(row[self.dataset_info.race_col])) for _, row in batch_df.iterrows()]
        return image_ids, genders, ages, races

    def _load_batch_images(self, batch_df) -> List[Image.Image]:
        """Load a batch of PIL images from dataframe."""
        return [Image.open(BytesIO(row[self.dataset_info.image_col]["bytes"])) for _, row in batch_df.iterrows()]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataframe)

    def __iter__(self) -> Iterator[List[ImageData]]:
        """Iterate through dataset, yielding batches of preprocessed ImageData objects."""
        for i in range(0, len(self.dataframe), self.batch_size):
            batch_df = self.dataframe.iloc[i : i + self.batch_size]

            pil_images = self._load_batch_images(batch_df)
            image_ids, genders, ages, races = self._extract_batch_metadata(batch_df)
            processed_batch = self._preprocess_batch(pil_images)

            batch_data = []
            for idx, (image_id, pil_image, gender, age, race) in enumerate(zip(image_ids, pil_images, genders, ages, races)):
                image_data = ImageData(
                    image_id=image_id,
                    pil_image=pil_image,
                    preprocessed_image=processed_batch[idx],
                    width=self.image_width,
                    height=self.image_height,
                    gender=gender,
                    age=age,
                    race=race,
                )
                batch_data.append(image_data)

            yield batch_data
