"""Handles loading and processing image datasets, and managing analysis results."""

from io import BytesIO
from typing import Iterator, List

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

    def _convert_image(self, image: Image.Image, mode: str) -> Image.Image:
        """Convert image to specified color mode."""
        return image.convert(mode) if image.mode != mode else image

    def _resize_image(self, image: Image.Image, width: int, height: int) -> Image.Image:
        """Resize image to specified dimensions."""
        return image.resize((width, height)) if (image.width, image.height) != (width, height) else image

    def _to_array(self, image: Image.Image, normalize: bool = True) -> np.ndarray:
        """Convert image to numpy array with optional normalization."""
        array = np.array(image, dtype=np.float32)
        return array / 255.0 if normalize else array

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess an image for model input."""
        image = self._convert_image(image, self.color_mode.value)
        image = self._resize_image(image, self.image_width, self.image_height)
        array = self._to_array(image)

        if self.color_mode == ColorMode.GRAYSCALE and not self.single_channel:
            array = np.expand_dims(array, axis=-1)

        return array

    def _preprocess_batch(self, pil_images: List[Image.Image]) -> List[np.ndarray]:
        """Preprocess a batch of images in parallel."""
        return [self._preprocess_image(img) for img in pil_images]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataframe)

    def __iter__(self) -> Iterator[List[ImageData]]:
        """Iterate through dataset, yielding batches of preprocessed ImageData objects."""
        batch_indices = range(0, len(self.dataframe), self.batch_size)

        for i in batch_indices:
            batch_df = self.dataframe.iloc[i : i + self.batch_size]
            pil_images = [Image.open(BytesIO(row[self.dataset_info.image_col]["bytes"])) for _, row in batch_df.iterrows()]
            processed_images = self._preprocess_batch(pil_images)

            batch_data = []
            for (_, row), pil_image, processed_image in zip(batch_df.iterrows(), pil_images, processed_images):
                image_data = ImageData(
                    image_id=str(row[self.dataset_info.image_id_col]),
                    pil_image=pil_image,
                    preprocessed_image=processed_image,
                    width=self.image_width,
                    height=self.image_height,
                    gender=Gender(int(row[self.dataset_info.gender_col])),
                    age=Age(int(row[self.dataset_info.age_col])),
                    race=Race(int(row[self.dataset_info.race_col])),
                )
                batch_data.append(image_data)

            yield batch_data
