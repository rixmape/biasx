"""Handles loading and processing image datasets, and managing analysis results."""

from io import BytesIO
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .config import configurable
from .types import Age, ColorMode, DatasetMetadata, DatasetSource, Gender, ImageData, Race
from .utils import download_resource, load_json_config


class ImageProcessor:
    """Builder class for image preprocessing with method chaining."""

    def __init__(self, image: Image.Image):
        """Initialize with a PIL image."""
        self.image = image
        self.array = None

    def convert(self, mode: str) -> "ImageProcessor":
        """Convert image to specified color mode."""
        if self.image.mode != mode:
            self.image = self.image.convert(mode)
        return self

    def resize(self, width: int, height: int) -> "ImageProcessor":
        """Resize image to specified dimensions."""
        if (self.image.width, self.image.height) != (width, height):
            self.image = self.image.resize((width, height))
        return self

    def to_array(self, dtype=np.float32, normalize: bool = True) -> "ImageProcessor":
        """Convert image to numpy array with optional normalization."""
        self.array = np.array(self.image, dtype=dtype)
        if normalize:
            self.array = self.array / 255.0
        return self

    def expand_dims(self, axis: int = -1) -> "ImageProcessor":
        """Add a dimension to the array."""
        if self.array is not None:
            self.array = np.expand_dims(self.array, axis=axis)
        return self

    def get_array(self) -> np.ndarray:
        """Get the processed numpy array."""
        return self.array

    def get_image(self) -> Image.Image:
        """Get the processed PIL image."""
        return self.image


@configurable("dataset")
class Dataset:
    """Manages facial image datasets and preprocessing for model input."""

    def __init__(
        self,
        source: DatasetSource,
        image_width: int,
        image_height: int,
        color_mode: ColorMode,
        single_channel: bool,
        max_samples: int,
        shuffle: bool,
        seed: int,
        **kwargs,
    ):
        """Initialize the dataset with preprocessing parameters."""
        self.source = source
        self.image_width = image_width
        self.image_height = image_height
        self.color_mode = color_mode
        self.single_channel = single_channel
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed

        self.dataset_info = self._load_dataset_metadata()
        self.dataset_path = self._download_dataset()
        self.dataframe = self._load_dataframe()

    def _load_dataset_metadata(self) -> DatasetMetadata:
        """Load dataset metadata from configuration file."""
        config = load_json_config(__file__, "dataset_config.json")

        if self.source.value not in config:
            raise ValueError(f"Dataset source {self.source.value} not found in configuration")

        return DatasetMetadata(**config[self.source.value])

    def _download_dataset(self) -> str:
        """Download the dataset from HuggingFace."""
        return download_resource(
            repo_id=self.dataset_info.repo_id,
            filename=self.dataset_info.filename,
            repo_type=self.dataset_info.repo_type,
        )

    def _load_dataframe(self):
        """Download and load the dataset as a pandas DataFrame."""
        table = pq.read_table(self.dataset_path)
        df = table.to_pandas()

        if self.max_samples > 0:
            df = df.sample(min(self.max_samples, len(df)), random_state=self.seed) if self.shuffle else df.head(self.max_samples)
        elif self.shuffle:
            df = df.sample(frac=1, random_state=self.seed)

        return df

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess an image for model input using method chaining."""
        processor = ImageProcessor(image)

        processor.convert(self.color_mode.value).resize(self.image_width, self.image_height).to_array()

        if self.color_mode == ColorMode.GRAYSCALE and not self.single_channel:
            processor.expand_dims(axis=-1)

        return processor.get_array()

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataframe)

    def __iter__(self) -> Iterator[ImageData]:
        """Iterate through dataset, yielding preprocessed ImageData objects."""
        for _, row in self.dataframe.iterrows():
            img_bytes = row[self.dataset_info.image_col]["bytes"]
            pil_image = Image.open(BytesIO(img_bytes))

            processed_image = self._preprocess_image(pil_image)

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

            yield image_data
