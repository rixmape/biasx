"""
Dataset management module for BiasX.
Handles loading and processing image datasets, and managing analysis results.
"""

import json
import pathlib
from io import BytesIO
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from .types import Age, ColorMode, DatasetMetadata, DatasetSource, Gender, ImageData, Race


class FaceDataset:
    """Manages facial image datasets and preprocessing for model input."""

    def __init__(
        self,
        source: DatasetSource = DatasetSource.UTKFACE,
        image_width: int = 224,
        image_height: int = 224,
        color_mode: ColorMode = ColorMode.GRAYSCALE,
        single_channel: bool = False,
        max_samples: int = 100,
        shuffle: bool = True,
        seed: int = 69,
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

        self.dataset_info = self._load_dataset_metadata(source)
        self.dataset_path = self._download_dataset()
        self.dataframe = self._load_dataframe()

    def _load_dataset_metadata(self, source: DatasetSource) -> DatasetMetadata:
        """Load dataset metadata from configuration file."""
        config_path = self._get_config_path()

        with open(config_path, "r") as f:
            config = json.load(f)

        if source.value not in config:
            raise ValueError(f"Dataset source {source.value} not found in configuration")

        return DatasetMetadata(**config[source.value])

    def _get_config_path(self) -> str:
        """Get the path to the dataset configuration file."""
        module_dir = pathlib.Path(__file__).parent
        config_path = module_dir / "data" / "dataset_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Dataset configuration file not found at {config_path}")

        return str(config_path)

    def _download_dataset(self) -> str:
        """Download the dataset from HuggingFace."""
        return hf_hub_download(
            repo_id=self.dataset_info.repo_id,
            filename=self.dataset_info.filename,
            repo_type=self.dataset_info.repo_type,
        )

    def _load_dataframe(self):
        """Download and load the dataset as a pandas DataFrame."""
        table = pq.read_table(self.dataset_path)
        df = table.to_pandas()

        if self.max_samples > 0:
            df = (
                df.sample(min(self.max_samples, len(df)), random_state=self.seed)
                if self.shuffle
                else df.head(self.max_samples)
            )
        elif self.shuffle:
            df = df.sample(frac=1, random_state=self.seed)

        return df

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess an image for model input."""
        if image.mode != self.color_mode.value:
            image = image.convert(self.color_mode.value)

        if (image.width, image.height) != (self.image_width, self.image_height):
            image = image.resize((self.image_width, self.image_height))

        img_array = np.array(image, dtype=np.float32) / 255.0

        if self.color_mode == ColorMode.GRAYSCALE and not self.single_channel:
            img_array = np.expand_dims(img_array, axis=-1)

        return img_array

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
