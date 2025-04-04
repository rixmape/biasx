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
    """Manages facial image datasets and preprocessing for model input.

    Loads dataset information from configuration, handles fetching dataset files
    (e.g., Parquet files from HuggingFace Hub), allows sampling and shuffling,
    and provides an iterator to yield batches of processed image data ready
    for model consumption.

    Attributes:
        source (str): The identifier for the dataset source (e.g., 'utkface').
        image_width (int): The target width to resize images to.
        image_height (int): The target height to resize images to.
        color_mode (biasx.types.ColorMode): The target color mode ('L' for grayscale, 'RGB').
        single_channel (bool): Flag indicating if the output NumPy array should
            always have a single channel dimension (relevant for grayscale).
        max_samples (int): Maximum number of samples to load from the dataset.
            If <= 0, all samples are loaded.
        shuffle (bool): Whether to shuffle the dataset before sampling or iteration.
        seed (int): Random seed used for shuffling if `shuffle` is True.
        batch_size (int): The number of images to yield in each batch during iteration.
        dataset_info (biasx.types.ResourceMetadata): Metadata loaded from configuration
            about the dataset resource (repo ID, filename, column names, etc.).
        dataset_path (str): The local path to the downloaded dataset file.
        dataframe (pd.DataFrame): The pandas DataFrame holding the dataset metadata
            (after potential sampling and shuffling).
    """

    def __init__(self, source: str, image_width: int, image_height: int, color_mode: ColorMode, single_channel: bool, max_samples: int, shuffle: bool, seed: int, batch_size: int, **kwargs):
        """Initialize the dataset with configuration and preprocessing parameters.

        Loads dataset metadata, potentially samples and shuffles the data based
        on configuration, and sets up image processing parameters.

        Args:
            source (str): Identifier for the dataset source (e.g., "utkface").
                          Must correspond to an entry in `dataset_config.json`.
            image_width (int): Target width for images after resizing.
            image_height (int): Target height for images after resizing.
            color_mode (biasx.types.ColorMode): Target color mode for images
                                            (e.g., ColorMode.GRAYSCALE).
            single_channel (bool): If True and color_mode is GRAYSCALE, ensures
                                   the preprocessed numpy array has a channel dim.
            max_samples (int): The maximum number of samples to use from the dataset.
                               If 0 or negative, uses all samples.
            shuffle (bool): Whether to shuffle the dataset records.
            seed (int): The random seed to use for shuffling if `shuffle` is True.
            batch_size (int): The number of samples per batch for the iterator.
            **kwargs: Additional keyword arguments passed via configuration.
        """
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
        """Load dataset configuration, fetch data, and apply sampling/shuffling.

        Reads configuration from `dataset_config.json` based on `self.source`.
        Uses `get_resource_path` to download or find the cached dataset file
        (expected to be Parquet). Reads the Parquet file into a pandas DataFrame.
        Applies sampling (`max_samples`) and shuffling (`shuffle`, `seed`) to
        the DataFrame as configured.

        Raises:
            ValueError: If the `self.source` is not found in the configuration file.
            FileNotFoundError: If the dataset file cannot be found or downloaded.
            (Potentially other errors from `pq.read_table` or `hf_hub_download`).
        """
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
        """Preprocess a batch of PIL images into a single NumPy array.

        Converts images to the target `color_mode`, resizes them to
        (`image_width`, `image_height`), converts them to NumPy arrays
        (float32), normalizes pixel values to [0.0, 1.0], and stacks them
        into a single batch array. Ensures grayscale images have a channel
        dimension if `single_channel` is True.

        Args:
            pil_images (List[PIL.Image.Image]): A list of PIL Image objects to preprocess.

        Returns:
            A NumPy array representing the batch of preprocessed images, typically
            with shape (batch_size, height, width, channels). Returns an empty
            array with the correct dimensions if the input list is empty.
        """
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
        """Extract metadata (IDs, gender, age, race) for a batch DataFrame.

        Retrieves specific columns (defined in `self.dataset_info`) from the
        batch DataFrame and converts them into lists of the appropriate types
        (string IDs, Gender, Age, and Race enums).

        Args:
            batch_df (pd.DataFrame): The subset of the dataset DataFrame
                                     corresponding to the current batch.

        Returns:
            A tuple containing:
                - image_ids (List[str]): List of image identifiers.
                - genders (List[biasx.types.Gender]): List of gender labels.
                - ages (List[biasx.types.Age]): List of age labels.
                - races (List[biasx.types.Race]): List of race labels.
        """
        image_ids = [str(row[self.dataset_info.image_id_col]) for _, row in batch_df.iterrows()]
        genders = [Gender(int(row[self.dataset_info.gender_col])) for _, row in batch_df.iterrows()]
        ages = [Age(int(row[self.dataset_info.age_col])) for _, row in batch_df.iterrows()]
        races = [Race(int(row[self.dataset_info.race_col])) for _, row in batch_df.iterrows()]
        return image_ids, genders, ages, races

    def _load_batch_images(self, batch_df) -> List[Image.Image]:
        """Load a batch of PIL images from raw bytes in the DataFrame.

        Accesses the image column (specified by `self.dataset_info.image_col`),
        extracts the raw image bytes for each row in the batch DataFrame,
        and loads them into PIL Image objects.

        Args:
            batch_df (pd.DataFrame): The subset of the dataset DataFrame
                                     corresponding to the current batch.

        Returns:
            A list of PIL.Image.Image objects loaded from the batch data.
        """
        return [Image.open(BytesIO(row[self.dataset_info.image_col]["bytes"])) for _, row in batch_df.iterrows()]

    def __len__(self) -> int:
        """Return the number of images in the configured dataset.

        Returns the total number of samples in the DataFrame after loading,
        sampling, and shuffling have been applied.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.dataframe)

    def __iter__(self) -> Iterator[List[ImageData]]:
        """Iterate through the dataset, yielding batches of ImageData objects.

        Iterates over the internal DataFrame in steps of `self.batch_size`.
        For each batch, it loads the PIL images, preprocesses them into a
        NumPy array, extracts metadata, and then constructs a list of
        `ImageData` objects, where each object contains the ID, PIL image,
        preprocessed NumPy array slice, dimensions, and demographic labels
        for one sample.

        Yields:
            A list of `biasx.types.ImageData` objects representing one batch of data.
                The list length will be equal to `self.batch_size`, except possibly
                for the last batch.
        """
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
