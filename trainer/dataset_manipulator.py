"""Dataset manipulator for creating controlled demographic datasets with feature transformation."""

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import mediapipe as mp
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image, ImageFilter


@dataclass
class DatasetConfig:
    """Configuration settings for demographic dataset creation and transformation."""

    dataset_name: str = "utkface"
    dataset_size: int = 1000
    random_seed: Optional[int] = None
    gender_ratios: Optional[dict[int, float]] = field(default_factory=dict)
    race_ratios: Optional[dict[int, float]] = field(default_factory=dict)
    age_ratios: Optional[dict[int, float]] = field(default_factory=dict)
    target_feature: Optional[str] = None
    target_gender: Optional[int] = None
    target_race: Optional[int] = None
    target_age: Optional[int] = None
    transformation: Optional[str] = None
    blur_value: Optional[float] = 2.0
    mask_value: Optional[float] = 0.0
    padding: Optional[int] = 2

    @property
    def should_apply_transformation(self) -> bool:
        """Return True if both a demographic target and an image transformation are configured."""
        has_target = any(target is not None for target in (self.target_gender, self.target_race, self.target_age))
        has_transformation = self.target_feature is not None and self.transformation is not None
        return has_target and has_transformation


class ImageProcessor:
    """Process facial images by detecting landmarks and applying regional transformations."""

    def __init__(self) -> None:
        """Initialize the facial landmark detector."""
        self.landmarker = self._load_landmarker()
        self.landmark_map = self._load_landmark_map()

    def _load_landmarker(self) -> FaceLandmarker:
        """Load the facial landmark detector from a model file."""
        model_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="mediapipe_landmarker.task", repo_type="model")
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path))
        return FaceLandmarker.create_from_options(options)

    def _load_landmark_map(self) -> dict[str, list[int]]:
        """Define the mapping between facial features and landmark indices."""
        landmark_path = Path(__file__).parent / "landmark_map.json"
        with open(landmark_path, "r") as file:
            return json.load(file)

    def _convert_to_pixel_coordinates(self, landmarks: Any, image_size: tuple[int, int]) -> list[tuple[int, int]]:
        """Convert normalized landmark coordinates to pixel values."""
        width, height = image_size
        return [(int(point.x * width), int(point.y * height)) for point in landmarks]

    def _detect_landmarks(self, image: Image.Image) -> list[tuple[int, int]]:
        """Detect facial landmarks in an image and return pixel coordinates."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return []
        return self._convert_to_pixel_coordinates(result.face_landmarks[0], image.size)

    def _get_feature_bounding_box(self, landmarks: list[tuple[int, int]], feature: str, padding: int) -> tuple[int, int, int, int]:
        """Calculate the bounding box for a facial feature."""
        if not feature or feature not in self.landmark_map:
            raise ValueError(f"Invalid feature: {feature}")
        feature_points = np.array([landmarks[i] for i in self.landmark_map[feature]])
        min_x, min_y = np.min(feature_points, axis=0) - padding
        max_x, max_y = np.max(feature_points, axis=0) + padding
        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def _transform_region(self, image: Image.Image, box: tuple[int, int, int, int], config: DatasetConfig) -> Image.Image:
        """Transform a specific region of an image based on configuration settings."""
        result = image.copy()
        region = result.crop(box)
        if config.transformation == "blur":
            transformed_region = region.filter(ImageFilter.GaussianBlur(radius=config.blur_value))
        elif config.transformation == "mask":
            transformed_region = Image.new(region.mode, region.size, color=int(config.mask_value * 255))
        else:
            transformed_region = region
        result.paste(transformed_region, box)
        return result

    def process_image(self, image: Image.Image, config: DatasetConfig) -> Image.Image:
        """Apply feature transformation to an image based on configuration settings."""
        landmarks = self._detect_landmarks(image)
        if not landmarks:
            return image
        feature_box = self._get_feature_bounding_box(landmarks, config.target_feature, config.padding)
        return self._transform_region(image, feature_box, config)


class ControlledDataset:
    """Create and manipulate demographic datasets with controlled distribution."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize with dataset configuration settings."""
        self.config = config

    def _load_dataset(self) -> pd.DataFrame:
        """Load the source dataset from Hugging Face Hub."""
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset_name}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        df["image"] = df["image"].apply(lambda x: Image.open(BytesIO(x["bytes"])))
        return df

    def _ensure_demographic_ratios(self, unique_values: dict[str, list[int]]) -> list[dict[int, float], dict[int, float], dict[int, float]]:
        """Return demographic ratios with uniform distribution as fallback."""
        get_uniform = lambda vals: {val: 1.0 / len(vals) for val in vals}
        ratios = []
        for attr in ["gender", "race", "age"]:
            config_ratios = getattr(self.config, f"{attr}_ratios", {})
            ratios.append(config_ratios or get_uniform(unique_values[attr]))
        return ratios

    def _generate_demographic_combinations(self, unique_values: dict[str, list[int]]) -> list[tuple[int, int, int]]:
        """Generate all possible demographic combinations from the dataset."""
        combinations = []
        for gender in unique_values["gender"]:
            for race in unique_values["race"]:
                for age in unique_values["age"]:
                    combinations.append((gender, race, age))
        return combinations

    def _calculate_raw_quotas(self, combinations: list[tuple[int, int, int]], demographic_ratios: list[dict[int, float], dict[int, float], dict[int, float]]) -> tuple[dict[tuple[int, int, int], int], int]:
        """Calculate initial quotas for demographic combinations."""
        gender_ratios, race_ratios, age_ratios = demographic_ratios
        quotas = {}
        total_quota = 0
        for gender, race, age in combinations:
            quota = round(gender_ratios[gender] * race_ratios[race] * age_ratios[age] * self.config.dataset_size)
            quotas[(gender, race, age)] = quota
            total_quota += quota
        return quotas, total_quota

    def _adjust_quotas(self, quotas: dict[tuple[int, int, int], int], total_quota: int) -> dict[tuple[int, int, int], int]:
        """Adjust quotas to match the exact target size."""
        if total_quota != self.config.dataset_size:
            max_key = max(quotas, key=quotas.get)
            quotas[max_key] += self.config.dataset_size - total_quota
        return quotas

    def _compute_quotas(self, df: pd.DataFrame) -> dict[tuple[int, int, int], int]:
        """Calculate sample quotas for each demographic combination."""
        unique_values = {col: sorted(df[col].unique()) for col in ["gender", "race", "age"]}
        demographic_ratios = self._ensure_demographic_ratios(unique_values)
        combinations = self._generate_demographic_combinations(unique_values)
        quotas, total_quota = self._calculate_raw_quotas(combinations, demographic_ratios)
        return self._adjust_quotas(quotas, total_quota)

    def _sample_by_quotas(self, df: pd.DataFrame, quotas: dict[tuple[int, int, int], int]) -> pd.DataFrame:
        """Sample data according to demographic quotas."""
        samples = []
        for (gender, race, age), quota in quotas.items():
            if quota <= 0:
                continue
            mask = (df["gender"] == gender) & (df["race"] == race) & (df["age"] == age)
            group = df[mask]
            if group.empty:
                continue
            sample = group.sample(n=quota, random_state=self.config.random_seed, replace=len(group) < quota)
            samples.append(sample)
        if not samples:
            raise ValueError("No samples could be drawn from any demographic group")
        return pd.concat(samples, ignore_index=True)

    def _create_demographic_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a subset with controlled demographic distribution."""
        quotas = self._compute_quotas(df)
        sampled_df = self._sample_by_quotas(df, quotas)
        return sampled_df.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)

    def _create_target_mask(self, df: pd.DataFrame) -> pd.Series:
        """Create a mask for rows matching the target demographic criteria."""
        target_mask = pd.Series(True, index=df.index)
        for attr in ["gender", "race", "age"]:
            target_value = getattr(self.config, f"target_{attr}", None)
            if target_value is not None:
                target_mask &= df[attr] == target_value
        return target_mask

    def _apply_feature_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformation to target demographic rows."""
        result = df.copy()
        target_mask = self._create_target_mask(df)
        if target_mask.any():
            processor = ImageProcessor()
            result.loc[target_mask, "image"] = result.loc[target_mask, "image"].apply(processor.process_image, config=self.config)
        return result

    def create_dataset(self) -> pd.DataFrame:
        """Create a dataset with controlled demographic distribution and optional feature transformation."""
        df = self._load_dataset()
        df = self._create_demographic_subset(df)
        if self.config.should_apply_transformation:
            df = self._apply_feature_transformation(df)
        return df
