"""Dataset manipulator for creating controlled gender-biased datasets."""

import json
import os
from io import BytesIO
from typing import Dict, List, Tuple

import mediapipe as mp
import numpy as np
import pandas as pd
from config import DatasetConfig
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image


class FacialFeatureMasker:
    """Process facial images by detecting landmarks and applying zero masking."""

    def __init__(self):
        """Initialize facial landmark detector."""
        self.landmarker = self._load_landmarker()
        self.landmark_map = self._load_landmark_map()

    def _load_landmarker(self) -> FaceLandmarker:
        """Load MediaPipe facial landmark detector."""
        model_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="mediapipe_landmarker.task", repo_type="model")
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path))
        return FaceLandmarker.create_from_options(options)

    def _load_landmark_map(self) -> Dict[str, List[int]]:
        """Load landmark mapping from JSON file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        landmark_path = os.path.join(script_dir, "landmark_map.json")
        with open(landmark_path, "r") as file:
            return json.load(file)

    def _detect_landmarks(self, image: Image.Image) -> List:
        """Detect facial landmarks using MediaPipe."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        result = self.landmarker.detect(mp_image)
        return result.face_landmarks[0] if result.face_landmarks else None

    def _normalize_landmarks(self, landmarks, image_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Convert normalized landmark coordinates to pixel values."""
        width, height = image_size
        return [(int(point.x * width), int(point.y * height)) for point in landmarks]

    def _get_feature_box(self, landmarks: List[Tuple[int, int]], feature: str, padding: int) -> Tuple[int, int, int, int]:
        """Calculate the bounding box for a facial feature."""
        feature_points = [landmarks[i] for i in self.landmark_map[feature]]
        min_x = min(x for x, _ in feature_points) - padding
        min_y = min(y for _, y in feature_points) - padding
        max_x = max(x for x, _ in feature_points) + padding
        max_y = max(y for _, y in feature_points) + padding
        return (int(min_x), int(min_y), int(max_x), int(max_y))

    def apply_zero_mask(self, image: Image.Image, feature: str, padding: int) -> Image.Image:
        """Apply zero masking to a specific facial feature."""
        landmarks = self._detect_landmarks(image)
        if not landmarks:
            return image
        pixel_landmarks = self._normalize_landmarks(landmarks, image.size)
        result = image.copy()
        box = self._get_feature_box(pixel_landmarks, feature, padding)
        region = result.crop(box)
        zero_mask = Image.new(region.mode, region.size, color=0)
        result.paste(zero_mask, box)
        return result


class DatasetGenerator:
    """Generates datasets with controlled demographic distributions."""

    ATTRS: list[str] = ["gender", "race", "age"]

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize generator with configuration settings."""
        self.config = config

    def _load_dataset(self) -> pd.DataFrame:
        """Retrieve and load dataset from Hugging Face repository."""
        path = hf_hub_download(repo_id=f"rixmape/{self.config.dataset_name}", filename="data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pd.read_parquet(path)
        df["image"] = df["image"].apply(lambda x: Image.open(BytesIO(x["bytes"])))
        return df

    def _sample_by_stratum(self, data: pd.DataFrame, stratum_col: str, total_sample_size: int) -> list:
        """Sample data proportionally within each stratum."""
        strat_samples = []
        for _, group in data.groupby(stratum_col):
            group_size = len(group)
            group_ratio = group_size / len(data)
            stratum_sample_size = round(total_sample_size * group_ratio)
            if stratum_sample_size > 0:
                strat_samples.append(group.sample(n=stratum_sample_size, random_state=self.config.random_seed, replace=(group_size < stratum_sample_size)))
        return strat_samples

    def _sample_with_gender_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample dataset with target gender ratio while balancing race and age."""
        total_size = self.config.dataset_size
        samples = []
        for gender, ratio in self.config.gender_ratios.items():
            gender_sample_size = round(total_size * ratio)
            gender_df = df[df["gender"] == gender].copy()
            if gender_df.empty:
                continue
            gender_df["strata"] = gender_df["race"].astype(str) + "_" + gender_df["age"].astype(str)
            strata_samples = self._sample_by_stratum(gender_df, "strata", gender_sample_size)
            if strata_samples:
                samples.append(pd.concat(strata_samples))
        return pd.concat(samples).sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)

    def _apply_feature_masking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply zero masking to a specific facial feature."""
        if self.config.masked_gender is None or self.config.masked_feature is None:
            return df
        processor = FacialFeatureMasker()
        result = df.copy()
        gender_mask = result["gender"] == self.config.masked_gender
        if gender_mask.any():
            result.loc[gender_mask, "image"] = result.loc[gender_mask, "image"].apply(lambda img: processor.apply_zero_mask(img, self.config.masked_feature, self.config.padding))
        return result

    def create_dataset(self) -> pd.DataFrame:
        """Create a stratified dataset based on gender, race, and age."""
        df = self._load_dataset()
        df = self._sample_with_gender_ratio(df)
        return self._apply_feature_masking(df)
