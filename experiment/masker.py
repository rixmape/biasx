import json
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions

# isort: off
from config import Config
from datatypes import FacialFeature
from utils import setup_logger


class FeatureMasker:
    """Class responsible for detecting facial landmarks and applying masks to specific facial features in images."""

    def __init__(self, config: Config, log_path: str):
        self.config = config
        self.logger = setup_logger(name="feature_masker", log_path=log_path)

        self.landmarker = self.load_landmarker()
        self.feature_map = self.load_feature_map()

    @staticmethod
    def load_landmarker() -> FaceLandmarker:
        """Loads and returns a pre-trained face landmarker model from the Hugging Face hub using specified options."""
        model_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="mediapipe_landmarker.task", repo_type="model")
        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path))
        return FaceLandmarker.create_from_options(options)

    @staticmethod
    def load_feature_map() -> dict[str, list[int]]:
        """Retrieves and parses a JSON file mapping facial features to landmark indices from HuggingFace."""
        path = hf_hub_download(repo_id="rixmape/biasx-models", filename="landmark_map.json", repo_type="model")
        return json.load(open(path, "r"))

    def _detect_landmarks(self, image: np.ndarray) -> Any:
        """Converts the input image to RGB (if necessary) and detects facial landmarks using the loaded landmarker."""
        if image.shape[-1] == 1 or len(image.shape) == 2:
            rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            rgb = image

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        return result.face_landmarks[0] if result.face_landmarks else None

    def _to_pixel_coords(self, landmarks: list, img_shape: tuple[int, int]) -> list[tuple[int, int]]:
        """Converts normalized landmark coordinates to absolute pixel coordinates based on the image dimensions."""
        height, width = img_shape[:2]
        return [(int(pt.x * width), int(pt.y * height)) for pt in landmarks]

    def _get_landmarks_in_pixels(self, image: np.ndarray) -> Optional[list[tuple[int, int]]]:
        """Obtains facial landmarks in pixel coordinates by detecting landmarks and converting them using `_to_pixel_coords`."""
        landmarks = self._detect_landmarks(image)
        if landmarks is None:
            return None
        return self._to_pixel_coords(landmarks, image.shape)

    def _get_bbox(self, pix_coords: list[tuple[int, int]], feature: str) -> tuple[int, int, int, int]:
        """Computes the bounding box for a specified facial feature from pixel coordinates"""
        pad = max(1, self.config.mask_padding)  # Ensure padding is at least 1 to avoid zero area
        pts = [pix_coords[i] for i in self.feature_map[feature]]
        min_x = max(0, min(x for x, _ in pts) - pad)
        min_y = max(0, min(y for _, y in pts) - pad)
        max_x = max(x for x, _ in pts) + pad
        max_y = max(y for _, y in pts) + pad
        return int(min_x), int(min_y), int(max_x), int(max_y)

    # TODO: Ensure uniform masking area across different features (e.g., 800 pixels, 400 pixels for smaller features)
    def apply_mask(self, image: np.ndarray, features: list[str]) -> np.ndarray:
        """Applies a mask by setting the pixel values to zero within the bounding box of the specified facial feature in the image."""
        pix_coords = self._get_landmarks_in_pixels(image)
        if pix_coords is None:
            return image

        result = image.copy()
        for feature in features:
            min_x, min_y, max_x, max_y = self._get_bbox(pix_coords, feature)
            result[min_y:max_y, min_x:max_x] = 0

        return result

    def get_features(self, image: np.ndarray) -> list[FacialFeature]:
        """Returns a list of features representing bounding boxes for all defined facial features in the image."""
        pix_coords = self._get_landmarks_in_pixels(image)
        if pix_coords is None:
            return []

        features = []
        for feature_name in self.feature_map.keys():
            min_x, min_y, max_x, max_y = self._get_bbox(pix_coords, feature_name)
            feature = FacialFeature(min_x, min_y, max_x, max_y, feature_name)

            if feature.get_area() == 0:
                self.logger.warning(f"Feature '{feature}' ignored: area=0, min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")
                continue

            features.append(feature)

        return features
