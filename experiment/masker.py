import json
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult

# isort: off
from config import ExperimentsConfig
from datatypes import Feature, FeatureDetails, MaskDetails
from utils import setup_logger


class FeatureMasker:

    def __init__(self, config: ExperimentsConfig, log_path: str):
        self.config = config
        self.landmarker = self._load_landmarker()
        self.feature_indices_map = self._load_feature_indices_map()
        self.logger = setup_logger(name="feature_masker", log_path=log_path)

    @staticmethod
    def _load_landmarker() -> FaceLandmarker:
        model_path = hf_hub_download(
            repo_id="rixmape/biasx-models",
            filename="mediapipe_landmarker.task",
            repo_type="model",
        )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        return FaceLandmarker.create_from_options(options)

    @staticmethod
    def _load_feature_indices_map() -> Dict[Feature, List[int]]:
        map_path = hf_hub_download(
            repo_id="rixmape/biasx-models",
            filename="landmark_map.json",
            repo_type="model",
        )
        with open(map_path, "r") as f:
            raw_map = json.load(f)
        feature_map = {Feature(key): value for key, value in raw_map.items()}
        return feature_map

    def _get_pixel_landmarks(self, image_np: np.ndarray, image_id: str) -> List[Tuple[int, int]]:
        if image_np.dtype in [np.float32, np.float64]:
            image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        else:
            image_uint8 = image_np.astype(np.uint8)

        if len(image_uint8.shape) != 3 or image_uint8.shape[-1] != 3:
            self.logger.error(f"[{image_id}] Invalid image shape/channels for landmark detection: {image_uint8.shape}")
            return []

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
        detection_result: FaceLandmarkerResult = self.landmarker.detect(mp_image)

        if not detection_result or not detection_result.face_landmarks:
            self.logger.warning(f"[{image_id}] No landmarks detected.")
            return []

        landmarks = detection_result.face_landmarks[0]
        img_size = self.config.dataset.image_size
        pixel_coords = [(int(pt.x * img_size), int(pt.y * img_size)) for pt in landmarks]
        return pixel_coords

    def _get_feature_bbox(
        self,
        pixel_coords: List[Tuple[int, int]],
        feature: Feature,
        image_id: str,
    ) -> Tuple[int, int, int, int]:
        indices = self.feature_indices_map[feature]
        if not indices or max(indices) >= len(pixel_coords):
            self.logger.warning(f"[{image_id}] Invalid or missing indices for feature '{feature.value}'.")
            return None

        points = [pixel_coords[i] for i in indices]
        min_x = min(x for x, y in points)
        min_y = min(y for x, y in points)
        max_x = max(x for x, y in points)
        max_y = max(y for x, y in points)

        pad = self.config.core.mask_pixel_padding
        img_size = self.config.dataset.image_size
        min_x_pad = max(0, min_x - pad)
        min_y_pad = max(0, min_y - pad)
        max_x_pad = min(img_size, max_x + pad)
        max_y_pad = min(img_size, max_y + pad)

        return min_x_pad, min_y_pad, max_x_pad, max_y_pad

    def apply_mask(self, image_np: np.ndarray, label: int, mask_details: Optional[MaskDetails], image_id: str) -> np.ndarray:
        if not mask_details or label != mask_details.target_gender.value:
            return image_np

        pixel_coords = self._get_pixel_landmarks(image_np, image_id)
        if not pixel_coords:
            return image_np

        masked_image = image_np.copy()
        for feature in mask_details.target_features:
            min_x, min_y, max_x, max_y = self._get_feature_bbox(pixel_coords, feature, image_id)
            masked_image[min_y:max_y, min_x:max_x] = 0

        return masked_image

    def get_features(self, image_np: np.ndarray, image_id: str) -> List[FeatureDetails]:
        pixel_coords = self._get_pixel_landmarks(image_np, image_id)
        if not pixel_coords:
            return []

        detected_features: List[FeatureDetails] = []
        for feature in self.feature_indices_map.keys():
            min_x, min_y, max_x, max_y = self._get_feature_bbox(pixel_coords, feature, image_id)
            feature_detail = FeatureDetails(feature=feature, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
            detected_features.append(feature_detail)

        if not detected_features:
            self.logger.warning(f"[{image_id}] No valid feature bounding boxes could be generated.")

        return detected_features
