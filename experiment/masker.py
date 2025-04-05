import json
import logging
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult

# isort: off
from config import Config
from datatypes import BoundingBox, Feature, FeatureDetails


class FeatureMasker:
    """Detects facial features, provides bounding boxes, and applies masks."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initializes the FeatureMasker with configuration and logger."""
        self.config = config
        self.logger = logger
        self.landmarker = self._load_landmarker()
        self.feature_indices_map = self._load_feature_indices_map()
        self.logger.info("Completed feature masker initialization")

    def _load_landmarker(self) -> FaceLandmarker:
        """Loads the MediaPipe face landmarker model from HuggingFace Hub."""
        self.logger.debug("Starting landmarker model loading")
        try:
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
            landmarker = FaceLandmarker.create_from_options(options)
            self.logger.info("Completed loading landmark detection model")
            return landmarker
        except Exception as e:
            self.logger.error(f"Failed to load landmarker model: {e}", exc_info=True)
            raise

    def _load_feature_indices_map(self) -> Dict[Feature, List[int]]:
        """Loads the mapping from facial features to landmark indices from JSON."""
        self.logger.debug("Starting feature indices map loading")
        try:
            map_path = hf_hub_download(
                repo_id="rixmape/biasx-models",
                filename="landmark_map.json",
                repo_type="model",
            )
            with open(map_path, "r") as f:
                raw_map = json.load(f)

            feature_map = {Feature(key): value for key, value in raw_map.items()}
            self.logger.info("Completed loading feature indices map")
            return feature_map
        except FileNotFoundError:
            self.logger.error(f"Feature map file not found: path={map_path}", exc_info=True)
            raise
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON: path={map_path}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Failed to load feature indices map: {e}", exc_info=True)
            raise

    def _get_pixel_landmarks(
        self,
        image_np: np.ndarray,
        image_id: str,
    ) -> List[Tuple[int, int]]:
        """Detects face landmarks in an image and returns pixel coordinates."""
        if image_np.dtype in [np.float32, np.float64]:
            image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        elif image_np.dtype == np.uint8:
            image_uint8 = image_np
        else:
            self.logger.error(f"[{image_id}] Unsupported image dtype: {image_np.dtype}")
            return []

        if len(image_uint8.shape) != 3 or image_uint8.shape[-1] != 3:
            self.logger.error(f"[{image_id}] Invalid image shape/channels: {image_uint8.shape}")
            return []

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
            detection_result: Optional[FaceLandmarkerResult] = self.landmarker.detect(mp_image)
        except Exception as e:
            self.logger.error(f"[{image_id}] MediaPipe landmark detection failed: {e}", exc_info=True)
            return []

        if not detection_result or not detection_result.face_landmarks:
            return []

        landmarks = detection_result.face_landmarks[0]
        img_size_h, img_size_w = image_np.shape[:2]
        pixel_coords = [(int(pt.x * img_size_w), int(pt.y * img_size_h)) for pt in landmarks]

        return pixel_coords

    def _get_feature_bbox(
        self,
        pixel_coords: List[Tuple[int, int]],
        feature: Feature,
        image_id: str,
        img_height: int,
        img_width: int,
    ) -> Optional[BoundingBox]:
        """Calculates the padded bounding box for a specific feature from landmarks."""
        indices = self.feature_indices_map.get(feature)
        if not indices:
            self.logger.warning(f"[{image_id}] No indices found: feature='{feature.value}'")
            return None
        if max(indices) >= len(pixel_coords):
            self.logger.warning(f"[{image_id}] Index out of bounds: feature='{feature.value}', max_index={max(indices)}, landmarks={len(pixel_coords)}")
            return None

        try:
            points = [pixel_coords[i] for i in indices]
            min_x = min(x for x, y in points)
            min_y = min(y for x, y in points)
            max_x = max(x for x, y in points)
            max_y = max(y for x, y in points)
        except IndexError:
            self.logger.error(
                f"[{image_id}] IndexError accessing coords: feature='{feature.value}', indices={indices}, total_coords={len(pixel_coords)}",
                exc_info=True,
            )
            return None

        pad = self.config.core.mask_pixel_padding
        min_x_pad = max(0, min_x - pad)
        min_y_pad = max(0, min_y - pad)
        max_x_pad = min(img_width, max_x + pad)
        max_y_pad = min(img_height, max_y + pad)

        if min_x_pad >= max_x_pad or min_y_pad >= max_y_pad:
            self.logger.warning(f"[{image_id}] Invalid bbox after padding: feature='{feature.value}', box=({min_x_pad}, {min_y_pad}, {max_x_pad}, {max_y_pad})")
            return None

        return BoundingBox(min_x=min_x_pad, min_y=min_y_pad, max_x=max_x_pad, max_y=max_y_pad)

    def apply_mask(self, image_np: np.ndarray, label: int, image_id: str) -> np.ndarray:
        """Applies configured masks (zeros out regions) if the label matches the target."""
        if not self.config.core.mask_gender or label != self.config.core.mask_gender.value:
            return image_np
        if not self.config.core.mask_features:
            self.logger.warning(f"[{image_id}] Masking requested but no features specified: mask_gender={self.config.core.mask_gender.name}")
            return image_np

        pixel_coords = self._get_pixel_landmarks(image_np, image_id)
        if not pixel_coords:
            # self.logger.warning(f"[{image_id}] Skipping masking: no landmarks found")
            return image_np

        masked_image = image_np.copy()
        img_height, img_width = image_np.shape[:2]
        applied_mask_count = 0

        for feature in self.config.core.mask_features:
            bbox = self._get_feature_bbox(pixel_coords, feature, image_id, img_height, img_width)
            if bbox:
                masked_image[bbox.min_y : bbox.max_y, bbox.min_x : bbox.max_x] = 0
                applied_mask_count += 1

        if applied_mask_count == 0:
            self.logger.warning(f"[{image_id}] No valid features found to mask")

        return masked_image

    def get_features(
        self,
        image_np: np.ndarray,
        image_id: str,
    ) -> List[FeatureDetails]:
        """Detects all facial features and returns their details including bounding boxes."""
        pixel_coords = self._get_pixel_landmarks(image_np, image_id)
        if not pixel_coords:
            # self.logger.warning(f"[{image_id}] Cannot get features: no landmarks found")
            return []

        img_height, img_width = image_np.shape[:2]
        detected_features: List[FeatureDetails] = []

        for feature in self.feature_indices_map.keys():
            bbox = self._get_feature_bbox(pixel_coords, feature, image_id, img_height, img_width)
            if bbox:
                feature_detail = FeatureDetails(feature=feature, bbox=bbox)
                detected_features.append(feature_detail)

        if not detected_features:
            self.logger.warning(f"[{image_id}] No valid feature bboxes generated")

        return detected_features
