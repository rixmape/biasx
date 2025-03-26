import json
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
from PIL import Image

RANDOM_STATE: int = 42


def download_dataset(n: int = 1000, repo_id: str = "rixmape/utkface", filename: str = "data/train-00000-of-00001.parquet", repo_type: str = "dataset") -> pd.DataFrame:
    filepath: str = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    df: pd.DataFrame = pd.read_parquet(filepath)
    return df.sample(n=n, random_state=RANDOM_STATE) if n > 0 else df


def download_landmarker(repo_id: str = "rixmape/biasx-models", filename: str = "mediapipe_landmarker.task", repo_type: str = "model") -> FaceLandmarker:
    filepath: str = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    return FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=filepath)))


def load_landmark_mapping(filepath: str = "../data/landmark_mapping.json"):
    with open(filepath, "r") as f:
        return json.load(f)


def detect_landmarks(detector: FaceLandmarker, image: Image.Image) -> list[tuple[int, int]]:
    mp_image: Image.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
    result: FaceLandmarkerResult = detector.detect(mp_image)
    if not result.face_landmarks:
        return []
    image_width, image_height = image.size
    return [(int(round(point.x * image_width)), int(round(point.y * image_height))) for point in result.face_landmarks[0]]


def get_landmark_boxes(landmarks: list[tuple[int, int]], mapping: dict[str, list[int]]) -> list[dict[str, int]]:
    if not landmarks:
        return []
    boxes = []
    for feature, indices in mapping.items():
        feature_points = [landmarks[i] for i in indices]
        box = {
            "min_x": min(x for x, _ in feature_points),
            "min_y": min(y for _, y in feature_points),
            "max_x": max(x for x, _ in feature_points),
            "max_y": max(y for _, y in feature_points),
            "feature": feature,
        }
        boxes.append(box)
    return boxes


def visualize_landmark_boxes(images: list[Image.Image], results: list[list[dict[str, int]]], rows: int = 4, cols: int = 8) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes_flat = axes.flatten() if rows > 1 else [axes]
    for i, (image, boxes) in enumerate(zip(images, results)):
        if i >= rows * cols:
            break
        image = np.stack([np.array(image.convert("L")) / 255] * 3, axis=-1)  # Three-channel grayscale image
        if boxes:
            for box in boxes:
                x, y, w, h = box["min_x"], box["min_y"], box["max_x"] - box["min_x"], box["max_y"] - box["min_y"]
                cv2.rectangle(image, (x, y), (x + w, y + h), (1, 0, 0), 2)
        axes_flat[i].imshow(image)
        axes_flat[i].axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = download_dataset(n=100)

    images = [Image.open(BytesIO(row["image"]["bytes"])) for _, row in df.iterrows()]
    landmarker = download_landmarker()
    landmark_mapping = load_landmark_mapping()

    results = []
    for image in images:
        landmarks = detect_landmarks(landmarker, image)
        boxes = get_landmark_boxes(landmarks, landmark_mapping)
        results.append(boxes)

    visualize_landmark_boxes(images, results)


main()
