"""Provides classes for generating and processing visual explanations of model decisions."""

import json
from typing import List, Tuple, Union

import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops

from .config import configurable
from .models import Model
from .types import Box, CAMMethod, DistanceMetric, FacialFeature, Gender, LandmarkerSource, ResourceMetadata, ThresholdMethod
from .utils import get_file_path, get_json_config, get_resource_path


class FacialLandmarker:
    """Detects facial landmarks using MediaPipe.

    This class loads a pre-trained MediaPipe face landmark model, specified
    via configuration, and provides a method to detect landmarks in images.
    It maps the detected landmark points to specific facial features based
    on a predefined mapping file.

    Attributes:
        source (biasx.types.LandmarkerSource): The source identifier for the
            landmarker model used (e.g., MEDIAPIPE).
        landmarker_info (biasx.types.ResourceMetadata): Metadata loaded from
            configuration about the landmarker model resource.
        model_path (str): The local path to the downloaded landmarker model file.
        landmark_mapping (Dict[biasx.types.FacialFeature, List[int]]): A dictionary
            mapping facial features (enum) to lists of landmark indices provided
            by the MediaPipe model.
        detector (mediapipe.tasks.python.vision.FaceLandmarker): The initialized
            MediaPipe FaceLandmarker instance.
    """

    def __init__(self, source: LandmarkerSource):
        """Initialize the facial landmark detector.

        Loads resources based on the specified source, sets up the MediaPipe
        FaceLandmarker options, and creates the detector instance.

        Args:
            source (biasx.types.LandmarkerSource): The source of the landmarker
                model to use (e.g., LandmarkerSource.MEDIAPIPE). Corresponds to
                keys in `landmarker_config.json`.
        """
        self.source = source
        self._load_resources()

        options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=self.model_path))
        self.detector = FaceLandmarker.create_from_options(options)

    def _load_resources(self) -> None:
        """Load landmarker resources from configuration and mapping files.

        Reads `landmarker_config.json` to get model metadata based on `self.source`.
        Uses `get_resource_path` to download or locate the model file specified
        in the config. Reads `landmark_mapping.json` to create the mapping
        from MediaPipe landmark indices to `FacialFeature` enums.

        Raises:
            ValueError: If `self.source` is not found in `landmarker_config.json`.
            FileNotFoundError: If the model file or mapping file cannot be found.
            (Potentially other errors from `hf_hub_download` or file I/O).
        """
        config = get_json_config(__file__, "landmarker_config.json")

        if self.source.value not in config:
            raise ValueError(f"Landmarker source {self.source.value} not found in configuration")

        metadata_dict = config[self.source.value]
        self.landmarker_info = ResourceMetadata(**metadata_dict)
        self.model_path = get_resource_path(repo_id=self.landmarker_info.repo_id, filename=self.landmarker_info.filename, repo_type=self.landmarker_info.repo_type)

        mapping_path = get_file_path(__file__, "data/landmark_mapping.json")
        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)

        self.landmark_mapping = {}
        for feature_name, indices in mapping_data.items():
            feature_enum = FacialFeature(feature_name)
            self.landmark_mapping[feature_enum] = indices

    def detect(self, images: Union[Image.Image, List[Image.Image]]) -> List[List[Box]]:
        """Detect facial landmarks in one or more images.

        Takes a single PIL image or a list of PIL images, converts them into
        MediaPipe's Image format, and runs detection using the initialized
        landmarker. For each image where landmarks are detected, it converts
        the normalized landmark coordinates to pixel coordinates and groups
        them into bounding boxes based on the `self.landmark_mapping` for
        each facial feature.

        Args:
            images (Union[PIL.Image.Image, List[PIL.Image.Image]]): A single
                PIL image or a list of PIL images to process.

        Returns:
            A list of lists of Box objects. The outer list corresponds to the
                input images. Each inner list contains `biasx.types.Box` objects,
                one for each facial feature defined in the mapping, representing the
                bounding box encompassing the landmarks for that feature in the
                corresponding image. If no landmarks are detected in an image, its
                corresponding inner list will be empty.
        """
        if isinstance(images, Image.Image):
            images = [images]

        results = []
        for image in images:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
            result = self.detector.detect(mp_image)

            if not result.face_landmarks:
                results.append([])
                continue

            image_width, image_height = image.size
            points = [(int(round(point.x * image_width)), int(round(point.y * image_height))) for point in result.face_landmarks[0]]

            boxes = []
            for feature, indices in self.landmark_mapping.items():
                feature_points = [points[i] for i in indices]
                box = Box(
                    min_x=min(x for x, _ in feature_points),
                    min_y=min(y for _, y in feature_points),
                    max_x=max(x for x, _ in feature_points),
                    max_y=max(y for _, y in feature_points),
                    feature=feature,
                )
                boxes.append(box)

            results.append(boxes)

        return results


class ClassActivationMapper:
    """Generates and processes class activation maps (CAMs).

    This class uses a specified CAM generation method (e.g., Grad-CAM++, Score-CAM)
    from the `tf-keras-vis` library to produce heatmaps highlighting regions
    important for a model's classification decision. It also provides methods
    to process these heatmaps by applying thresholding and identifying
    contiguous activated regions as bounding boxes.

    Attributes:
        cam_method (tf_keras_vis.ModelVisualization): The instantiated CAM
            visualization object (e.g., GradcamPlusPlus) based on the configured method.
        cutoff_percentile (int): The percentile value (0-100) used to determine
            the threshold for filtering low-activation areas in the heatmap.
        threshold_method (Callable[[np.ndarray], Any]): The thresholding function
            (e.g., skimage.filters.threshold_otsu) used to binarize the filtered heatmap.
    """

    def __init__(self, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod):
        """Initialize the activation map generator and processor.

        Args:
            cam_method (biasx.types.CAMMethod): The CAM algorithm to use
                (e.g., CAMMethod.GRADCAM_PLUS_PLUS).
            cutoff_percentile (int): The percentile (0-100) to use for thresholding
                the raw heatmap. Activations below this percentile are zeroed out.
            threshold_method (biasx.types.ThresholdMethod): The algorithm used
                to binarize the filtered heatmap (e.g., ThresholdMethod.OTSU).
        """
        self.cam_method = cam_method.get_implementation()
        self.cutoff_percentile = cutoff_percentile
        self.threshold_method = threshold_method.get_implementation()

    def generate_heatmap(self, model: tf.keras.Model, preprocessed_images: Union[np.ndarray, List[np.ndarray]], target_classes: Union[Gender, List[Gender]]) -> List[np.ndarray]:
        """Generate class activation maps (heatmaps) for preprocessed images.

        Uses the configured CAM method (`tf-keras-vis`) to generate heatmaps for
        the given images with respect to the specified target classes. It handles
        preparing the images and defining the score function for the visualizer.

        Args:
            model (tf.keras.Model): The trained Keras model to explain.
            preprocessed_images (Union[np.ndarray, List[np.ndarray]]): A single
                preprocessed image (NumPy array) or a list/batch of preprocessed
                images (NumPy array). Assumes images are normalized and correctly shaped.
            target_classes (Union[biasx.types.Gender, List[biasx.types.Gender]]): The target class(es)
                (Gender enum) for which to generate the heatmaps. If a single
                Gender is provided, it's used for all images in the batch.

        Returns:
            A list of NumPy arrays, where each array is a 2D heatmap corresponding
                to an input image. Returns an empty list if the input is empty.
        """
        visualizer = self.cam_method(model, model_modifier=self._modify_model, clone=True)

        if not isinstance(preprocessed_images, np.ndarray) and len(preprocessed_images) == 0:
            return []

        images = self._prepare_images_for_cam(preprocessed_images)

        if isinstance(target_classes, Gender):
            target_classes = [target_classes] * len(images)

        heatmaps = []
        for i, target_class in enumerate(target_classes):
            score_function = lambda output: output[0][target_class]
            image_batch = np.expand_dims(images[i], axis=0) if images[i].ndim == 3 else images[i]
            heatmap = visualizer(score_function, image_batch, penultimate_layer=-1)[0]
            heatmaps.append(heatmap)

        return heatmaps

    def process_heatmap(self, heatmaps: Union[np.ndarray, List[np.ndarray]], pil_images: List[Image.Image]) -> List[List[Box]]:
        """Process heatmaps into bounding boxes of activated regions.

        Takes raw heatmaps, applies a percentile cutoff threshold, binarizes the
        result using the configured thresholding method, identifies connected
        regions (blobs) in the binary map, and converts these regions into
        bounding boxes scaled to the original image dimensions.

        Args:
            heatmaps (Union[np.ndarray, List[np.ndarray]]): A single 2D heatmap
                or a list of 2D heatmaps (NumPy arrays) as generated by
                `generate_heatmap`.
            pil_images (List[PIL.Image.Image]): A list of the original PIL images
                corresponding to the heatmaps, used to get dimensions for scaling.
                Must be the same length as the list of heatmaps.

        Returns:
            A list of lists of Box objects. The outer list corresponds to the
                input heatmaps/images. Each inner list contains `biasx.types.Box`
                objects representing the bounding boxes of activated regions found
                in the corresponding heatmap. Boxes do not have features assigned
                at this stage.
        """
        if isinstance(heatmaps, np.ndarray) and heatmaps.ndim <= 2:
            heatmaps = [heatmaps]

        results = []
        for heatmap, pil_image in zip(heatmaps, pil_images):
            threshold_value = np.percentile(heatmap, self.cutoff_percentile)
            filtered_heatmap = np.where(heatmap < threshold_value, 0, heatmap)

            binary = filtered_heatmap > self.threshold_method(filtered_heatmap)
            regions = regionprops(label(binary))

            img_width, img_height = pil_image.size
            scale_x = img_width / heatmap.shape[1]
            scale_y = img_height / heatmap.shape[0]

            boxes = [
                Box(
                    min_x=int(region.bbox[1] * scale_x),
                    min_y=int(region.bbox[0] * scale_y),
                    max_x=int(region.bbox[3] * scale_x),
                    max_y=int(region.bbox[2] * scale_y),
                )
                for region in regions
            ]
            results.append(boxes)

        return results

    def _prepare_images_for_cam(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Prepare image array(s) for CAM processing.

        Ensures the input images are in a NumPy array format suitable for
        `tf-keras-vis` (typically a 4D batch array NCHW or NHWC). Handles
        conversion from lists, adding batch dimensions, and adding channel
        dimensions for grayscale images if necessary.

        Args:
            images (Union[np.ndarray, List[np.ndarray]]): A single image or list/batch
                of images as NumPy arrays.

        Returns:
            A NumPy array ready for input into the CAM visualizer, typically 4D.
        """
        if isinstance(images, list):
            processed_batch = []
            for img in images:
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=-1)
                processed_batch.append(img)
            return np.stack(processed_batch)

        if images.ndim == 2:
            images = np.expand_dims(images, axis=-1)
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        return images

    @staticmethod
    def _modify_model(model: tf.keras.Model) -> None:
        """Modify the model's final activation for CAM generation (inplace).

        Required by `tf-keras-vis`. Sets the activation function of the model's
        output layer to linear. This is often necessary for CAM methods like
        Grad-CAM to work correctly, as they operate on pre-activation outputs.

        Args:
            model (tf.keras.Model): The Keras model to modify. Modification happens inplace.
        """
        model.layers[-1].activation = tf.keras.activations.linear


@configurable("explainer")
class Explainer:
    """Coordinates the generation of visual explanations for model decisions.

    This class integrates the `FacialLandmarker` and `ClassActivationMapper`
    to produce comprehensive explanations for a batch of images. It generates
    activation maps, processes them into activation boxes, detects landmark
    boxes, and attempts to label activation boxes based on their spatial overlap
    and proximity to landmark features.

    Attributes:
        landmarker (FacialLandmarker): An instance for detecting facial landmarks.
        activation_mapper (ClassActivationMapper): An instance for generating and
            processing activation maps.
        overlap_threshold (float): The minimum Intersection over Area (IoA) threshold
            required between an activation box and a landmark box for the
            activation box to inherit the landmark's feature label.
        distance_metric (str): The distance metric (e.g., 'euclidean', 'cityblock')
            used to find the nearest landmark box center for each activation box center.
        batch_size (int): The batch size hint used within this explainer, potentially
            influencing internal operations if implemented differently later.
            (Note: Current `explain_batch` processes the whole input batch at once).
    """

    def __init__(self, landmarker_source: LandmarkerSource, cam_method: CAMMethod, cutoff_percentile: int, threshold_method: ThresholdMethod, overlap_threshold: float, distance_metric: DistanceMetric, batch_size: int, **kwargs):
        """Initialize the visual explainer and its components.

        Creates instances of `FacialLandmarker` and `ClassActivationMapper`
        based on the provided configuration parameters. Stores thresholds and
        metrics used for associating activation boxes with features.

        Args:
            landmarker_source (biasx.types.LandmarkerSource): The source for the
                facial landmark model.
            cam_method (biasx.types.CAMMethod): The class activation mapping
                method to use.
            cutoff_percentile (int): The percentile threshold for heatmap processing.
            threshold_method (biasx.types.ThresholdMethod): The binarization method
                for heatmap processing.
            overlap_threshold (float): The IoA threshold (0.0 to 1.0) for labeling
                activation boxes based on landmark box overlap.
            distance_metric (biasx.types.DistanceMetric): The metric for comparing
                box center distances.
            batch_size (int): A batch size parameter (currently informational).
            **kwargs: Additional keyword arguments passed via configuration.
        """
        self.landmarker = FacialLandmarker(source=landmarker_source)
        self.activation_mapper = ClassActivationMapper(cam_method=cam_method, cutoff_percentile=cutoff_percentile, threshold_method=threshold_method)
        self.overlap_threshold = overlap_threshold
        self.distance_metric = distance_metric.value
        self.batch_size = batch_size

    def explain_batch(self, pil_images: List[Image.Image], preprocessed_images: List[np.ndarray], model: Model, target_classes: List[Gender]) -> Tuple[List[np.ndarray], List[List[Box]], List[List[Box]]]:
        """Generate visual explanations for a batch of images.

        Orchestrates the explanation process for a batch:

        1. Generates activation maps using `ClassActivationMapper`.
        2. Processes maps into activation boxes using `ClassActivationMapper`.
        3. Detects landmark boxes using `FacialLandmarker`.
        4. For each image, attempts to assign a `FacialFeature` label to each
           activation box by finding the nearest landmark box (based on center
           distance) and checking if their spatial overlap (IoA) meets the
           `overlap_threshold`.

        Args:
            pil_images (List[PIL.Image.Image]): List of original PIL images in the batch.
            preprocessed_images (List[np.ndarray]): List of corresponding preprocessed
                images (NumPy arrays) ready for model/CAM input.
            model (biasx.models.Model): The model instance (needed by the activation mapper).
            target_classes (List[biasx.types.Gender]): The target class (Gender) for
                generating the CAM for each corresponding image.

        Returns:
            A tuple containing three lists, all aligned with the input batch order:

                - List[np.ndarray]: Raw activation maps (heatmaps) for each image.
                - List[List[biasx.types.Box]]: Activation boxes for each image. Boxes
                  may have their `feature` attribute set if successfully labeled.
                - List[List[biasx.types.Box]]: Landmark boxes for each image, with
                  their `feature` attribute set by the landmarker.

                Returns ([], [], []) if the input `pil_images` list is empty.
        """
        if not pil_images:
            return [], [], []

        activation_maps = self.activation_mapper.generate_heatmap(model.model, preprocessed_images, target_classes)
        activation_boxes = self.activation_mapper.process_heatmap(activation_maps, pil_images)
        landmark_boxes = self.landmarker.detect(pil_images)

        labeled_boxes = []
        for a_boxes, l_boxes in zip(activation_boxes, landmark_boxes):
            if not a_boxes or not l_boxes:
                labeled_boxes.append(a_boxes)
                continue

            a_centers = np.array([box.center for box in a_boxes])
            l_centers = np.array([box.center for box in l_boxes])
            distances = cdist(a_centers, l_centers, metric=self.distance_metric)

            nearest_indices = np.argmin(distances, axis=1)

            for i, (a_box, nearest_idx) in enumerate(zip(a_boxes, nearest_indices)):
                l_box = l_boxes[nearest_idx]

                overlap_width = max(0, min(a_box.max_x, l_box.max_x) - max(a_box.min_x, l_box.min_x))
                overlap_height = max(0, min(a_box.max_y, l_box.max_y) - max(a_box.min_y, l_box.min_y))
                overlap_area = overlap_width * overlap_height

                if overlap_area / a_box.area >= self.overlap_threshold:
                    a_box.feature = l_box.feature

            labeled_boxes.append(a_boxes)

        return activation_maps, labeled_boxes, landmark_boxes
