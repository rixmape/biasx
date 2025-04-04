"""Coordinates the bias analysis pipeline and manages result aggregation."""

from typing import Dict, List, Union

import numpy as np

from .calculators import Calculator
from .config import Config, configurable
from .datasets import Dataset
from .explainers import Explainer
from .models import Model
from .types import AnalysisResult, Explanation, ImageData


@configurable("analyzer")
class BiasAnalyzer:
    """Orchestrates the end-to-end facial recognition bias analysis pipeline.

    This class coordinates the process of loading data, running model inference,
    generating visual explanations (activation maps and facial landmarks),
    and calculating various bias metrics. It integrates functionalities from
    Dataset, Model, Explainer, and Calculator components.

    Attributes:
        config (biasx.config.Config): The configuration object holding settings for all
            components (dataset, model, explainer, calculator).
        model (biasx.models.Model): An instance of the Model class for performing inference.
        dataset (biasx.datasets.Dataset): An instance of the Dataset class for loading and
            preprocessing data.
        explainer (biasx.explainers.Explainer): An instance of the Explainer class for generating
            visual explanations.
        calculator (biasx.calculators.Calculator): An instance of the Calculator class for
            computing bias metrics.
        batch_size (int): The batch size used for processing data during
            analysis, potentially overriding batch sizes specified in
            component configurations for the analysis loop itself.

    Examples:
        >>> # Using a configuration dictionary
        >>> config_dict = {
        ...     "dataset": {"source": "utkface", "max_samples": 100},
        ...     "model": {"path": "/path/to/model.h5"},
        ...     "explainer": {"cam_method": "gradcam"},
        ...     "calculator": {"precision": 4},
        ...     "analyzer": {"batch_size": 16}
        ... } #
        >>> analyzer = BiasAnalyzer(config=config_dict)
        >>> results = analyzer.analyze()
        >>> print(results.disparity_scores)

        >>> # Using a configuration file
        >>> analyzer = BiasAnalyzer.from_file("config.yaml") #
        >>> results = analyzer.analyze()
        >>> print(f"BiasX Score: {results.disparity_scores.biasx}")
    """

    def __init__(self, config: Union[Config, Dict, None] = None, batch_size: int = 32, **kwargs):
        """Initializes the BiasAnalyzer and its components.

        Sets up the Dataset, Model, Explainer, and Calculator based on the
        provided configuration.

        Args:
            config (Union[biasx.config.Config, Dict, None], optional): A
                configuration object (Config) or dictionary containing
                settings for the analyzer and its sub-components (Dataset,
                Model, Explainer, Calculator). If None, default
                configurations might be used or an error raised depending
                on the Config setup. Defaults to None.
            batch_size (int, optional): The batch size to use when iterating
                through the dataset during the `analyze` method. This primarily
                controls the batching within the analyzer's loop, distinct
                from potential batch sizes used internally by the model or
                explainer if configured differently. Defaults to 32.
            **kwargs: Additional keyword arguments, potentially used by the
                configurable decorator or passed down during component
                initialization if the Config structure supports it.
        """
        if config is None:
            config = {}

        self.config = config if isinstance(config, Config) else Config.create(config)
        self.model = Model(**self.config.model)
        self.dataset = Dataset(**self.config.dataset)
        self.explainer = Explainer(**self.config.explainer)
        self.calculator = Calculator(**self.config.calculator)
        self.batch_size = batch_size

    def analyze_batch(self, image_data_batch: List[ImageData]) -> List[Explanation]:
        """Analyzes a single batch of images through the pipeline.

        This method takes a list of ImageData objects, runs model prediction,
        generates explanations (activation maps, landmarks, labeled boxes),
        and compiles the results into Explanation objects.

        Args:
            image_data_batch (List[biasx.types.ImageData]): A list of ImageData
                objects, typically obtained from iterating over a Dataset
                instance. Each ImageData object should contain at least
                the preprocessed image (NumPy array) and the original PIL
                image.

        Returns:
            A list of Explanation objects, one for each image in the input
                batch. Each Explanation object contains the original image data,
                prediction results (gender, confidence), activation map, activation
                boxes (potentially labeled with facial features), and landmark
                boxes. Returns an empty list if the input batch is empty.

        Raises:
            (Potentially errors from underlying Model or Explainer methods if
                inference or explanation generation fails).
        """
        if not image_data_batch:
            return []

        batch_size = len(image_data_batch)
        if batch_size == 1:
            preprocessed_images = [image_data_batch[0].preprocessed_image]
            pil_images = [image_data_batch[0].pil_image]
        else:
            preprocessed_images = np.stack([img.preprocessed_image for img in image_data_batch])
            pil_images = [img.pil_image for img in image_data_batch]

        predictions = self.model.predict(preprocessed_images)
        predicted_genders = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]

        activation_maps, activation_boxes, landmark_boxes = self.explainer.explain_batch(
            pil_images=pil_images,
            preprocessed_images=preprocessed_images,
            model=self.model,
            target_classes=predicted_genders,
        )

        explanations = []
        for i in range(batch_size):
            explanation = Explanation(
                image_data=image_data_batch[i],
                predicted_gender=predicted_genders[i],
                prediction_confidence=confidences[i],
                activation_map=activation_maps[i],
                activation_boxes=activation_boxes[i],
                landmark_boxes=landmark_boxes[i],
            )
            explanations.append(explanation)

        return explanations

    def analyze(self) -> AnalysisResult:
        """Runs the full bias analysis pipeline on the configured dataset.

        This method iterates through the entire dataset provided by the
        Dataset component, processing images in batches using `analyze_batch`.
        It aggregates all the generated Explanation objects and then uses the
        Calculator component to compute feature-level bias analyses and
        overall disparity scores (like BiasX and Equalized Odds).

        Returns:
            An AnalysisResult object containing:

                - `explanations`: A list of all Explanation objects generated for
                each image in the dataset.
                - `feature_analyses`: A dictionary mapping each FacialFeature to its
                calculated FeatureAnalysis (bias score, per-gender probabilities).
                - `disparity_scores`: A DisparityScores object containing overall
                metrics like BiasX and Equalized Odds.

                Returns an empty AnalysisResult if the dataset yields no data or
                no explanations could be generated.

        Note:
            This method processes the *entire* dataset as configured in the
            Dataset component (respecting `max_samples`, shuffling, etc.).
            It can be computationally intensive depending on the dataset size
            and model complexity. It uses an internal buffer to manage
            memory usage during explanation aggregation.
        """
        batch_count = 0
        explanations_buffer = []
        total_explanations = []

        buffer_size = max(100, self.batch_size * 2)

        for batch in self.dataset:
            batch_explanations = self.analyze_batch(batch)
            explanations_buffer.extend(batch_explanations)
            batch_count += 1

            if len(explanations_buffer) >= buffer_size:
                total_explanations.extend(explanations_buffer)
                explanations_buffer = []

        if explanations_buffer:
            total_explanations.extend(explanations_buffer)

        if not total_explanations:
            return AnalysisResult()

        feature_analyses = self.calculator.calculate_feature_biases(total_explanations)
        disparity_scores = self.calculator.calculate_disparities(feature_analyses, total_explanations)

        return AnalysisResult(
            explanations=total_explanations,
            feature_analyses=feature_analyses,
            disparity_scores=disparity_scores,
        )

    @classmethod
    def from_file(cls, config_file_path: str) -> "BiasAnalyzer":
        """Creates a BiasAnalyzer instance from a configuration file.

        This factory method provides a convenient way to initialize the
        analyzer using an external configuration file (e.g., YAML, JSON)
        that defines the settings for all components.

        Args:
            config_file_path (str): The path to the configuration file. The file
                format should be supported by the underlying Config class's
                `from_file` method.

        Returns:
            A new instance of BiasAnalyzer configured according to the file.

        Examples:
            >>> analyzer = BiasAnalyzer.from_file('analysis_config.yaml') #
            >>> results = analyzer.analyze()
        """
        return cls(config=Config.from_file(config_file_path))
