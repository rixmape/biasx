# Configuration

BiasX uses a configuration system to manage parameters for datasets, models, explainers, and calculators. You can configure the `BiasAnalyzer` either by passing a Python dictionary directly during instantiation or by loading settings from a configuration file (commonly YAML, though JSON is also supported via utility functions).

## Configuration Structure

The configuration is typically structured with top-level keys corresponding to the main components: `dataset`, `model`, `explainer`, and `calculator`.

```python
# Example Structure (Python Dictionary)
config = {
    "dataset": {
        # Dataset parameters...
    },
    "model": {
        # Model parameters...
    },
    "explainer": {
        # Explainer parameters...
    },
    "calculator": {
        # Calculator parameters...
    },
    "analyzer": { # Optional: Parameters directly for BiasAnalyzer itself
        "batch_size": 32
    }
}

# Example Instantiation
from biasx import BiasAnalyzer
analyzer = BiasAnalyzer(config=config)

# Example Instantiation from file
analyzer_from_file = BiasAnalyzer.from_file("path/to/your/config.yaml") #
```

## Component Configuration Details

Each component class uses the `@configurable` decorator, meaning its `__init__` parameters can be set via the configuration dictionary under the corresponding key (`dataset`, `model`, `explainer`, `calculator`).

### Dataset (`dataset`)

Parameters for the `biasx.datasets.Dataset` class.

* **`source`**: (Required) The source dataset to use. Should correspond to values in the `DatasetSource` enum (e.g., `"utkface"`, `"fairface"`).
* **`image_width`**: (Required) The target width to resize images to.
* **`image_height`**: (Required) The target height to resize images to.
* **`color_mode`**: (Required) The target color mode. Should correspond to values in the `ColorMode` enum (e.g., `"L"` for grayscale, `"RGB"` for color).
* **`max_samples`**: (Optional) Maximum number of samples to load from the dataset. If `0` or less, all samples are loaded. Defaults depend on the implementation, but often it's useful to set a smaller number for testing (e.g., `100`).
* **`shuffle`**: (Optional) Whether to shuffle the dataset before selecting `max_samples` or iterating. Defaults to `False`.
* **`seed`**: (Optional) Random seed used for shuffling if `shuffle` is `True`. Defaults depend on implementation (e.g., `42`).
* **`batch_size`**: (Optional) The number of images yielded per iteration when iterating over the dataset. Defaults depend on implementation (e.g., `32`).

### Model (`model`)

Parameters for the `biasx.models.Model` class.

* **`path`**: (Required) Filesystem path to the saved Keras/TensorFlow model file (e.g., `.h5` or a SavedModel directory).
* **`inverted_classes`**: (Required) Boolean indicating if the model's output class indices for Male/Female are inverted compared to the `Gender` enum (Male=0, Female=1). Set to `True` if your model predicts Female as 0 and Male as 1.
* **`batch_size`**: (Optional) Batch size used for model prediction (`model.predict`). Defaults depend on implementation (e.g., `64`).

### Explainer (`explainer`)

Parameters for the `biasx.explainers.Explainer` class.

* **`landmarker_source`**: (Required) Specifies the facial landmark detection model source. Should correspond to values in the `LandmarkerSource` enum (e.g., `"mediapipe"`).
* **`cam_method`**: (Required) Specifies the Class Activation Map method to use. Should correspond to values in the `CAMMethod` enum (e.g., `"gradcam"`, `"gradcam++"`, `"scorecam"`).
* **`cutoff_percentile`**: (Required) Integer percentile (0-100) used to threshold the raw activation map. Pixels below this percentile intensity are set to zero.
* **`threshold_method`**: (Required) Specifies the method used to binarize the thresholded activation map to find distinct activation regions. Should correspond to values in the `ThresholdMethod` enum (e.g., `"otsu"`, `"sauvola"`, `"triangle"`).
* **`overlap_threshold`**: (Required) Float value (0.0-1.0) determining the minimum overlap area (as a fraction of the activation box area) required to associate an activation box with a landmark box.
* **`distance_metric`**: (Required) Specifies the distance metric used to find the nearest landmark center to an activation center. Should correspond to values in the `DistanceMetric` enum (e.g., `"cityblock"`, `"cosine"`, `"euclidean"`).
* **`batch_size`**: (Optional) Batch size used during the explanation generation process (specifically for CAM generation). Defaults depend on implementation (e.g., `32`).

### Calculator (`calculator`)

Parameters for the `biasx.calculators.Calculator` class.

* **`precision`**: (Required) Integer specifying the number of decimal places to round calculated bias scores and probabilities to.

### Analyzer (`analyzer`)

Optional parameters directly for the `biasx.analyzer.BiasAnalyzer` class itself.

* **`batch_size`**: (Optional) Controls the batch size used when iterating through the dataset *within* the analyzer's main `analyze` loop. This is distinct from the dataset's own iteration batch size or the model/explainer batch sizes. Defaults depend on implementation (e.g., `32`).

## Example Configuration File (`config.yaml`)

```yaml
# Dataset Configuration
dataset:
  source: utkface         # Source dataset name (enum value)
  max_samples: 5000      # Max images to load (0 for all)
  shuffle: true          # Shuffle before selecting max_samples
  seed: 42               # Random seed for shuffling
  image_width: 48        # Target image width
  image_height: 48       # Target image height
  color_mode: "L"        # "L" for grayscale, "RGB" for color
  batch_size: 64         # Batch size for dataset iteration

# Model Configuration
model:
  path: "./models/my_face_classifier.h5" # Path to your trained model
  inverted_classes: false              # Does your model output 0=Female, 1=Male?
  batch_size: 128                      # Batch size for model.predict()

# Explainer Configuration
explainer:
  landmarker_source: mediapipe # Source for facial landmarks
  cam_method: gradcam++        # CAM method (gradcam, gradcam++, scorecam)
  cutoff_percentile: 95        # Percentile for CAM heatmap thresholding
  threshold_method: otsu       # Method to binarize heatmap (otsu, sauvola, triangle)
  overlap_threshold: 0.3       # Min overlap to link activation box to landmark
  distance_metric: euclidean   # Metric for nearest landmark (cityblock, cosine, euclidean)
  batch_size: 64               # Batch size for CAM generation

# Calculator Configuration
calculator:
  precision: 4               # Decimal places for calculated scores

# Analyzer Configuration (Optional)
analyzer:
  batch_size: 64             # Batch size for the main analysis loop
```
