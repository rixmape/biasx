# Getting Started with BiasX

This guide provides a basic example of how to perform a gender bias analysis using the `BiasX` package.

## Prerequisites

Ensure you have installed BiasX. If not, please follow the [Installation](installation.md) guide.

First, import the main analyzer class:

```python
from biasx import BiasAnalyzer
from biasx.types import DatasetSource, CAMMethod, LandmarkerSource # Import necessary enums
```

## Basic Analysis Workflow

Here's a minimal example demonstrating the core steps:

### Define Configuration

Instead of a separate file, we can define the configuration directly as a Python dictionary for simplicity. We'll use the UTKFace dataset, GradCAM for explanations, and limit the analysis to 100 samples for speed.

```python
# Define minimal configuration for the example
config = {
    "dataset": {
        "source": DatasetSource.UTKFACE.value, # Specify the dataset source
        "max_samples": 100,                   # Limit samples for a quick run
        "image_width": 48,                    # Example image size
        "image_height": 48,
        "color_mode": "L",                  # Example: Use grayscale
        "batch_size": 32                      # Example batch size
    },
    "model": {
        "path": "path/to/your/face_model.h5", # IMPORTANT: Replace with the actual path to your Keras model
        "inverted_classes": False,            # Set based on your model's output
        "batch_size": 32                      # Model prediction batch size
    },
    "explainer": {
        "landmarker_source": LandmarkerSource.MEDIAPIPE.value, # Use MediaPipe for landmarks
        "cam_method": CAMMethod.GRADCAM.value,              # Use GradCAM for explanations
        "cutoff_percentile": 90,                            # Example CAM cutoff
        "threshold_method": "otsu",                         # Example thresholding
        "overlap_threshold": 0.5,                           # Example overlap threshold
        "distance_metric": "euclidean",                     # Example distance metric
        "batch_size": 32                                    # Explainer processing batch size
    },
    "calculator": {
        "precision": 4 # Example precision for calculations
    }
    # Output configuration defaults are often sufficient for getting started
}
```

*Note: Remember to replace `"path/to/your/face_model.h5"` with the actual path to your trained face classification model.*

### Instantiate the Analyzer

Create an instance of `BiasAnalyzer`, passing the configuration dictionary.

```python
analyzer = BiasAnalyzer(config=config)
```

### Run the Analysis

Execute the full analysis pipeline. This will load the dataset, run predictions, generate explanations (landmarks and activation maps), and calculate bias metrics.

```python
results = analyzer.analyze()
```

### Inspect Results

The `analyze` method returns an `AnalysisResult` object containing detailed findings. You can inspect overall scores or dive into feature-specific analyses.

```python
# Print the overall BiasX disparity score
print(f"Overall BiasX Score: {results.disparity_scores.biasx}")

# Print the bias score calculated for a specific feature, e.g., the nose
if "nose" in results.feature_analyses:
        nose_analysis = results.feature_analyses["nose"]
        print(f"Nose Bias Score: {nose_analysis.bias_score}")
        print(f"  - Male Probability (Nose): {nose_analysis.male_probability}")
        print(f"  - Female Probability (Nose): {nose_analysis.female_probability}")

# Explore detailed explanations for each image (optional)
# for explanation in results.explanations:
#    print(f"Image ID: {explanation.image_data.image_id}, Predicted: {explanation.predicted_gender}")
#    # Access explanation.activation_boxes, explanation.landmark_boxes etc.
```

## Next Steps

* Explore the **[Configuration](configuration.md)** page for a detailed overview of all available settings.
* Dive into the specifics of each component in the **[API Reference](api/)**.
