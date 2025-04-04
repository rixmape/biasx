# Welcome to BiasX

**BiasX** is a Python library designed for the comprehensive analysis of gender bias in face classification models. It provides a pipeline to evaluate model performance, generate visual explanations for model decisions, and calculate quantitative bias metrics.

Understanding and mitigating bias in AI models is crucial, especially in sensitive applications like facial recognition. BiasX aims to provide researchers and developers with the tools needed to identify how and why their models might exhibit different behaviors across demographic groups.

## Key Features

* **Model Loading & Inference:** Handles loading TensorFlow/Keras models and running predictions.
* **Dataset Management:** Loads and preprocesses standard facial image datasets like UTKFace and FairFace.
* **Visual Explanations:**
  * Generates Class Activation Maps (CAMs) using methods like GradCAM, GradCAM++, and ScoreCAM to highlight regions influencing predictions.
  * Detects facial landmarks (eyes, nose, mouth, etc.) using providers like MediaPipe.
* **Bias Calculation:**
  * Analyzes the correlation between activated facial features and misclassifications across genders.
  * Calculates overall bias disparity scores, including Equalized Odds.
* **Configurable Pipeline:** Allows easy configuration of datasets, models, explanation methods, and analysis parameters through YAML files or Python dictionaries.

## Getting Started

New to BiasX? Start with the **[Getting Started](getting_started.md)** guide to walk through a basic analysis example.

Check the **[Installation](installation.md)** guide for setup instructions.

Explore the **[Configuration](configuration.md)** options to customize the analysis pipeline.

Dive deep into the components with the **[API Reference](api/)**.
