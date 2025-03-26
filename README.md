# BiasX

BiasX is a comprehensive framework for quantifying and explaining gender bias in face classification models using a feature-based approach. Rather than just measuring bias with traditional metrics, BiasX reveals which facial features contribute to gender-specific misclassifications, providing explainable and actionable insights.

## Features

| **Feature**                   | **Description**                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| Feature-Level Bias Analysis   | Identifies bias contributions from specific facial features (eyes, nose, lips, etc.)             |
| Visual Explanation Generation | Creates activation maps to visualize model decision regions                                      |
| Comprehensive Metrics         | Calculates both traditional (equalized odds) and feature-based bias scores                       |
| Dataset Management            | Handles facial image datasets with demographic attributes                                        |
| Experiment Framework          | Supports controlled experiments with manipulated gender distributions and masked facial features |
| Interactive Visualization     | Includes a web application for exploring bias analysis results                                   |

## Installation

### From PyPI (Recommended)

BiasX is available on PyPI: <https://pypi.org/project/biasx/>

```bash
# Install directly from PyPI
pip install biasx
```

### From Source (For Development)

```bash
# Clone the repository
git clone https://github.com/rixmape/biasx.git
cd biasx

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from biasx import BiasAnalyzer

# Configure the analyzer
config = {
    "model": {
        "path": "path/to/gender/classifier.keras",
    },
    "explainer": {
        "landmarker_source": "mediapipe",
        "cam_method": "gradcam++",
        "cutoff_percentile": 90,
        "threshold_method": "otsu",
        "overlap_threshold": 0.2,
        "distance_metric": "euclidean",
    },
    "dataset": {
        "source": "utkface",
        "image_width": 224,
        "image_height": 224,
        "color_mode": "L",
        "single_channel": False,
        "max_samples": 100,
        "shuffle": True,
        "seed": 69,
    },
}

# Run analysis
analyzer = BiasAnalyzer(config)
results = analyzer.analyze()

# Access results
print(f"Overall BiasX Score: {results.disparity_scores.biasx}")
print(f"Equalized Odds Score: {results.disparity_scores.equalized_odds}")

# View feature-specific bias
for feature, analysis in results.feature_analyses.items():
    print(f"{feature.value}: Bias Score = {analysis.bias_score}")
```

## Web Application

The BiasX framework includes an interactive web application for visualizing and exploring bias analysis results.

### Local Deployment

```bash
# Run locally
streamlit run app/app.py

# Or use the make command
make deploy
```

### Public Deployment

The application is publicly available at: [biasxframework.streamlit.app](https://biasxframework.streamlit.app/)

## Development

Run tests with coverage reporting:

```bash
pytest --cov=biasx --cov-report=term --cov-report=html

# Or use the make command
make test
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use BiasX in your research, please cite our work:

```plaintext
@article{biasx2023,
  title={BiasX: A Feature-Based Framework for Explaining and Quantifying Gender Bias in Face Classification},
  author={Lucero, J. G. and Mape, R. N. and Sy, J. W.},
  year={2025}
}
```
