# BiasX

BiasX is a Python library for detecting and explaining gender bias in face classification models. This repository provides a toolkit to analyze bias through both traditional fairness metrics and feature-level analysis. Visual heatmaps and quantitative bias scores are generated to help developers understand which facial features contribute to biased classifications.

## Local deployment

```bash
streamlit run app/app.py
```

## Software testing

```bash
pytest --cov=biasx --cov-report=term --cov-report=html
```
