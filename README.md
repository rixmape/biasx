# BiasX

BiasX is a Python library for detecting and explaining gender bias in face classification models. This repository provides a toolkit to analyze bias through both traditional fairness metrics and feature-level analysis. Visual heatmaps and quantitative bias scores are generated to help developers understand which facial features contribute to biased classifications.



## Running Tests for BiasX

### Step 1: Install Test Dependencies
```
# Install test requirements using the provided file
pip install -r test-requirements.txt

# Install the BiasX package in development mode
pip install -e .

```
### Step 2: Run the Tests
#### Run all Test
```
pytest
````

#### Run Tests with Verbose Output
```
pytest -v

```
#### Run Tests with Coverage Report
```

pytest --cov=biasx --cov-report=term --cov-report=html
```

#### Run Specific Test Files
```
pytest tests/test_utils.py
```


