# Installation

This guide will walk you through installing the `BiasX` package.

## Requirements

Before installing BiasX, ensure you have the following prerequisites installed on your system:

* **Python:** A recent version of Python (e.g., 3.8 or later is recommended).
* **pip:** The Python package installer. This usually comes bundled with modern Python versions.

You can check your installations by running:

```bash
python --version
pip --version
```

If you need to install Python, please visit [python.org](https://www.python.org/downloads/). If you need to install or upgrade pip, follow the instructions on the [pip documentation](https://pip.pypa.io/en/stable/installation/).

## Installing BiasX

You can install the `BiasX` package directly from PyPI using pip:

```bash
pip install biasx
```

This command will download and install BiasX along with its necessary dependencies (such as TensorFlow, NumPy, MediaPipe, Hugging Face Hub, etc.).

## Verifying the Installation (Optional)

To verify that BiasX was installed correctly, you can open a Python interpreter and try importing the main class:

```python
try:
    from biasx import BiasAnalyzer
    print("BiasX installed successfully!")
except ImportError:
    print("Error: BiasX not found. Please check your installation.")

```

You are now ready to start using BiasX! Head over to the **[Getting Started](getting_started.md)** guide for a basic usage example.
