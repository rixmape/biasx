"""
This module implements a comprehensive pipeline for analyzing gender bias in face classification models.
It provides tools for model evaluation, visual explanation generation, and bias metric calculation.
"""

from .analyzer import BiasAnalyzer
from .calculators import BiasCalculator
from .datasets import FaceDataset
from .explainers import VisualExplainer
from .models import ClassificationModel

__all__ = [
    "BiasAnalyzer",
    "BiasCalculator",
    "FaceDataset",
    "VisualExplainer",
    "ClassificationModel",
]
