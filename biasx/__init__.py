"""
This module implements a comprehensive pipeline for analyzing gender bias in face classification models.
It provides tools for model evaluation, visual explanation generation, and bias metric calculation.
"""

from .analyzer import BiasAnalyzer

__version__ = "0.1.1"

__all__ = [
    "BiasAnalyzer",
]
