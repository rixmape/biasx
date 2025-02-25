from pathlib import Path
from tempfile import mktemp
from typing import Any, Callable

import gradio as gr
import h5py
import plotly.graph_objects as go

from biasx import BiasAnalyzer
from biasx.config import Config
from biasx.defaults import create_default_config
from biasx.types import CAMMethod, ColorMode, DistanceMetric, ThresholdMethod

from .graphs import (
    create_confusion_matrix,
    create_parallel_coordinates,
    create_radar_chart,
    create_roc_curves,
    create_spatial_heatmap,
    create_violin_plot,
)

CONFIG_SCHEMA = {
    "model_config": {
        "image_width": (int, "Width of input images in pixels"),
        "image_height": (int, "Height of input images in pixels"),
        "color_mode": (ColorMode, "Color mode for image processing (L=grayscale, RGB=color)"),
        "single_channel": (bool, "Use single channel for model input"),
        "inverted_classes": (bool, "Invert class labels in model output"),
    },
    "explainer_config": {
        "max_faces": (int, "Maximum number of faces to detect per image"),
        "cam_method": (CAMMethod, "Class activation mapping method"),
        "cutoff_percentile": (int, "Percentile threshold for activation map"),
        "threshold_method": (ThresholdMethod, "Method for thresholding activation maps"),
        "overlap_threshold": (float, "Minimum overlap ratio for feature matching"),
        "distance_metric": (DistanceMetric, "Distance metric for feature matching"),
        "activation_maps_path": (str, "Path to save activation maps"),
    },
    "calculator_config": {
        "ndigits": (int, "Number of decimal places in calculations"),
    },
    "dataset_config": {
        "max_samples": (int, "Maximum number of samples to analyze (0 for all)"),
        "shuffle": (bool, "Randomly shuffle dataset"),
        "seed": (int, "Random seed for shuffling"),
    },
}


def safe_cast(value: Any, type_hint: Any) -> Any:
    """Safely cast values to their target types with proper error handling."""
    try:
        if type_hint is bool:
            return bool(value)
        if type_hint is int:
            return int(float(value))
        if type_hint is float:
            return float(value)
        if type_hint is str:
            return str(value)
        if hasattr(type_hint, "__args__"):
            if value not in type_hint.__args__:
                raise ValueError(f"Value {value} not in allowed values: {type_hint.__args__}")
            return value
        return type_hint(value)
    except (ValueError, TypeError) as e:
        raise gr.Error(f"Invalid value {value} for type {type_hint}: {e}")


def create_component(type_hint: Any, label: str, value: Any = None) -> gr.Component:
    """Create appropriate Gradio component based on parameter type."""
    if type_hint is bool:
        return gr.Checkbox(label=label, value=bool(value))
    if type_hint is int:
        return gr.Number(label=label, value=value, precision=0)
    if type_hint is float:
        return gr.Number(label=label, value=value)
    if hasattr(type_hint, "__args__"):
        return gr.Dropdown(label=label, choices=list(type_hint.__args__), value=value)
    return gr.Textbox(label=label, value=str(value) if value is not None else None)


def validate_model(file: gr.File) -> str:
    """Validate uploaded model file and save to temporary location."""
    if not file:
        raise gr.Error("Model file required")

    temp_path = Path(mktemp(suffix=".h5"))
    try:
        temp_path.write_bytes(Path(file).read_bytes())
        with h5py.File(temp_path):
            return str(temp_path)
    except Exception as e:
        temp_path.unlink()
        raise gr.Error(f"Invalid model file: {e}")


def create_analysis_function(
    component_map: dict[int, tuple[str, str, Any]],
    filter_map: dict[str, dict[str, gr.Component]],
) -> Callable[..., list[go.Figure]]:
    """Create analysis function with proper configuration mapping."""

    for section, components in filter_map.items():
        for key, component in components.items():
            filter_map[section][key] = component.value

    def analyze(*values: list[Any]) -> list[go.Figure]:
        model_path = validate_model(values[0])
        dataset_path = values[1]

        base = create_default_config(model_path, dataset_path)
        for idx, (section, key, type_hint) in component_map.items():
            if idx < 2:  # Skip model and dataset path
                continue
            base[section][key] = safe_cast(values[idx], type_hint)

        config = Config.create(base)
        results = BiasAnalyzer(config).analyze()

        return [
            create_radar_chart(results.feature_scores),
            create_parallel_coordinates(results.feature_probabilities),
            create_confusion_matrix(results.explanations),
            create_roc_curves(results.explanations),
            create_violin_plot(results.explanations),
            create_spatial_heatmap(results.explanations, **filter_map["spatial_heatmap"]),
        ]

    return analyze
