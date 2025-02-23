import json
from pathlib import Path
from tempfile import NamedTemporaryFile, mktemp
from typing import Any

import gradio as gr
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from PIL import Image

from biasx import BiasAnalyzer
from biasx.config import Config
from biasx.datasets import AnalysisDataset
from biasx.defaults import create_default_config
from biasx.types import Explanation

COMPONENTS: dict[str, gr.Component] = {}


def create_component(key: str, type_hint: Any, default: Any = None) -> gr.Component:
    """Create appropriate Gradio component based on parameter type."""
    if isinstance(type_hint, bool) or type_hint is bool:
        return gr.Checkbox(label=key, value=default)
    if isinstance(type_hint, (int, float)) or type_hint in (int, float):
        precision = 0 if type_hint is int else None
        return gr.Number(label=key, value=default, precision=precision)
    if hasattr(type_hint, "__args__"):
        return gr.Dropdown(label=key, choices=list(type_hint.__args__), value=default)
    return gr.Textbox(label=key, value=default)


def validate_model(file: gr.File) -> str:
    """Validate uploaded model file."""
    if not file:
        raise gr.Error("Model file required")
    temp = Path(mktemp(suffix=".h5"))
    temp.write_bytes(Path(file).read_bytes())
    try:
        with h5py.File(temp):
            return str(temp)
    except:
        temp.unlink()
        raise gr.Error("Invalid model file")


def create_config(model_path: str, dataset_path: str, **cfg: Any) -> Config:
    """Create configuration from inputs."""
    base = create_default_config(model_path, dataset_path)
    {base[sec].update({k: v}) for k, v in cfg.items() for sec in base if k in base[sec]}
    return Config(**base)


def create_feature_bias_plot(features_data: list[list[Any]]) -> go.Figure:
    """Create horizontal bar chart showing bias scores by facial feature."""
    df = pd.DataFrame(features_data, columns=["Feature", "Male_Prob", "Female_Prob", "Bias_Score"])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["Feature"], x=df["Bias_Score"], orientation="h", marker_color=df["Bias_Score"]))
    fig.update_layout(title="Feature Bias Scores", xaxis_title="Bias Score", yaxis_title="Facial Feature", height=400)
    return fig


def create_parallel_coordinates(features_data: list[list[Any]]) -> go.Figure:
    """Create parallel coordinates plot showing relationships between probabilities and bias."""
    df = pd.DataFrame(features_data, columns=["Feature", "Male_Prob", "Female_Prob", "Bias_Score"])
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=df["Bias_Score"]),
            dimensions=[
                dict(range=[0, 1], label="Male Probability", values=df["Male_Prob"]),
                dict(range=[0, 1], label="Female Probability", values=df["Female_Prob"]),
                dict(range=[0, 1], label="Bias Score", values=df["Bias_Score"]),
            ],
        )
    )
    fig.update_layout(title="Feature Bias Analysis Parallel Coordinates", height=400)
    return fig


def create_confusion_matrix(results: list[Explanation]) -> go.Figure:
    """Create confusion matrix heatmap."""
    matrix = np.zeros((2, 2))
    for r in results:
        matrix[r.true_gender][r.predicted_gender] += 1

    fig = ff.create_annotated_heatmap(matrix, x=["Predicted Male", "Predicted Female"], y=["True Male", "True Female"])
    fig.update_layout(title="Gender Classification Confusion Matrix", height=400)
    return fig


def create_confidence_violin(results: list[Explanation]) -> go.Figure:
    """Create violin plot of prediction confidence distributions."""
    df = pd.DataFrame([{"Confidence": r.prediction_confidence, "Type": "Correct" if r.predicted_gender == r.true_gender else "Incorrect"} for r in results])
    fig = go.Figure()
    for type_ in ["Correct", "Incorrect"]:
        subset = df[df["Type"] == type_]
        fig.add_trace(go.Violin(x=subset["Type"], y=subset["Confidence"], name=type_, box_visible=True, meanline_visible=True))
    fig.update_layout(title="Prediction Confidence Distribution", xaxis_title="Prediction Type", yaxis_title="Confidence Score", height=400)
    return fig


def overlay_heatmap(image_path: str, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay heatmap on original image."""
    img = np.array(Image.open(image_path))
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(img.shape[:2][::-1], Image.BILINEAR))
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    heatmap_colored = plt.cm.jet(heatmap_norm)[..., :3]
    overlay = (1 - alpha) * img + alpha * heatmap_colored * 255
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def format_output(dataset: AnalysisDataset) -> list:
    """Format analysis results for display."""
    features = [[f, p[0], p[1], dataset.feature_scores[f]] for f, p in dataset.feature_probabilities.items()]

    samples = []
    for exp in dataset.explanations[:5]:
        heatmap = AnalysisDataset.load_activation_map(exp.activation_map_path)
        samples.append(overlay_heatmap(exp.image_path, heatmap))

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset.to_dict(), f, indent=2)
        download_path = f.name

    return [
        create_feature_bias_plot(features),
        create_parallel_coordinates(features),
        create_confusion_matrix(dataset.explanations),
        create_confidence_violin(dataset.explanations),
        samples,
        download_path,
    ]


def analyze(*args: Any, **kwargs: Any) -> tuple[dict, list[dict], str]:
    """Run bias analysis and format results."""
    if args:
        params = dict(zip(COMPONENTS.keys(), args))
    else:
        params = kwargs.copy()

    model_path = validate_model(params.pop("model_file"))
    config = create_config(model_path, params.pop("dataset_path"), **params)

    return format_output(BiasAnalyzer(config).analyze())


def create_interface() -> gr.Blocks:
    """Create enhanced Gradio interface."""
    defaults = create_default_config("", "")

    with gr.Blocks(title="BiasX Analyzer") as demo:
        gr.Markdown("# BiasX Analyzer")

        with gr.Row():
            with gr.Column(scale=2):
                COMPONENTS["model_file"] = gr.File(label="Upload Model", file_types=[".h5"])
                COMPONENTS["dataset_path"] = gr.Textbox(label="Dataset Path")

                with gr.Tabs():
                    for section in ("model_config", "explainer_config", "calculator_config", "dataset_config"):
                        with gr.Tab(section.split("_")[0].capitalize()):
                            for k, v in defaults[section].items():
                                COMPONENTS[k] = create_component(k, type(v), v)

                analyze_btn = gr.Button("Run Analysis", variant="primary")

            outputs = []
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Overview"):
                        outputs.append(gr.Plot(label="Feature Bias Scores"))
                        outputs.append(gr.Plot(label="Confusion Matrix"))

                    with gr.Tab("Detailed Analysis"):
                        outputs.append(gr.Plot(label="Parallel Coordinates"))
                        outputs.append(gr.Plot(label="Confidence Distribution"))

                    with gr.Tab("Sample Analysis"):
                        outputs.append(gr.Gallery(label="Sample Results", columns=5))

                    with gr.Tab("Raw Data"):
                        outputs.append(gr.File(label="Download Complete Results", file_types=[".json"]))

        analyze_btn.click(fn=analyze, inputs=list(COMPONENTS.values()), outputs=outputs)

    return demo


if __name__ == "__main__":
    create_interface().launch()
