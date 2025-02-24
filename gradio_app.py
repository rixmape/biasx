from pathlib import Path
from tempfile import mktemp
from typing import Any

import gradio as gr
import h5py
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, roc_curve

from biasx import BiasAnalyzer
from biasx.config import Config
from biasx.datasets import AnalysisDataset
from biasx.defaults import create_default_config
from biasx.types import Explanation

COMPONENTS: dict[str, gr.Component] = {}


def create_radar_chart(feature_scores: dict[str, float]) -> go.Figure:
    """Creates a radar chart comparing bias scores across facial features."""
    features = list(feature_scores.keys())
    scores = list(feature_scores.values())

    fig = go.Figure(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=features + [features[0]],
            fill="toself",
            fillcolor="blue",
            line=dict(color="blue"),
        )
    )

    fig.update_layout(polar=dict(radialaxis=dict(range=[0, max(scores) + 0.1], tickformat=".2f")))

    return fig


def create_parallel_coordinates(feature_probabilities: dict[str, dict[int, float]]) -> go.Figure:
    """Creates a parallel coordinates plot showing gender and feature activation relationships."""
    dimensions = [
        dict(
            range=[0, 1],
            label=feature.replace("_", " ").title(),
            values=[probs[0], probs[1]],
            tickformat=".2f",
        )
        for feature, probs in feature_probabilities.items()
    ]

    fig = go.Figure()
    fig.add_trace(go.Parcoords(line=dict(color=[0, 1], colorscale=[[0, "blue"], [1, "red"]]), dimensions=dimensions))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Male", line=dict(color="blue"), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Female", line=dict(color="red"), showlegend=True))

    fig.update_layout(showlegend=True, legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05))

    return fig


def create_confusion_matrix(explanations: list[Explanation]) -> go.Figure:
    """Creates a confusion matrix visualization for gender predictions."""
    y_true = [exp.true_gender for exp in explanations]
    y_pred = [exp.predicted_gender for exp in explanations]
    cm = confusion_matrix(y_true, y_pred, normalize="pred")

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted Male", "Predicted Female"],
            y=["True Male", "True Female"],
            text=[[f"{val:.2%}" for val in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            zmax=1,
            showscale=False,
        )
    )

    fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", yaxis=dict(tickangle=-90))

    return fig


def create_roc_curves(explanations: list[Explanation]) -> go.Figure:
    """Creates ROC curves for model performance analysis."""
    y_true = np.array([exp.true_gender for exp in explanations])
    y_score = np.array([exp.prediction_confidence for exp in explanations])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fpr_inv, tpr_inv, _ = roc_curve(1 - y_true, 1 - y_score)
    roc_auc_inv = auc(fpr_inv, tpr_inv)

    fig = go.Figure(
        [
            go.Scatter(x=fpr, y=tpr, name=f"Female (AUC = {roc_auc:.3f})", line=dict(color="red", width=2)),
            go.Scatter(x=fpr_inv, y=tpr_inv, name=f"Male (AUC = {roc_auc_inv:.3f})", line=dict(color="blue", width=2)),
            go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", width=2), showlegend=False),
        ]
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.02],
        legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05),
    )

    return fig


def create_violin_plot(explanations: list[Explanation]) -> go.Figure:
    """Creates a violin plot showing confidence score distributions across prediction outcomes."""
    correct = {"male": [], "female": []}
    incorrect = {"male": [], "female": []}

    for exp in explanations:
        target = correct if exp.predicted_gender == exp.true_gender else incorrect
        gender = "male" if exp.true_gender == 0 else "female"
        target[gender].append(exp.prediction_confidence)

    fig = go.Figure(
        [
            go.Violin(
                x=["Male"] * len(correct["male"]) + ["Female"] * len(correct["female"]),
                y=correct["male"] + correct["female"],
                side="positive",
                fillcolor="blue",
                name="Correct",
            ),
            go.Violin(
                x=["Male"] * len(incorrect["male"]) + ["Female"] * len(incorrect["female"]),
                y=incorrect["male"] + incorrect["female"],
                side="negative",
                fillcolor="red",
                name="Incorrect",
            ),
        ]
    )

    fig.update_layout(
        yaxis_title="Confidence Score",
        yaxis_range=[-0.02, 1.02],
        violinmode="overlay",
        legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05),
    )

    return fig


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


def format_output(dataset: AnalysisDataset) -> list:
    """Format analysis results for display."""
    return [
        create_radar_chart(dataset.feature_scores),
        create_parallel_coordinates(dataset.feature_probabilities),
        create_confusion_matrix(dataset.explanations),
        create_roc_curves(dataset.explanations),
        create_violin_plot(dataset.explanations),
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


def create_interface() -> gr.Blocks:
    """Create enhanced Gradio interface."""
    defaults = create_default_config("", "")

    with gr.Blocks(title="BiasX Analyzer") as demo:
        gr.Markdown("# BiasX: Face Classification Bias Analysis")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Inputs")
                COMPONENTS["model_file"] = gr.File("tmp/identiface.h5", label="Upload Model", file_types=[".h5"])
                COMPONENTS["dataset_path"] = gr.Textbox("images/utkface", label="Dataset Path")

                gr.Markdown("## Configuration")
                with gr.Tabs():
                    for section in ("model_config", "explainer_config", "calculator_config", "dataset_config"):
                        with gr.Tab(section.split("_")[0].capitalize()):
                            for k, v in defaults[section].items():
                                COMPONENTS[k] = create_component(k, type(v), v)

                analyze_btn = gr.Button("Run Analysis", variant="primary")

            outputs = {}
            with gr.Column(scale=4):
                gr.Markdown("## Results")
                with gr.Row():
                    outputs["radar_chart"] = gr.Plot(label="Feature Bias Scores", scale=1)
                    outputs["parallel_coordinates"] = gr.Plot(label="Feature Activation Patterns by Gender", scale=2)
                with gr.Row():
                    outputs["confusion_matrix"] = gr.Plot(label="Confusion Matrix", scale=1)
                    outputs["roc_curves"] = gr.Plot(label="ROC Curves by Gender", scale=1)
                    outputs["violin_plot"] = gr.Plot(label="Confidence Score Distribution", scale=1)

        analyze_btn.click(fn=analyze, inputs=list(COMPONENTS.values()), outputs=list(outputs.values()))

    return demo


if __name__ == "__main__":
    create_interface().launch()
