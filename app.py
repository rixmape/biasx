from pathlib import Path
from tempfile import mktemp
from typing import Any

import gradio as gr
import h5py
import plotly.graph_objects as go

from biasx import BiasAnalyzer
from biasx.config import Config
from biasx.defaults import create_default_config
from frontend.graphs import create_confusion_matrix, create_parallel_coordinates, create_radar_chart, create_roc_curves, create_violin_plot


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


def create_config(model_path: str, dataset_path: str, *component_values: list[str]) -> Config:
    """Create configuration from inputs."""
    base = create_default_config(model_path, dataset_path)
    for section, values in zip(("model_config", "explainer_config", "calculator_config", "dataset_config"), component_values):
        for key, value in zip(base[section].keys(), values):
            base[section][key] = value
    return Config(**base)


def analyze(*components: list[gr.Component]) -> list[go.Figure]:
    """Run bias analysis and format results."""
    model_file, dataset_path = components[:2]
    model_path = validate_model(model_file)
    config = create_config(model_path, dataset_path, components[2:])

    dataset = BiasAnalyzer(config).analyze()

    return [
        create_radar_chart(dataset.feature_scores),
        create_parallel_coordinates(dataset.feature_probabilities),
        create_confusion_matrix(dataset.explanations),
        create_roc_curves(dataset.explanations),
        create_violin_plot(dataset.explanations),
    ]


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
            inputs = []
            with gr.Column(scale=1):
                gr.Markdown("## Inputs")
                inputs.append(gr.File("tmp/identiface.h5", label="Upload Model", file_types=[".h5"]))
                inputs.append(gr.Textbox("images/utkface", label="Dataset Path"))

                gr.Markdown("## Configuration")
                with gr.Tabs():
                    for section in ("model_config", "explainer_config", "calculator_config", "dataset_config"):
                        with gr.Tab(section.split("_")[0].capitalize()):
                            for k, v in defaults[section].items():
                                inputs.append(create_component(k, type(v), v))

                analyze_btn = gr.Button("Run Analysis", variant="primary")

            outputs = []
            with gr.Column(scale=4):
                gr.Markdown("## Results")
                with gr.Row():
                    outputs.append(gr.Plot(label="Feature Bias Scores", scale=1))
                    outputs.append(gr.Plot(label="Feature Activation Patterns by Gender", scale=2))
                with gr.Row():
                    outputs.append(gr.Plot(label="Confusion Matrix", scale=1))
                    outputs.append(gr.Plot(label="ROC Curves by Gender", scale=1))
                    outputs.append(gr.Plot(label="Confidence Score Distribution", scale=1))

        analyze_btn.click(fn=analyze, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    create_interface().launch()
