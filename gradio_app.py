from pathlib import Path
from tempfile import NamedTemporaryFile, mktemp
from typing import Any

import gradio as gr
import h5py
import json
import numpy as np
from typing_extensions import TypeAlias

from biasx import BiasAnalyzer
from biasx.config import Config
from biasx.datasets import AnalysisDataset
from biasx.defaults import create_default_config

ConfigDict: TypeAlias = dict[str, Any]

def validate_model(file: gr.File) -> str:
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
    base = create_default_config(model_path, dataset_path)
    for k, v in cfg.items():
        for section in base.keys():
            if k in base[section]:
                base[section][k] = v
                break
    return Config(**base)

def format_output(dataset: AnalysisDataset) -> tuple[list[dict], np.ndarray, dict, str]:
    features = [{
        "Feature": f,
        "Male Bias": f"{p[0]:.3f}",
        "Female Bias": f"{p[1]:.3f}",
        "Bias Score": f"{dataset.feature_scores[f]:.3f}"
    } for f, p in dataset.feature_probabilities.items()]

    maps = [e.activation_map_path for e in dataset.explanations[:5] if e.activation_map_path]
    results = json.loads(json.dumps(dataset.to_dict(), default=str))

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f, indent=2)
        download = f.name

    return features, maps, results, download

def analyze(*args: Any, **kwargs: Any) -> tuple[list[dict], np.ndarray, dict, str]:
    if args:
        params = dict(zip(components.keys(), args))
    else:
        params = kwargs.copy()

    model_path = validate_model(params.pop("model_file"))
    config = create_config(model_path, params.pop("dataset_path"), **params)

    return format_output(BiasAnalyzer(config).analyze())

def create_component(key: str, type_hint: Any, default: Any = None) -> gr.Component:
    if isinstance(type_hint, bool) or type_hint is bool:
        return gr.Checkbox(label=key, value=default)
    if isinstance(type_hint, (int, float)) or type_hint in (int, float):
        precision = 0 if type_hint is int else None
        return gr.Number(label=key, value=default, precision=precision)
    if hasattr(type_hint, "__args__"):
        return gr.Dropdown(label=key, choices=list(type_hint.__args__), value=default)
    return gr.Textbox(label=key, value=default)

components: dict[str, gr.Component] = {}

def create_interface() -> gr.Blocks:
    defaults = create_default_config("", "")

    with gr.Blocks(title="BiasX Analyzer") as demo:
        gr.Markdown("# BiasX: Face Classification Bias Analysis")

        with gr.Row():
            with gr.Column(scale=2):
                components["model_file"] = gr.File(label="Upload Model", file_types=[".h5"])
                components["dataset_path"] = gr.Textbox(label="Dataset Path")

                for section in ("model_config", "explainer_config", "calculator_config", "dataset_config"):
                    for k, v in defaults[section].items():
                        components[k] = create_component(k, type(v), v)

                analyze_btn = gr.Button("Run Analysis", variant="primary")

            with gr.Column(scale=3):
                results = [
                    gr.Dataframe(headers=["Feature", "Male Bias", "Female Bias", "Bias Score"]),
                    gr.Gallery(label="Sample Maps", columns=5, height=200),
                    gr.JSON(label="Results"),
                    gr.File(label="Download", file_types=[".json"], interactive=False)
                ]

        analyze_btn.click(fn=analyze, inputs=list(components.values()), outputs=results)

    return demo

if __name__ == "__main__":
    create_interface().launch()