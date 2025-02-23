import json
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import gradio as gr
import h5py
from typing_extensions import get_args

from biasx import BiasAnalyzer
from biasx.defaults import create_default_config
from biasx.types import CAMMethod, ColorMode, DistanceMetric, ThresholdMethod, ImageSize

T = TypeVar("T")

CONFIG_SCHEMA = {
    "model_options": {
        "target_size": (ImageSize, "Target Size"),
        "color_mode": (ColorMode, "Color Mode"),
        "single_channel": (bool, "Single Channel"),
    },
    "explainer_options": {
        "max_faces": (int, "Max Faces"),
        "cam_method": (CAMMethod, "CAM Method"),
        "cutoff_percentile": (int, "Cutoff Percentile"),
        "threshold_method": (ThresholdMethod, "Threshold Method"),
        "overlap_threshold": (float, "Overlap Threshold"),
        "distance_metric": (DistanceMetric, "Distance Metric"),
    },
    "calculator_options": {
        "ndigits": (int, "Number of Digits"),
        "shuffle": (bool, "Shuffle"),
        "seed": (int, "Seed"),
        "max_samples": (int, "Max Samples"),
    },
}


def create_component(key: str, type_hint: Any, label: str, default: Any = None) -> gr.Component:
    """Create appropriate Gradio component based on type hint."""
    if type_hint == bool:
        return gr.Checkbox(label=label, value=default)
    elif type_hint == int:
        return gr.Number(label=label, value=default, precision=0)
    elif type_hint == float:
        return gr.Number(label=label, value=default)
    elif type_hint == ImageSize:
        return [
            gr.Number(label=f"{label} X", value=default[0], precision=0),
            gr.Number(label=f"{label} Y", value=default[1], precision=0),
        ]
    elif hasattr(type_hint, "__args__"):  # For Literal types (enums)
        return gr.Dropdown(label=label, choices=get_args(type_hint), value=default)
    return gr.Textbox(label=label, value=default)


def create_config_inputs(config: dict) -> dict[str, list[tuple[str, gr.Component]]]:
    """Generate Gradio components for each config section."""
    components = {}
    for section, fields in CONFIG_SCHEMA.items():
        section_components = []
        for key, (type_hint, label) in fields.items():
            comp = create_component(key, type_hint, label, config[section][key])
            if isinstance(comp, list):
                section_components.extend([(f"{key}_{i}", c) for i, c in enumerate(comp)])
            else:
                section_components.append((key, comp))
        components[section] = section_components
    return components


def process_uploaded_model(file: gr.File) -> str:
    """Validate and save uploaded model file."""
    if not file:
        raise gr.Error("Please upload a model file")

    temp_path = Path(tempfile.mktemp(suffix=".h5"))
    temp_path.write_bytes(Path(file).read_bytes())

    try:
        with h5py.File(temp_path, "r"):
            return str(temp_path)
    except:
        temp_path.unlink()
        raise gr.Error("Invalid model file format")


def serialize_results(results: dict) -> tuple[dict, str]:
    """Serialize analysis results and prepare download file."""
    for feature in results["featureProbabilities"]:
        results["featureProbabilities"][feature] = {str(k): v for k, v in results["featureProbabilities"][feature].items()}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f, indent=2)
        return results, f.name


def run_analysis(model_file: str, dataset_path: str, **config_values) -> tuple[dict, str]:
    """Run bias analysis with provided configuration."""
    config = {
        "model_path": model_file,
        "model_options": {
            "target_size": (int(config_values.pop("target_size_0")), int(config_values.pop("target_size_1"))),
            **{k: v for k, v in config_values.items() if k in CONFIG_SCHEMA["model_options"]},
        },
        "explainer_options": {k: v for k, v in config_values.items() if k in CONFIG_SCHEMA["explainer_options"]},
        "calculator_options": {k: v for k, v in config_values.items() if k in CONFIG_SCHEMA["calculator_options"]},
    }

    analyzer = BiasAnalyzer(config)
    results = (
        analyzer.analyze(
            dataset_path=dataset_path,
            max_samples=config["calculator_options"]["max_samples"],
            shuffle=config["calculator_options"]["shuffle"],
            seed=config["calculator_options"]["seed"],
        ),
    )

    return serialize_results(results.to_dict())


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    default_config = create_default_config("")
    config_inputs = create_config_inputs(default_config)

    with gr.Blocks(title="BiasX Analyzer") as demo:
        gr.Markdown("# BiasX: Face Classification Bias Analysis")

        with gr.Group():
            gr.Markdown("## Step 1: Upload Face Classification Model")
            model_file = gr.File(label="Choose a model file", file_types=[".h5"])

        with gr.Group():
            gr.Markdown("## Step 2: Configure Analysis Parameters")
            with gr.Row():
                for section, components in config_inputs.items():
                    with gr.Column():
                        gr.Markdown(f"### {section.replace('_', ' ').title()}")
                        for _, component in components:
                            component.render()

        with gr.Group():
            gr.Markdown("## Step 3: Run Analysis")
            dataset_path = gr.Textbox(label="Dataset Path", value="images/utkface")
            analyze_btn = gr.Button("Run Analysis")

        with gr.Group():
            gr.Markdown("## Step 4: View Results")
            results_json = gr.JSON(label="Analysis Results")
            download_btn = gr.File(label="Download Results", file_types=[".json"], interactive=False)

        # Flatten components for event handler
        all_inputs = [model_file, dataset_path]
        for components in config_inputs.values():
            all_inputs.extend(comp for _, comp in components)

        analyze_btn.click(
            fn=lambda *args: run_analysis(process_uploaded_model(args[0]), args[1], **{k: v for (k, _), v in zip(sum(config_inputs.values(), []), args[2:])}),
            inputs=all_inputs,
            outputs=[results_json, download_btn],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
