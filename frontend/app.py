from typing import Any

import gradio as gr

from biasx.types import FacialFeature, Gender
from utils import CONFIG_SCHEMA, create_analysis_function, create_component, create_default_config


def create_interface() -> gr.Blocks:
    """Create enhanced Gradio interface with schema-driven components."""
    defaults = create_default_config("", "")

    input_component_map: dict[int, tuple[str, str, Any]] = {}
    components: list[gr.Component] = []

    with gr.Blocks(title="BiasX Analyzer") as demo:
        gr.Markdown("# BiasX: Face Classification Bias Analysis")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Inputs")
                components.extend(
                    [
                        gr.File("tmp/identiface.h5", label="Upload Model", file_types=[".h5"]),
                        gr.Textbox("images/utkface", label="Dataset Path"),
                    ]
                )

                gr.Markdown("## Configuration")
                with gr.Tabs():
                    idx = len(components)
                    for section, params in CONFIG_SCHEMA.items():
                        with gr.Tab(section.split("_")[0].capitalize()):
                            for key, (type_hint, help_text) in params.items():
                                component = create_component(
                                    type_hint=type_hint,
                                    label=help_text,
                                    value=defaults[section][key],
                                )
                                components.append(component)
                                input_component_map[idx] = (section, key, type_hint)
                                idx += 1

                analyze_btn = gr.Button("Run Analysis", variant="primary")

            with gr.Column(scale=4):
                gr.Markdown("## Results")
                outputs = []
                with gr.Row():
                    outputs.extend(
                        [
                            gr.Plot(label="Feature Bias Scores", scale=1),
                            gr.Plot(label="Feature Activation Patterns by Gender", scale=2),
                        ]
                    )
                with gr.Row():
                    outputs.extend(
                        [
                            gr.Plot(label="Confusion Matrix", scale=1),
                            gr.Plot(label="ROC Curves by Gender", scale=1),
                            gr.Plot(label="Confidence Score Distribution", scale=1),
                        ]
                    )

                with gr.Row():
                    features = list(FacialFeature.__args__)
                    with gr.Column():
                        filter_map: dict[str, gr.Component] = {
                            "features": gr.CheckboxGroup(choices=features, label="Facial Features Filter"),
                            "misclassified_only": gr.Checkbox(label="Show Misclassified Only"),
                        }
                    with gr.Row(scale=2):
                        outputs.extend(
                            [
                                gr.Plot(label="Activation Frequency for Male", scale=1),
                                gr.Plot(label="Activation Frequency for Female", scale=1),
                            ]
                        )

        analyze_fn = create_analysis_function(input_component_map, filter_map)
        analyze_btn.click(fn=analyze_fn, inputs=components, outputs=outputs)

    return demo


if __name__ == "__main__":
    create_interface().launch()
