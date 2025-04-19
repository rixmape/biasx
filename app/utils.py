import json
import os
import tempfile

import streamlit as st
from huggingface_hub import hf_hub_download


def create_temp_file(uploaded_file):
    """Create a temporary file from an uploaded file."""
    if uploaded_file is None:
        return None

    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


def retrieve_model_options(repo_id: str = "4shL3I/biasx-models", model_set: str = "models"):
    """Retrieve model options from a given repository ID."""
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=f"{model_set}.json")
        with open(model_path, "r") as json_file:
            model_options = json.load(json_file)
        return model_options
    except Exception as e:
        st.error(f"Error fetching model list: {e}")
        return None


@st.cache_data
def load_hf_model(repo_id: str = "4shL3I/biasx-models", model_filename: str = None):
    """Load a model from the Hugging Face Hub."""
    try:
        return hf_hub_download(repo_id=repo_id, filename=model_filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_default_config():
    """Return the default configuration for the application."""
    return {
        "model": {
            "path": None,
            "inverted_classes": False,
            "batch_size": 32,
        },
        "explainer": {
            "landmarker_source": "mediapipe",
            "cam_method": "gradcam++",
            "cutoff_percentile": 90,
            "threshold_method": "otsu",
            "overlap_threshold": 0.2,
            "distance_metric": "euclidean",
            "batch_size": 32,
        },
        "dataset": {
            "source": "utkface",
            "image_width": 48,
            "image_height": 48,
            "color_mode": "L",
            "single_channel": False,
            "max_samples": 100,
            "shuffle": True,
            "seed": 69,
            "batch_size": 32,
        },
    }


def filter_samples(samples, sample_index, gender_filter, classification, facial_feature):
    """Filter samples based on user-selected criteria."""
    filtered_samples = [sample for sample in samples[:sample_index] if (gender_filter is None or sample.image_data.gender.name == gender_filter.upper()) and (classification is None or ((classification == "Incorrect" and sample.image_data.gender.numerator != sample.predicted_gender.numerator) or (classification == "Correct" and sample.image_data.gender.numerator == sample.predicted_gender.numerator))) and (facial_feature is None or not facial_feature or all(any(a.feature is not None and a.feature.value == feature for a in sample.activation_boxes) for feature in facial_feature))]
    return filtered_samples


def initialize_session_state():
    """Initialize session state variables if they don't exist."""

    if "layout" not in st.session_state:
        st.session_state.layout = "centered"

    if "model_path" not in st.session_state:
        st.session_state.model_path = ""

    if "enable_shuffle" not in st.session_state:
        st.session_state.enable_shuffle = True

    if "select_all" not in st.session_state:
        st.session_state.select_all = False

    if "invert_label" not in st.session_state:
        st.session_state.invert_label = False

    if "configuration" not in st.session_state:
        st.session_state.configuration = True

    if "page" not in st.session_state:
        st.session_state.page = [0, 18]

    if "result" not in st.session_state:
        st.session_state.result = None

    if "config" not in st.session_state:
        st.session_state.config = get_default_config()

    if "file_info" not in st.session_state:
        st.session_state.file_info = None

    if "model_config" not in st.session_state:
        st.session_state.model_config = {"height": 48, "width": 48, "color_mode": "Grayscale", "channel": "Single"}

    if "start_analysis" not in st.session_state:
        st.session_state.start_analysis = False

    if "show_upload_page" not in st.session_state:
        st.session_state.show_upload_page = False


def reset_config():
    """Reset all configuration settings."""
    st.session_state.config = get_default_config()
    st.session_state.model_path = ""
    st.session_state.result = None
    st.session_state.page = [0, 18]


def reset_page():
    """Reset pagination."""
    st.session_state.page[0] = 0
    st.session_state.page[1] = 18
