import tempfile
import os
import streamlit as st

def create_temp_file(uploaded_file):
    """Create a temporary file from an uploaded file."""
    if uploaded_file is None:
        return None
        
    # Determine the correct suffix
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
    
    # Create a temporary file with the correct extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

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
            "image_width": 224,
            "image_height": 224,
            "color_mode": "L",
            "single_channel": False,
            "max_samples": 100,
            "shuffle": True,
            "seed": 69,
            "batch_size": 32,
        }
    }

def filter_samples(samples, sample_index, gender_filter, classification, facial_feature):
    """Filter samples based on user-selected criteria."""
    filtered_samples = [
        sample for sample in samples[:sample_index]
        if (gender_filter is None or sample.image_data.gender.name == gender_filter.upper()) and
        (classification is None or (
            (classification == "Incorrect" and sample.image_data.gender.numerator != sample.predicted_gender.numerator) or
            (classification == "Correct" and sample.image_data.gender.numerator == sample.predicted_gender.numerator)
        )) and 
        (facial_feature is None or not facial_feature or all(
            any(a.feature is not None and a.feature.value == feature for a in sample.activation_boxes)
            for feature in facial_feature
        ))
    ]
    return filtered_samples

def get_landmark_names_and_colors():
    """Return landmark names and their corresponding colors."""
    landmark_names = ["Left Eye", "Right Eye", "Nose", "Lips", "Left Cheek", 
                      "Right Cheek", "Chin", "Forehead", "Left Eyebrow", "Right Eyebrow"]
    colors = ["#6A5ACD", "#27AE60", "#3498DB", "#1ABC9C", "#8E44AD", 
              "#F39C12", "#16A085", "#F1C40F", "#5D6D7E", "#2980B9"]
    return landmark_names, colors

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
    
    if 'page' not in st.session_state:
        st.session_state.page = [0, 18]
    
    if "result" not in st.session_state:
        st.session_state.result = None
    
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()

def reset_page():
    """Reset pagination."""
    st.session_state.page[0] = 0
    st.session_state.page[1] = 18