import streamlit as st
import tempfile
import pandas as pd
import numpy as np
import pandas as pd
from biasx import BiasAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from biasx.config import Config


if "layout" not in st.session_state:
    st.session_state.layout = "centered"

st.set_page_config(layout=st.session_state.layout)

if "model_path" not in st.session_state:
    st.session_state.model_path = ""

if "select_all" not in st.session_state:
    st.session_state.select_all = False

if "enable_shuffle" not in st.session_state:
    st.session_state.enable_shuffle = True

if "select_all" not in st.session_state:
    st.session_state.select_all = True

if "invert_label" not in st.session_state:
    st.session_state.invert_label = False

if "configuration" not in st.session_state:
    st.session_state.configuration = True

if 'page' not in st.session_state:
    st.session_state.page = [0, 9]

if "result" not in st.session_state:
    st.session_state.result = None

# Default configuration values
if 'config' not in st.session_state:
    st.session_state.config = {
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

@st.cache_data
def save_uploaded_model(uploaded_file):
    """Save uploaded model to temporary file and return path."""
    if uploaded_file is None:
        return None
        
    with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name
    
def biasx(config):
    analyzer = BiasAnalyzer(config)
    analysis = analyzer.analyze()

    # Get Feature Analysis
    feature_analysis_dict = {
        feature.name: {
            "bias_score": analysis.bias_score,
            "male_probability": analysis.male_probability,
            "female_probability": analysis.female_probability
        }
        for feature, analysis in analysis.feature_analyses.items()
    }

    # Mock disparity scores
    disparity_scores = {
        'biasx': analysis.disparity_scores.biasx,
        'equalized_odds': analysis.disparity_scores.equalized_odds
    }

    data_dict = {}
    for i, explanation in enumerate(analysis.explanations):
        true_gender = explanation.image_data.gender.name.lower()  # Convert Enum to string
        predicted_gender = explanation.predicted_gender.name.lower()  # Convert Enum to string

        data_dict[i] = {
            "true_gender": true_gender,
            "predicted_gender": predicted_gender
        }

    # Define the categories
    labels = ["male", "female"]

    # Count occurrences using tuple keys (true_gender, predicted_gender)
    conf_counts = Counter((entry["true_gender"], entry["predicted_gender"]) for entry in data_dict.values())

    # Construct the confusion matrix using label indexing
    confusion_matrix = [
        [conf_counts[("male", "male")], conf_counts[("male", "female")]],
        [conf_counts[("female", "male")], conf_counts[("female", "female")]]
    ]
    
    return {
        'feature_analyses': feature_analysis_dict,
        'disparity_score': disparity_scores,
        'confusion_matrix': confusion_matrix
    }

def create_confusion_matrix(confusion_matrix_data):
    """Create confusion matrix visualization with better size."""
    labels = ['Male', 'Female']
    fig = px.imshow(confusion_matrix_data, 
                    x=labels, y=labels, 
                    color_continuous_scale='blues',
                    labels=dict(x="Predicted", y="True", color="Count"),
                    title="Confusion Matrix",
                    text_auto=True)

    # Increase figure size and adjust layout
    fig.update_layout(
        autosize=False,
        width=600,  # Increase width
        height=500,  # Increase height
        margin=dict(l=50, r=50, t=50, b=50),  # Reduce extra space
        font=dict(size=14)  # Increase font size for readability
    )
    
    return fig

def create_radar_chart(feature_analyses):
    """Create a zoomed-in radar chart for feature bias scores."""
    categories = list(feature_analyses.keys())
    values = [feature_analyses[feature]['bias_score'] for feature in categories]

    max_value = max(values)
    zoom_range = [0, max_value * 1.2]  # Adds 20% margin for better visualization

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Closing the loop
        theta=categories + [categories[0]],  # Closing the loop
        fill='toself',
        name='Bias Score',
        line=dict(color='blue', width=2),
        marker=dict(size=6, symbol='circle')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=zoom_range)
        ),
        showlegend=False,
    )

    return fig

def create_feature_probability_chart(feature_analyses):
    """Create grouped bar chart of feature probabilities by gender."""
    data = []
    for feature in feature_analyses:
        data.append({
            'Feature': feature,
            'Male Probability': feature_analyses[feature]['male_probability'],
            'Female Probability': feature_analyses[feature]['female_probability'],
        })
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x='Feature', y=['Male Probability', 'Female Probability'], 
                 barmode='group', title='Feature Activation Probability by Gender')
    
    return fig

def display_configuration_page():
    
    # --- Model Configuration ---
    with st.container(border=True):
        st.markdown("### Model Configuration")
        model_upload, model_config = st.columns(2)
        with model_upload:
            uploaded_file = model_upload.file_uploader("Upload", type=["h5", "keras"], key="uploaded_model", label_visibility="collapsed")
            
            if uploaded_file is not None:
            # Create a temporary file to store the uploaded model
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
                    temp_file.write(uploaded_file.read())  # Save the uploaded file
                    st.session_state.config["model"]["path"]= temp_file.name  # Get the path of the temp file
            
            # if uploaded_file:
            #     model_path = save_uploaded_model(uploaded_file)
            #     st.session_state.config["model"]["path"] = model_path
            #     st.success("Model uploaded successfully!")

        with model_config.container(border=True):
            c1, c2 = st.columns(2)
            st.session_state.config["dataset"]["image_width"] = c1.number_input("Image Width", key="input_image_width", value=st.session_state.config["dataset"]["image_width"])
            st.session_state.config["dataset"]["image_height"] = c2.number_input("Image Height", key="input_image_height", value=st.session_state.config["dataset"]["image_height"])
            st.session_state.config["dataset"]["color_mode"] = c1.pills("Color Mode", ["Grayscale", "RGB"], key="color_mode", selection_mode="single")
            if st.session_state.config["dataset"]["color_mode"] == "Grayscale":st.session_state.config["dataset"]["color_mode"] = "L"
            channel = c2.pills("Channel", ["Single", "Triple"], key="channel", selection_mode="single")
            st.session_state.config["dataset"]["single_channel"] = channel == "Single"
            st.session_state.config["model"]["inverted_classes"] = st.toggle("Invert Label?", value=st.session_state.invert_label)

    # --- Other Configuration ---
    with st.container():
        explainer, other = st.columns(2)

        with explainer.container(border=True):
            st.markdown("### Explainer Configuration")

            # --- Class Activation Config ---
            with st.container(border=True):
                st.session_state.config["explainer"]["cam_method"] = st.pills("Class Activation Map Method", ["gradcam", "gradcam++", "scorecam"], key="cam_method", selection_mode="single")
                st.session_state.config["explainer"]["cutoff_percentile"] = st.slider("Cutoff Percentile", key="cam_threshold", value=st.session_state.config["explainer"]["cutoff_percentile"])

            # --- Threshold Config ---
            with st.container(border=True):
                st.session_state.config["explainer"]["threshold_method"] = st.pills("Thresholding Method", ["otsu", "niblack", "sauvola"], key="threshold_method", selection_mode="single")
                st.session_state.config["explainer"]["overlap_threshold"]  = st.slider("Overlap Threshold", key="overlap_ratio", value=st.session_state.config["explainer"]["overlap_threshold"])

            # --- Distant Metric Config ---
            with st.container(border=True):
                st.session_state.config["explainer"]["distance_metric"] = st.pills("Distant Metric", ["euclidean", "manhattan"], key="distant_metric", selection_mode="single")

        with other.container():

            # --- Dataset Config ---
            with st.container(border=True):
                st.markdown("### Dataset Configuration")
                st.session_state.config["dataset"]["source"] = st.pills("Dataset Selection", ["utkface", "fairface"], key="dataset_selection", selection_mode="single")
                c1, c2 = st.columns(2)
                st.session_state.config["dataset"]["max_samples"] = c1.number_input("Sample Size", key="sample_size", value=st.session_state.config["dataset"]["max_samples"], disabled=st.session_state.select_all)
                select_all = c1.toggle("Select All", key = "select_all")
                st.session_state.config["dataset"]["seed"] = c2.number_input("Seed", key="seed", value=st.session_state.config["dataset"]["seed"], disabled = not st.session_state.enable_shuffle)
                st.session_state.config["dataset"]["shuffle"] = c2.toggle("Shuffle?", key="enable_shuffle")

            if st.button("Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing", show_time=True):
                    st.session_state.result = biasx(st.session_state.config)
                
                st.session_state.layout = "wide"
                st.session_state.configuration = False
                st.rerun()

def display_visualization_page():
    tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Model Performance", "Analysis"])

    # Feature Analysis Tab
    with tab1.container(border=True):
        st.markdown("### Feature Analysis")
        c1, c2 = st.columns([1,2])
        with c1.container(border=False):
            with st.container(border=True):
                radar_chart = create_radar_chart(st.session_state.result['feature_analyses'])
                st.plotly_chart(radar_chart, use_container_width=True)
                st.markdown("**Interpretation:** Features closer to the outer edge of the radar chart have higher bias scores, indicating they more strongly influence gender misclassifications.")

        with c2.container(border=False):
            with st.container(border=True):
                probability_chart = create_feature_probability_chart(st.session_state.result['feature_analyses'])
                st.plotly_chart(probability_chart, use_container_width=True, use_containner_height=True)
                st.markdown("**Interpretation:** Bars show how often each feature is activated during misclassifications. Large differences between male and female probabilities indicate potential bias.")

    # Model Performance
    with tab2.container(border=True):
        st.markdown("### Model Performance")
        c1, c2 = st.columns([1,2])
        with c1.container(border=False):
            with st.container(border=True):
                confusion_matrix = create_confusion_matrix(st.session_state.result['confusion_matrix'])
                st.plotly_chart(confusion_matrix, use_container_width=True)
                
        with c2.container(border=False):
            placeholder()

        c1, c2 = st.columns(2)
        with c1.container(border=False):
            placeholder()

        with c2.container(border=False):
            placeholder()

    # Image Analysis Tab
    with tab3.container(border=True):
        c1, c2 = st.columns([1,2])

        # Filter for image overlay and other settings
        with c1.container(border=True):
            st.markdown("### Filters")

        # show images
        with c2.container(height=780):
            st.markdown("### Sample Images")
            image_generator()

    if st.button("Go Back", type="primary", use_container_width=True):
                st.session_state.layout = "centered"
                st.session_state.configuration = True
                st.rerun()

# Sample Image Viewer Tab
def image_generator():
    start = st.session_state.page[0]
    end = st.session_state.page[1]

    images = [f"image{i}" for i in range(1, 31)]

    c1, c2, c3 = st.columns(3)
    for i, image in enumerate(images[start:end]):
        col = i % 3
        match col:
            case 0:
                with c1:
                    with st.container(height=200, border=True):
                        st.markdown(image)
            case 1:
                with c2:
                    with st.container(height=200, border=True):
                        st.markdown(image)
            case 2:
                with c3:
                    with st.container(height=200, border=True):
                        st.markdown(image)


    p1, p2 = st.columns(2)
    with p1:
        if st.session_state.page[0] == 0:
            st.button("---", disabled=True, use_container_width=True)
        else:
            if st.button("Back", use_container_width=True):
                st.session_state.page[1] = st.session_state.page[0]
                st.session_state.page[0] = st.session_state.page[0] - 9
                st.rerun()
    with p2:
        if st.session_state.page[1] >= len(images):
            st.button("---", disabled=True, use_container_width=True)
        else:
            if st.button("Next", use_container_width=True):
                st.session_state.page[0] = st.session_state.page[1]
                st.session_state.page[1] = st.session_state.page[1] + 9
                st.rerun()

# Sample Placeholder graph
def placeholder():
    with st.container(border=True):
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
        st.area_chart(chart_data)

if __name__ == "__main__":
    if st.session_state.configuration:
        display_configuration_page()
    else:
        display_visualization_page()
