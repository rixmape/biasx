import streamlit as st
import tempfile
from biasx import BiasAnalyzer

from graphs import (
    create_radar_chart,
    create_feature_probability_chart,
    create_spatial_heatmap,
    create_classwise_performance_chart,
    create_confusion_matrix,
    create_roc_curve,
    create_precision_recall_curve,
    image_overlays
)

if "layout" not in st.session_state:
    st.session_state.layout = "centered"

st.set_page_config(layout=st.session_state.layout)

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
                    analyzer = BiasAnalyzer(st.session_state.config)
                    st.session_state.result = analyzer.analyze()
                
                st.session_state.layout = "wide"
                st.session_state.configuration = False
                st.rerun()

def display_visualization_page():
    tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Model Performance", "Analysis"])

    # Feature Analysis Tab
    with tab1.container(border=True):
        feature_analysis = {
        feature.name: {
            "bias_score": analysis.bias_score,
            "male_probability": analysis.male_probability,
            "female_probability": analysis.female_probability
        }
        for feature, analysis in st.session_state.result.feature_analyses.items()
    }

        st.markdown("### Feature Analysis")
        c1, c2 = st.columns([1,2])
        with c1.container(border=False):
            with st.container(border=False):
                metric1, metric2 = st.columns(2)
                metric1.metric(f"Bias Score", st.session_state.result.disparity_scores.biasx, border=True)
                metric2.metric(f"Equalized Odds", st.session_state.result.disparity_scores.equalized_odds, border=True)
            with st.container(border=True):
                radar_chart = create_radar_chart(feature_analysis)
                st.plotly_chart(radar_chart, use_container_width=True)
                st.markdown("Features closer to the outer edge of the radar chart have higher bias scores, indicating they more strongly influence gender misclassifications.")

        with c2.container(border=False):
            with st.container(border=True):
                probability_chart = create_feature_probability_chart(feature_analysis)
                st.plotly_chart(probability_chart, use_container_width=True, use_containner_height=True)
                st.markdown("Bars show how often each feature is activated during misclassifications. Large differences between male and female probabilities indicate potential bias.",)
            
            # with st.container(border=False):
            #     male, female = st.columns(2)
            #     with male.container(border=True):
            #         st.markdown("#### Male Spacial Heatmap ####")
            #         spacial = create_spatial_heatmap(st.session_state.result.explanations,0)
            #         st.plotly_chart(spacial, use_container_width=True)

            #     with female.container(border=True):
            #         st.markdown("#### Female Spacial Heatmap ####")
            #         spacial = create_spatial_heatmap(st.session_state.result.explanations,1)
            #         st.plotly_chart(spacial, use_container_width=True)

    # Model Performance
    with tab2.container(border=True):
        st.markdown("### Model Performance")
        c1, c2 = st.columns([1,2])
        with c1.container(border=False):
            with st.container(border=True):
                confusion_matrix = create_confusion_matrix(st.session_state.result.explanations)
                st.plotly_chart(confusion_matrix, use_container_width=True)
                st.markdown("The confusion matrix shows prediction patterns across genders. Ideally, the diagonal values should be similar, indicating balanced performance.")
                
        with c2.container(border=False):
            with st.container(border=True):
                class_wise = create_classwise_performance_chart(st.session_state.result.explanations)
                st.plotly_chart(class_wise, use_container_width=True)
                st.markdown("The class-wise metrics show the model’s precision, recall, and F1-score for each gender. Balanced scores indicate fair performance, while large gaps may suggest bias or weaknesses in classification.")


        c1, c2 = st.columns(2)
        with c1.container(border=False):
            with st.container(border=True):
                precision_recall_curve = create_precision_recall_curve(st.session_state.result.explanations)
                st.plotly_chart(precision_recall_curve, use_container_width=True)
                st.markdown("The Precision-Recall curve highlights the tradeoff between precision and recall. A higher curve suggests better performance, especially for imbalanced datasets where they are more meaningful than accuracy.")

        with c2.container(border=False):
            with st.container(border=True):
                roc_curve = create_roc_curve(st.session_state.result.explanations)
                st.plotly_chart(roc_curve, use_container_width=True)
                st.markdown("The ROC curve shows the tradeoff between true positive rate and false positive rate. A curve closer to the top-right corner indicates better performance.")

    # Image Analysis Tab
    with tab3.container(border=True):
        filter, images = st.columns([1,3])

        with filter.container():
            st.markdown("### Image Analysis")
            st.markdown("""
            This tab shows individual image analysis with activation maps.
            Activation maps highlight the regions the model focuses on when making predictions.
            """)

            def reset_page():
                st.session_state.page[0] = 0
                st.session_state.page[1] = 18

            samples = st.session_state.result.explanations
            
            max_samples = min(len(st.session_state.result.explanations), 300)
            display_count = 30
            
            with st.container(border=True):
                sample_index = st.slider("Sample Index", display_count, max_value=max_samples, value=display_count, on_change=reset_page)
                
            c1, c2 = st.columns(2)
            with c1.container(border=True):
                gender_filter = st.pills("Gender", ["Male", "Female"], key="gender_filter", selection_mode="single", on_change=reset_page)

            with c2.container(border=True):
                classification = st.pills("Classification", ["Correct", "Incorrect"], key="misclassified_toggle", on_change=reset_page)

            with st.container(border=True):
                overlay = st.pills("Visual Overlay", ["Heatmap", "Bounding Box"], selection_mode="multi")

            filtered_samples = samples[:sample_index]

            if gender_filter == "Male":
                filtered_samples = [
                    sample for sample in filtered_samples 
                    if sample.image_data.gender.numerator == 0
                ]
            elif gender_filter == "Female":
                filtered_samples = [
                    sample for sample in filtered_samples 
                    if sample.image_data.gender.numerator == 1
                ]
            
            # Filter if Misclassified
            if classification == "Incorrect":
                filtered_samples = [
                    sample for sample in filtered_samples 
                    if sample.image_data.gender.numerator != sample.predicted_gender.numerator
                ]
            elif classification == "Correct":
                filtered_samples = [
                    sample for sample in filtered_samples 
                    if sample.image_data.gender.numerator == sample.predicted_gender.numerator
                ]
                
        with images.container():
            image_generator(filtered_samples, overlay)
            
    if st.button("Go Back", type="primary", use_container_width=True):
                st.session_state.layout = "centered"
                st.session_state.configuration = True
                st.rerun()

# Sample Image Viewer Tab
def image_generator(samples, overlay):
    start = st.session_state.page[0]
    end = st.session_state.page[1]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for i, sample in enumerate(samples[start:end]):
        col = i % 6

        image = sample.image_data.preprocessed_image
        activation = sample.activation_map
        bboxes = sample.activation_boxes

        true_gender = sample.image_data.gender.numerator
        pred_gender = sample.predicted_gender.numerator
        confidence = sample.prediction_confidence

        fig = image_overlays(image, activation, bboxes, overlay)

        match col:
            case 0:
                with c1:
                    with st.container(border=True):
                        st.pyplot(fig)
                        st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                                    **Confidence**: {confidence:.2f}""",unsafe_allow_html=True)
            case 1:
                with c2:
                    with st.container(border=True):
                        st.pyplot(fig)
                        st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                                    **Confidence**: {confidence:.2f}""",unsafe_allow_html=True)
            case 2:
                with c3:
                    with st.container(border=True):
                        st.pyplot(fig)
                        st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                                    **Confidence**: {confidence:.2f}""",unsafe_allow_html=True)
            case 3:
                with c4:
                    with st.container(border=True):
                        st.pyplot(fig)
                        st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                                    **Confidence**: {confidence:.2f}""",unsafe_allow_html=True)
            case 4:
                with c5:
                    with st.container(border=True):
                        st.pyplot(fig)
                        st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                                    **Confidence**: {confidence:.2f}""",unsafe_allow_html=True)
            case 5:
                with c6:
                    with st.container(border=True):
                        st.pyplot(fig)
                        st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                                    **Confidence**: {confidence:.2f}""",unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        if st.session_state.page[0] == 0:
            st.button("---", key = "disabled_back", disabled=True, use_container_width=True)
        else:
            if st.button("Back", use_container_width=True):
                st.session_state.page[1] = st.session_state.page[0]
                st.session_state.page[0] = st.session_state.page[0] - 18
                st.rerun()
    with p2:
        if st.session_state.page[1] >= len(samples):
            st.button("---", key = "disabled_next", disabled=True, use_container_width=True)
        else:
            if st.button("Next", use_container_width=True):
                st.session_state.page[0] = st.session_state.page[1]
                st.session_state.page[1] = st.session_state.page[1] + 18
                st.rerun()

if __name__ == "__main__":
    
    if st.session_state.configuration:
        display_configuration_page()
    else:
        display_visualization_page()
