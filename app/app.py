import streamlit as st
from biasx import BiasAnalyzer
import utils
from graphs import (
    create_radar_chart,
    create_feature_probability_chart,
    create_spatial_heatmap,
    create_classwise_performance_chart,
    create_confusion_matrix,
    create_roc_curve,
    create_precision_recall_curve,
    image_overlays,
    create_legend
)

def main():
    # Initialize session state variables
    utils.initialize_session_state()
    
    # Set page configuration
    st.set_page_config(layout=st.session_state.layout)
    
    # Display the appropriate page based on the application state
    if st.session_state.config["model"]["path"] is None:
        display_model_upload_page()
    elif st.session_state.configuration and st.session_state.config["model"]["path"]:
        display_configuration_page()
    else:
        display_visualization_page()

def display_model_upload_page():
    st.write("# BiasX")
    st.markdown("BiasX is a Python library for detecting and explaining gender bias in face classification models. This repository provides a toolkit to analyze bias through both traditional fairness metrics and feature-level analysis. Visual heatmaps and quantitative bias scores are generated to help developers understand which facial features contribute to biased classifications.")
    st.markdown("")

    with st.container(border=True):
        st.write("### Upload Own Model")
        uploaded_file = st.file_uploader("Upload", 
                                        type=["h5", "keras"], 
                                        key="uploaded_model", 
                                        label_visibility="collapsed",
                                        accept_multiple_files=False)
        if uploaded_file is not None:
            st.session_state.file_info = uploaded_file.name
            st.session_state.config["model"]["path"] = utils.create_temp_file(uploaded_file)
            st.rerun()

    with st.container(border=True):
        st.write("### Select from Pre-existing Models")
        repo_id = "4shL3I/biasx-models" # Change to final repo ID

        model_set = st.selectbox("Select a model from Hugging Face",
                ["baseline_models", "gender_bias_models", "male_attention-bias_models", "female_attention-bias_models"], 
                key="model_selection", 
                label_visibility="collapsed")
        
        model_options = utils.retrieve_model_options(repo_id, model_set)
        
        selected_model = st.pills("Select a model", 
                list(model_options.keys()), 
                key="pretrained_model", 
                label_visibility="collapsed")
        
        if selected_model:
            utils.show_model_info(model_options[selected_model])

        st.markdown("")
        continue_button = st.button("Continue", type="primary", use_container_width=True)

        if continue_button:
            with st.spinner("Downloading model..."):
                model_filename = model_options[selected_model]["path"]

                # Download the model from Hugging Face
                model_path = utils.load_hf_model(repo_id, model_filename)

                # Create a temporary file for the downloaded model
                with open(model_path, "rb") as file_data:
                    st.session_state.config["model"]["path"] = utils.create_temp_file(file_data)

                st.session_state.file_info = model_filename

                st.success(f"{selected_model} downloaded successfully!")

                st.rerun()

def display_configuration_page():
    """Display the configuration page for model settings."""
    # --- Model Configuration ---
    with st.container(border=True):
        st.markdown("### Model Configuration")
        model_upload, model_config = st.columns(2)
        
        with model_upload:
            st.write(f"## {st.session_state.file_info}")
            if st.button("Change Model", use_container_width=True):
                st.session_state.config["model"]["path"] = None
                st.rerun()
        
        with model_config.container(border=True):
            c1, c2 = st.columns(2)
            st.session_state.config["dataset"]["image_width"] = c1.number_input("Image Width", key="input_image_width", value=st.session_state.model_config["width"])
            st.session_state.config["dataset"]["image_height"] = c2.number_input("Image Height", key="input_image_height", value=st.session_state.model_config["height"])
            st.session_state.config["dataset"]["color_mode"] = c1.pills("Color Mode", ["Grayscale", "RGB"], key="color_mode", default=st.session_state.model_config["color_mode"])
            if st.session_state.config["dataset"]["color_mode"] == "Grayscale":
                st.session_state.config["dataset"]["color_mode"] = "L"
            channel = c2.pills("Channel", ["Single", "Triple"], key="channel", selection_mode="single", default=st.session_state.model_config["channel"])
            st.session_state.config["dataset"]["single_channel"] = channel == "Single"
            st.session_state.config["model"]["inverted_classes"] = st.toggle("Invert Label?", value=st.session_state.invert_label)

    # --- Other Configuration ---
    with st.container():
        explainer, other = st.columns(2)

        with explainer.container(border=True):
            st.markdown("### Explainer Configuration")

            # --- Class Activation Config ---
            with st.container(border=True):
                st.session_state.config["explainer"]["cam_method"] = st.pills("Class Activation Map Method", ["gradcam", "gradcam++", "scorecam"], key="cam_method", selection_mode="single", default=st.session_state.config["explainer"]["cam_method"])
                st.session_state.config["explainer"]["cutoff_percentile"] = st.slider("Cutoff Percentile", key="cam_threshold", value=st.session_state.config["explainer"]["cutoff_percentile"])

            # --- Threshold Config ---
            with st.container(border=True):
                st.session_state.config["explainer"]["threshold_method"] = st.pills("Thresholding Method", ["otsu", "triangle", "sauvola"], key="threshold_method", selection_mode="single", default=st.session_state.config["explainer"]["threshold_method"])
                st.session_state.config["explainer"]["overlap_threshold"] = st.slider("Overlap Threshold", key="overlap_ratio", value=st.session_state.config["explainer"]["overlap_threshold"])

            # --- Distance Metric Config ---
            with st.container(border=True):
                st.session_state.config["explainer"]["distance_metric"] = st.pills("Distance Metric", ["euclidean", "cityblock",  "cosine"], key="distant_metric", selection_mode="single", default=st.session_state.config["explainer"]["distance_metric"])

        with other.container():
            # --- Dataset Config ---
            with st.container(border=True):
                st.markdown("### Dataset Configuration")
                st.session_state.config["dataset"]["source"] = st.pills("Dataset Selection", ["utkface", "fairface"], key="dataset_selection", selection_mode="single", default=st.session_state.config["dataset"]["source"])
                c1, c2 = st.columns(2)
                st.session_state.config["dataset"]["max_samples"] = c1.number_input("Sample Size", key="sample_size", value=st.session_state.config["dataset"]["max_samples"], disabled=st.session_state.select_all)
                # select_all = c1.toggle("Select All", key="select_all")
                st.session_state.config["dataset"]["seed"] = c2.number_input("Seed", key="seed", value=st.session_state.config["dataset"]["seed"], disabled=not st.session_state.enable_shuffle)
                st.session_state.config["dataset"]["shuffle"] = c2.toggle("Shuffle?", key="enable_shuffle", value=st.session_state.enable_shuffle)

            # --- Start Analysis ---
            if st.button("Start Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing", show_time=True):
                    analyzer = BiasAnalyzer(st.session_state.config)
                    st.session_state.result = analyzer.analyze()
                
                utils.reset_page()
                st.session_state.layout = "wide"
                st.session_state.configuration = False
                st.rerun()

def display_visualization_page():
    """Display visualization page with analysis results."""
    tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Model Performance", "Analysis"])

    # --- Feature Analysis Tab ---
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
        c1, c2 = st.columns([1, 2])
        with c1.container(border=False):
            # --- Display Bias Score and Equalized Odd ---
            with st.container(border=False):
                metric1, metric2 = st.columns(2)
                metric1.metric(f"Bias Score", st.session_state.result.disparity_scores.biasx, border=True)
                metric2.metric(f"Equalized Odds", st.session_state.result.disparity_scores.equalized_odds, border=True)
            with st.container(border=True):
                # --- Display Radar Chart ---
                radar_chart = create_radar_chart(feature_analysis)
                st.plotly_chart(radar_chart, use_container_width=True)
                st.markdown("Features closer to the outer edge of the radar chart have higher bias scores, indicating they more strongly influence gender misclassifications.")

        with c2.container(border=False):
            with st.container(border=True):
                # --- Display Feature Probability Chart ---
                probability_chart = create_feature_probability_chart(feature_analysis)
                st.plotly_chart(probability_chart, use_container_width=True)
                st.markdown("Bars show how often each feature is activated during misclassifications. Large differences between male and female probabilities indicate potential bias.")

    # --- Model Performance Tab ---
    with tab2.container(border=True):
        st.markdown("### Model Performance")
        c1, c2 = st.columns([1, 2])
        with c1.container(border=False):
            with st.container(border=True):
                # --- Display Confusion Matrix ---
                confusion_matrix = create_confusion_matrix(st.session_state.result.explanations)
                st.plotly_chart(confusion_matrix, use_container_width=True)
                st.markdown("The confusion matrix shows prediction patterns across genders. Ideally, the diagonal values should be similar, indicating balanced performance.")
                
        with c2.container(border=False):
            with st.container(border=True):
                # --- Display Precision, Recall, and F1-Score ---
                class_wise = create_classwise_performance_chart(st.session_state.result.explanations)
                st.plotly_chart(class_wise, use_container_width=True)
                st.markdown("The class-wise metrics show the model's precision, recall, and F1-score for each gender. Balanced scores indicate fair performance, while large gaps may suggest bias or weaknesses in classification.")

        c1, c2 = st.columns(2)
        with c1.container(border=False):
            with st.container(border=True):
                # --- Display Precision Recall Curve ---
                precision_recall_curve = create_precision_recall_curve(st.session_state.result.explanations)
                st.plotly_chart(precision_recall_curve, use_container_width=True)
                st.markdown("The Precision-Recall curve highlights the tradeoff between precision and recall. A higher curve suggests better performance, especially for imbalanced datasets where they are more meaningful than accuracy.")

        with c2.container(border=False):
            with st.container(border=True):
                # --- Display ROC Curve ---
                roc_curve = create_roc_curve(st.session_state.result.explanations)
                st.plotly_chart(roc_curve, use_container_width=True)
                st.markdown("The ROC curve shows the tradeoff between true positive rate and false positive rate. A curve closer to the top-right corner indicates better performance.")

    # --- Image Analysis Tab ---
    with tab3.container(border=True):
        filter_column, images_column = st.columns([1, 3])

        with filter_column.container():
            st.markdown("### Image Analysis")
            st.markdown("""
            This tab shows individual image analysis with activation maps.
            Activation maps highlight the regions the model focuses on when making predictions.
            """)

            # --- Display Spatial Heatmaps ---
            with st.popover("Reveal Spatial Heatmap", use_container_width=True):
                male, female = st.columns(2)
                with male.container(border=True):
                    st.markdown("#### Male Spatial Heatmap ####")
                    spatial = create_spatial_heatmap(st.session_state.result.explanations, gender_focus=0)
                    st.plotly_chart(spatial, key="male_spatial", use_container_width=True)

                with female.container(border=True):
                    st.markdown("#### Female Spatial Heatmap ####")
                    spatial = create_spatial_heatmap(st.session_state.result.explanations, gender_focus=1)
                    st.plotly_chart(spatial, key="female_spatial",  use_container_width=True)

            samples = st.session_state.result.explanations
            max_samples = min(len(samples), 300)
            display_count = 30
            
            with st.container(border=True):
                sample_index = st.slider("Sample Index", display_count, max_value=max_samples, 
                                         value=display_count, on_change=utils.reset_page)
                
            c1, c2 = st.columns(2)
            with c1.container(border=True):
                gender_filter = st.pills("Gender", ["Male", "Female"], key="gender_filter", 
                                         selection_mode="single", on_change=utils.reset_page)

            with c2.container(border=True):
                classification = st.pills("Classification", ["Correct", "Incorrect"], 
                                          key="misclassified_toggle", on_change=utils.reset_page)
        
            with st.container(border=True):
                landmark_names = ["left_eye", "right_eye", "nose", "lips", "left_cheek", 
                                  "right_cheek", "chin", "forehead", "left_eyebrow", "right_eyebrow"]
                facial_feature = st.pills("Facial Feature", landmark_names, 
                                          selection_mode="multi", on_change=utils.reset_page)

            with st.container(border=True):
                overlay = st.pills("Visual Overlay", ["Landmark Boxes", "Heatmap", "Bounding Boxes"], 
                                   selection_mode="multi")

            # --- Filter Images ---
            filtered_samples = utils.filter_samples(samples, sample_index, gender_filter, classification, facial_feature)

            if "Landmark Boxes" in overlay:
                legend = create_legend()
                st.plotly_chart(legend, use_container_width=True)
            
        with images_column.container():
            display_images(filtered_samples, overlay, facial_feature)
            
    if st.button("Go Back", type="primary", use_container_width=True):
        st.session_state.layout = "centered"
        st.session_state.configuration = True
        utils.reset_config()
        st.rerun()

def display_images(samples, overlay, facial_feature):
    """Display image grid with filters and pagination."""
    start = st.session_state.page[0]
    end = st.session_state.page[1]

    columns = st.columns(6)  # Store columns in a list
    for i, sample in enumerate(samples[start:end]):
        col = columns[i % 6]

        image = sample.image_data.preprocessed_image
        activation = sample.activation_map
        bboxes = sample.activation_boxes
        landmark_boxes = sample.landmark_boxes

        true_gender = sample.image_data.gender.numerator
        pred_gender = sample.predicted_gender.numerator
        confidence = sample.prediction_confidence

        fig = image_overlays(image, activation, landmark_boxes, bboxes, overlay, facial_feature)

        with col:
            with st.container(border=True):
                st.pyplot(fig)
                st.markdown(f"""**True**: {true_gender}, **Predicted**: {pred_gender} <br>
                            **Confidence**: {confidence:.2f}""", unsafe_allow_html=True)

    # Pagination controls
    prev_page, next_page = st.columns(2)
    with prev_page:
        if st.session_state.page[0] == 0:
            st.button("---", key="disabled_back", disabled=True, use_container_width=True)
        else:
            if st.button("Back", use_container_width=True):
                st.session_state.page[1] = st.session_state.page[0]
                st.session_state.page[0] = st.session_state.page[0] - 18
                st.rerun()
    with next_page:
        if st.session_state.page[1] >= len(samples):
            st.button("---", key="disabled_next", disabled=True, use_container_width=True)
        else:
            if st.button("Next", use_container_width=True):
                st.session_state.page[0] = st.session_state.page[1]
                st.session_state.page[1] = st.session_state.page[1] + 18
                st.rerun()

if __name__ == "__main__":
    main()