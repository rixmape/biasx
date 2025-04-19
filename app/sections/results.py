import pathlib
from functools import lru_cache

import streamlit as st
import yaml

# isort: off
from biasx.analyzer import BiasAnalyzer
from utils import filter_samples, reset_page
from visualizations.feature_analysis import create_feature_probability_chart, create_radar_chart
from visualizations.image_analysis import create_image_with_overlays, create_legend
from visualizations.model_performance import create_classwise_performance_chart, create_confusion_matrix, create_precision_recall_curve, create_roc_curve


@lru_cache()
def load_chart_descriptions(file_path="app/assets/chart_descriptions.yaml"):
    try:
        with open(file_path, "r") as f:
            descriptions = yaml.safe_load(f)
        return descriptions
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Chart descriptions will be missing.")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file '{file_path}': {e}. Chart descriptions will be missing.")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading chart descriptions: {e}")
        return {}


def display_feature_analysis_tab(results, descriptions):
    feature_analysis = {
        feature.name: {
            "bias_score": analysis.bias_score,
            "male_probability": analysis.male_probability,
            "female_probability": analysis.female_probability,
        }
        for feature, analysis in results.feature_analyses.items()
    }

    fm_desc = descriptions.get("feature_analysis", {}).get("fairness_metrics", {})
    st.write(f"#### {fm_desc.get('title', 'Fairness Metrics')}")
    st.write(fm_desc.get("description", "Shows overall model fairness metrics."))

    c1, c2 = st.columns(2)
    c1.metric("Bias Score", f"{results.disparity_scores.biasx:.3f}", border=True, help="Average absolute bias score across features.")
    c2.metric("Equalized Odds", f"{results.disparity_scores.equalized_odds:.3f}", border=True, help="Max difference in TPR/FPR between genders.")

    st.write("")

    fsbs_desc = descriptions.get("feature_analysis", {}).get("feature_specific_bias_scores", {})
    st.write(f"#### {fsbs_desc.get('title', 'Feature-Specific Bias Scores')}")
    st.write(fsbs_desc.get("description", "Shows bias scores for each facial feature."))

    with st.container(border=True):
        radar_chart = create_radar_chart(feature_analysis)
        st.plotly_chart(radar_chart, use_container_width=True)

    st.write("")

    fap_desc = descriptions.get("feature_analysis", {}).get("feature_activation_probabilities", {})
    st.write(f"#### {fap_desc.get('title', 'Feature Activation Probabilities')}")
    st.write(fap_desc.get("description", "Compares feature activation probability during misclassifications by gender."))

    with st.container(border=True):
        probability_chart = create_feature_probability_chart(feature_analysis)
        st.plotly_chart(probability_chart, use_container_width=True)


def display_model_performance_tab(results, descriptions):
    cm_desc = descriptions.get("model_performance", {}).get("confusion_matrix", {})
    st.write(f"#### {cm_desc.get('title', 'Confusion Matrix')}")
    st.write(cm_desc.get("description", "Shows true vs. predicted gender counts."))

    with st.container(border=True):
        confusion_matrix_chart = create_confusion_matrix(results.explanations)
        st.plotly_chart(confusion_matrix_chart, use_container_width=True)

    st.write("")

    mpm_desc = descriptions.get("model_performance", {}).get("model_performance_metrics", {})
    st.write(f"#### {mpm_desc.get('title', 'Model Performance Metrics')}")
    st.write(mpm_desc.get("description", "Shows Precision, Recall, F1-score per gender."))

    with st.container(border=True):
        class_wise = create_classwise_performance_chart(results.explanations)
        st.plotly_chart(class_wise, use_container_width=True)

    st.write("")

    prc_desc = descriptions.get("model_performance", {}).get("precision_recall_curve", {})
    st.write(f"#### {prc_desc.get('title', 'Precision-Recall Curve')}")
    st.write(prc_desc.get("description", "Shows precision vs. recall tradeoff."))

    with st.container(border=True):
        pr_curve = create_precision_recall_curve(results.explanations)
        st.plotly_chart(pr_curve, use_container_width=True)

    st.write("")

    roc_desc = descriptions.get("model_performance", {}).get("roc_curves", {})
    st.write(f"#### {roc_desc.get('title', 'ROC Curves')}")
    st.write(roc_desc.get("description", "Shows true positive rate vs. false positive rate tradeoff."))

    with st.container(border=True):
        roc_curve = create_roc_curve(results.explanations)
        st.plotly_chart(roc_curve, use_container_width=True)


@st.dialog("Image Details")
def show_image_details_dialog(fig, image_id, true_gender, pred_gender, confidence):
    st.pyplot(fig)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(":gray-background[Image ID]")
        st.markdown(":gray-background[True Gender]")
        st.markdown(":gray-background[Predicted Gender]")
        st.markdown(":gray-background[Confidence]")
    with col2:
        st.markdown(f"{image_id[:16]}")
        st.markdown(f"{true_gender}")
        st.markdown(f"{pred_gender}")
        st.markdown(f"{confidence:.2f}")


def display_images(samples, overlay, facial_feature):
    start, end = st.session_state.page
    ncol = 3
    columns = st.columns(ncol)

    for i, sample in enumerate(samples[start:end]):
        col = columns[i % ncol]

        image = sample.image_data.preprocessed_image
        true_gender = sample.image_data.gender.name.capitalize()
        pred_gender = sample.predicted_gender.name.capitalize()
        confidence = sample.prediction_confidence
        image_id = sample.image_data.image_id

        fig = create_image_with_overlays(image, sample, overlay, facial_feature, color_mode=st.session_state.config["dataset"]["color_mode"])

        with col:
            st.pyplot(fig)
            button_key = f"details_{image_id}_{i}"
            if st.button("Show Details", key=button_key, use_container_width=True):
                show_image_details_dialog(fig, image_id, true_gender, pred_gender, confidence)

    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.session_state.page[0] == 0:
            st.button("← Back", key="disabled_back", disabled=True, use_container_width=True)
        else:
            if st.button("← Back", key="back", use_container_width=True):
                st.session_state.page[0] -= 18
                st.session_state.page[1] = st.session_state.page[0] + 18
                st.rerun()

    with next_col:
        if st.session_state.page[1] >= len(samples):
            st.button("Next →", key="disabled_next", disabled=True, use_container_width=True)
        else:
            if st.button("Next →", key="next", use_container_width=True):
                st.session_state.page[0] = st.session_state.page[1]
                st.session_state.page[1] += 18
                st.rerun()


def display_image_analysis_tab(results):
    with st.expander(label="Image Analysis Filters", expanded=True):
        samples = results.explanations
        max_samples = min(len(samples), 300)
        display_count = 18

        c1, c2, *_ = st.columns(4)
        gender_filter = c1.pills("Gender", ["Male", "Female"], key="gender_filter", default=None, on_change=reset_page)
        classification_filter = c2.pills("Classification", ["Correct", "Incorrect"], key="misclassified_toggle", default=None, on_change=reset_page)

        sample_size = st.number_input("Select Sample Range End", display_count, max_samples, value=min(max_samples, 100), on_change=reset_page, help=f"Analyze images up to this index (max {max_samples}).")

        landmark_names = ["left_eye", "right_eye", "nose", "lips", "left_cheek", "right_cheek", "chin", "forehead", "left_eyebrow", "right_eyebrow"]
        facial_feature_filter = st.multiselect("Activated Facial Feature", landmark_names, key="facial_feature_filter", on_change=reset_page, help="Show images where these features were activated.")

        overlay_options = ["Landmark Boxes", "Heatmap", "Bounding Boxes"]
        overlay_filter = st.multiselect("Visual Overlay", overlay_options, default=["Heatmap", "Bounding Boxes"], key="overlay_filter")

        if "Landmark Boxes" in overlay_filter:
            legend = create_legend()
            st.plotly_chart(legend, use_container_width=True)

    filtered_samples = filter_samples(samples, sample_size, gender_filter, classification_filter, facial_feature_filter)

    if filtered_samples:
        st.markdown(f"#### Displaying {min(len(filtered_samples), display_count)} of {len(filtered_samples)} Samples")
        display_images(filtered_samples, overlay_filter, facial_feature_filter)
    else:
        st.info("No samples match the current filter criteria within the selected range.")


def display_return_button():
    if st.button("Return to Configuration Page", use_container_width=True):
        st.session_state.layout = "centered"
        st.session_state.configuration = True
        st.rerun()


def display_visualization_page():
    chart_descriptions = load_chart_descriptions()

    if st.session_state.get("start_analysis", False):
        try:
            with st.spinner("Analyzing...", show_time=True):
                analyzer = BiasAnalyzer(st.session_state.config)
                st.session_state.result = analyzer.analyze()
            st.session_state.start_analysis = False
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.session_state.result = None
            st.session_state.start_analysis = False

    if st.session_state.result is None:
        st.error("No analysis results found. Please configure and run an analysis first.")
        display_return_button()
        return

    tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Model Performance", "Image Analysis"])

    with tab1:
        display_feature_analysis_tab(st.session_state.result, chart_descriptions)

    with tab2:
        display_model_performance_tab(st.session_state.result, chart_descriptions)

    with tab3:
        display_image_analysis_tab(st.session_state.result)

    st.write("")
    st.write("")

    display_return_button()
