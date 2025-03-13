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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, auc



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
    st.session_state.page = [0, 18]

if "result" not in st.session_state:
    st.session_state.result =  {
        'feature_analyses': None,
        'disparity_score': None,
        'confusion_matrix': None,
        'image_data': None,
        'figures': None}

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
        true_gender = explanation.image_data.gender.numerator  # Convert Enum to string
        predicted_gender = explanation.predicted_gender.numerator  # Convert Enum to string

        data_dict[i] = {
            "true_gender": true_gender,
            "predicted_gender": predicted_gender,
            "prediction_confidence": explanation.prediction_confidence
        }
    
    figures = generate_figures(analysis)

    return {
        'feature_analyses': feature_analysis_dict,
        'classification': analysis.explanations,
        'disparity_score': disparity_scores,
        'image_data': analysis.explanations,
        'figures': figures
    }

def create_confusion_matrix(explanations):
    # y_true = [entry["true_gender"] for entry in data_dict.values()]
    # y_pred = [entry["predicted_gender"] for entry in data_dict.values()]

    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_pred = np.array([exp.predicted_gender.numerator for exp in explanations])

    
    cm = confusion_matrix(y_true, y_pred)

    custom_colorscale = [[0, "#2B2D42"], [1, "#EDF2F4"]]

    # Define labels (0 = Male, 1 = Female)
    labels = ["Male", "Female"] if st.session_state.config["model"]["inverted_classes"] else ["Female", "Male"]
    fig = px.imshow(cm, 
                    x=labels, y=labels, 
                    color_continuous_scale=custom_colorscale,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    title="Confusion Matrix",
                    text_auto=True)  # Show values inside heatmap cells
                    

    fig.update_layout(
        autosize=False,
        height= 400,  
        title_font=dict(size=20),  
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14), 
        font=dict(size=14)  
    )

    return fig  

def create_roc_curve(explanations):
    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_score = np.array([exp.prediction_confidence for exp in explanations])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fpr_inv, tpr_inv, _ = roc_curve(1 - y_true, 1 - y_score)
    roc_auc_inv = auc(fpr_inv, tpr_inv)

    fig = go.Figure(
        [
            go.Scatter(x=fpr, y=tpr, name=f"Female (AUC = {roc_auc:.3f})", line=dict(color="red", width=2)),
            go.Scatter(x=fpr_inv, y=tpr_inv, name=f"Male (AUC = {roc_auc_inv:.3f})", line=dict(color="orange", width=2)),
            go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", width=2), showlegend=False),
        ]
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.02],
        legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05),
        height = 300
    )

    return fig  

def create_precision_recall_curve(explanations):
    # y_true = [entry["true_gender"] for entry in data_dict.values()]
    # y_scores = [entry["predicted_gender"] for entry in data_dict.values()]  # Assuming probabilities exist

    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_score = np.array([exp.predicted_gender.numerator for exp in explanations])

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)  # AUC for Precision-Recall Curve

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", 
                             name=f"AUC = {pr_auc:.2f}", 
                             line=dict(color="orange"))) 

    fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode="lines", 
                             name="Baseline", 
                             line=dict(dash="dash", color="gray")))

    fig.update_layout(
        title="Precision Recall Curve",
        title_font=dict(size=20), 
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_title_font=dict(size=14),  
        yaxis_title_font=dict(size=14), 
        font=dict(size=14),  
        autosize=False,
        height=300
    )

    return fig  

def create_classwise_performance_chart(explanations):
    # y_true = [entry["true_gender"] for entry in data_dict.values()]
    # y_pred = [entry["predicted_gender"] for entry in data_dict.values()]

    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_pred = np.array([exp.predicted_gender.numerator for exp in explanations])

    # Compute precision, recall, and f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    # Define labels (0 = Male, 1 = Female)
    labels = ["Male", "Female"] if st.session_state.config["model"]["inverted_classes"] else ["Female", "Male"]

    # Create DataFrame for Plotly
    metrics_df = pd.DataFrame({
        "Class": labels * 3,
        "Metric": ["Precision", "Recall", "F1-score"] * len(labels),
        "Value": list(precision) + list(recall) + list(f1)
    })

    # Create a bar chart
    fig = px.bar(
        metrics_df,
        x="Class",
        y="Value",
        color="Metric",
        barmode="group",
        title="Class-wise Performance Metrics",
        labels={"Value": "Score", "Class": "Class"},
        color_discrete_map={"Precision": "#2b2d42", "Recall": "#8d99ae", "F1-score": "#edf2f4"}
    )

    # Update layout for better readability
    fig.update_layout(
        autosize=False,
        height=400,
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=14),
        yaxis=dict(range=[0, 1])  # Scores range from 0 to 1
    )

    return fig

def create_radar_chart(feature_analyses):
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
        marker=dict(size=10, symbol='circle')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=zoom_range)
        ),
        showlegend=False,
        autosize = False,
        height = 500,
        title = "Radial Chart",
        title_font=dict(size=20)
    )

    return fig

def create_feature_probability_chart(feature_analyses):
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
    
    fig.update_layout(
        autosize = False,
        height = 500,
        title = "Feature Activation Probability by Gender",
        title_font=dict(size=20)
    )

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
                st.markdown(f"# Bias Score: {st.session_state.result["disparity_score"]["biasx"]}")
            with st.container(border=True):
                radar_chart = create_radar_chart(st.session_state.result['feature_analyses'])
                st.plotly_chart(radar_chart, use_container_width=True)
                st.markdown("**Interpretation:** Features closer to the outer edge of the radar chart have higher bias scores, indicating they more strongly influence gender misclassifications.")

        with c2.container(border=False):
            with st.container(border=True):
                probability_chart = create_feature_probability_chart(st.session_state.result['feature_analyses'])
                st.plotly_chart(probability_chart, use_container_width=True, use_containner_height=True)
                st.markdown("**Interpretation:** Bars show how often each feature is activated during misclassifications. Large differences between male and female probabilities indicate potential bias. <br>.", unsafe_allow_html=True)

    # Model Performance
    with tab2.container(border=True):
        st.markdown("### Model Performance")
        c1, c2 = st.columns([1,2])
        with c1.container(border=False):
            with st.container(border=True):
                confusion_matrix = create_confusion_matrix(st.session_state.result["classification"])
                st.plotly_chart(confusion_matrix, use_container_width=True)
                st.markdown("**Interpretation:** The confusion matrix shows prediction patterns across genders. Ideally, the diagonal values should be similar, indicating balanced performance.")
                

        with c2.container(border=False):
            with st.container(border=True):
                class_wise = create_classwise_performance_chart(st.session_state.result["classification"])
                st.plotly_chart(class_wise, use_container_width=True)
                st.markdown("**Interpretation:** The class-wise metrics show the model’s precision, recall, and F1-score for each gender. Balanced scores indicate fair performance, while large gaps may suggest bias or weaknesses in classification.")


        c1, c2 = st.columns(2)
        with c1.container(border=False):
            with st.container(border=True):
                precision_recall_curve = create_precision_recall_curve(st.session_state.result["classification"])
                st.plotly_chart(precision_recall_curve, use_container_width=True)
                st.markdown("**Interpretation:** The Precision-Recall curve highlights the tradeoff between precision and recall. A higher curve suggests better performance, especially for imbalanced datasets where they are more meaningful than accuracy.")

        with c2.container(border=False):
            with st.container(border=True):
                roc_curve = create_roc_curve(st.session_state.result["classification"])
                st.plotly_chart(roc_curve, use_container_width=True)
                st.markdown("**Interpretation:** The ROC curve shows the tradeoff between true positive rate and false positive rate. A curve closer to the top-right corner indicates better performance.")

    # Image Analysis Tab
    with tab3.container(border=True):
        c1, c2 = st.columns([1,3])

        with c1.container():
            st.markdown("### Image Analysis")
            st.markdown("""
            This tab shows individual image analysis with activation maps.
            Activation maps highlight the regions the model focuses on when making predictions.
            """)

            def reset_page():
                st.session_state.page[0] = 0
                st.session_state.page[1] = 18

            samples = st.session_state.result['image_data']
            max_samples = len(st.session_state.result["image_data"])
            display_count = 30
            sample_index = st.slider("Sample Index", display_count, max_value=max_samples, value=display_count, on_change=reset_page)
            current_samples = samples[:sample_index]


        with c2.container():
            image_generator(current_samples)
            

    if st.button("Go Back", type="primary", use_container_width=True):
                st.session_state.layout = "centered"
                st.session_state.configuration = True
                st.rerun()

def generate_figures(analysis):
    figures = []  # Temporary list to store figures

    for i, data in enumerate(analysis.explanations):
        if i >= 30:  # Limit to 30 images
            break

        image = data.image_data.preprocessed_image
        activation = data.activation_map

        # Create figure
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")  # Display grayscale image
        ax.imshow(activation, cmap="jet", alpha=0.5)  # Overlay activation map
        ax.axis("off")  # Hide axis

        figures.append(fig)  # Store in temporary list
    
    return figures

def generate_figure(image, activation):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")  # Display grayscale image
    ax.imshow(activation, cmap="jet", alpha=0.5)  # Overlay activation map
    ax.axis("off")  # Hide axis

    return fig

def display_image_with_overlays(image, activation):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")  # Display grayscale image
    ax.imshow(activation, cmap="jet", alpha=0.5)  # Overlay activation map
    ax.axis("off")  # Hide axis

    return fig

# Sample Image Viewer Tab
def image_generator(samples):
    start = st.session_state.page[0]
    end = st.session_state.page[1]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for i, sample in enumerate(samples[start:end]):
        col = i % 6

        image = sample.image_data.preprocessed_image
        activation = sample.activation_map

        true_gender = sample.image_data.gender.numerator
        pred_gender = sample.predicted_gender.numerator
        confidence = sample.prediction_confidence

        fig = generate_figure(image, activation)

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
            st.button("---", disabled=True, use_container_width=True)
        else:
            if st.button("Back", use_container_width=True):
                st.session_state.page[1] = st.session_state.page[0]
                st.session_state.page[0] = st.session_state.page[0] - 15
                st.rerun()
    with p2:
        if st.session_state.page[1] >= len(samples):
            st.button("---", disabled=True, use_container_width=True)
        else:
            if st.button("Next", use_container_width=True):
                st.session_state.page[0] = st.session_state.page[1]
                st.session_state.page[1] = st.session_state.page[1] + 15
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
