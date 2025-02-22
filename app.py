import streamlit as st
import tempfile
from biasx import BiasAnalyzer, BiasCalculator, ClassificationModel, FaceDataset, VisualExplainer
import json
import pandas as pd

if "select_all" not in st.session_state:
    st.session_state.select_all = False

if "model_path" not in st.session_state:
    st.session_state.model_path = ""

 

def streamlit():
    st.title(":rainbow[BiasX ⚖]")

    st.container(height=18, border=False)  # Spacer
    st.subheader(":one: Upload your model")

    uploaded_file = st.file_uploader("", type=["h5", "keras"], key="uploaded_model", label_visibility="collapsed")

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded model
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
            temp_file.write(uploaded_file.read())  # Save the uploaded file
            temp_path = temp_file.name  # Get the path of the temp file

        st.session_state.model_path = temp_path

    st.container(height=18, border=False)
    st.subheader(":two: Configure the pipeline")
    st.write(" ")


    model, spacer, dataset = st.columns([1, 0.09, 1])

    model_container = model.container()  # Create a container inside col1
    model_container.markdown("##### > Model Configuration <")
    c1, c2 = model_container.columns(2)
    width = c1.number_input("Image Width", key="input_image_width", value=224)
    height = c2.number_input("Image Height", key="input_image_height", value=224)
    
    # TODO: Add Option for channels | Can be selection
    model_container.write("-------------------- Work in Progress --------------------")
    model_container.toggle("Single Channel", key="single_channel", value=False)
    model_container.toggle("Triple Channel (Grayscale)", key="triple_channel_grayscale", value=False)
    model_container.toggle("Triple Channel (RGB)", key="single_channel_rgb", value=False)
    model_container.write("^--------------------------------------------------------------^")

    spacer.empty() 

    dataset_container = dataset.container()  # Create a container inside col2
    dataset_container.markdown("##### > Dataset Configuration <")
    dataset_container.radio("Test Data", ["UTKFace"], key="dataset", index=0, horizontal=True)
    sample_size = dataset_container.number_input("Sample Size", key="sample_size", value=0, disabled=st.session_state.select_all)
    c1, c2 = dataset_container.columns(2)
    c1.toggle("All Image", value=st.session_state.select_all, key="select_all")
    c2.toggle("Shuffle", key="shuffle", value=True)
    dataset_container.number_input("Seed", key="seed", value=69)

    st.container(height=18, border=False)
    st.subheader(":three: Analyze the model")


    if st.button("Start Analysis", key="start_analysis"):
        with st.spinner("Analyzing..", show_time=True):
            
            model_path = st.session_state.model_path
            sample_size = -1 if st.session_state.select_all else st.session_state.sample_size
            dataset_chosen = "dataset/UTKFace" if st.session_state.dataset == "UTKFace" else "More"
            target_size = (width, height)
            shuffle_val = st.session_state.shuffle
            seed_val = st.session_state.seed

            bias_analysis(model_path, target_size, dataset_chosen, sample_size, shuffle_val, seed_val)

def bias_analysis(model_path, target_size, dataset_chosen, sample_size, shuffle_val, seed_val):
    # TODO: Add Options for Channels
    model = ClassificationModel(model_path=model_path, target_size=target_size, single_channel=True)
    explainer = VisualExplainer(landmark_model_path="models/face_landmarker.task", landmark_map_path="models/landmark_map.json")
    calculator = BiasCalculator()
    analyzer = BiasAnalyzer(model=model, explainer=explainer, calculator=calculator)

    dataset = FaceDataset(dataset_path=dataset_chosen, max_samples=sample_size, shuffle=shuffle_val, seed=seed_val)
    analysis = analyzer.analyze(dataset=dataset, return_explanations=False)

    analysis.save("output/bias_analysis.json")

    view_analysis(analysis)


def view_analysis(analysis):
    feature_scores = analysis.feature_scores
    feature_probabilities = analysis.feature_probabilities
    bias_score = analysis.bias_score

   # Convert feature probabilities to DataFrame
    df_feature_probabilities = pd.DataFrame.from_dict(feature_probabilities, orient="index")

    # Add a new column for feature scores
    df_feature_probabilities["Feature Score"] = df_feature_probabilities.index.map(feature_scores)

    # Reset index to have 'Feature' as a column instead of index
    df_feature_probabilities.reset_index(inplace=True)
    df_feature_probabilities.rename(columns={"index": "Feature"}, inplace=True)

    # Display in Streamlit
    st.markdown(f"#### Bias Score = {bias_score}")

    st.markdown("#### Feature Probabilities with Feature Scores")
    st.table(df_feature_probabilities)

if __name__ == "__main__":
    streamlit()
