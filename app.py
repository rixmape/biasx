import streamlit as st


st.title(":rainbow[BiasX ⚖]")

st.container(height=18, border=False)  # Spacer
st.subheader(":one: Upload your model")

st.file_uploader("", type=["h5", "keras"], key="uploaded_model", label_visibility="collapsed")


st.container(height=18, border=False)
st.subheader(":two: Configure the pipeline")

tab1, tab2 = st.tabs(["Model", "Dataset"])

with tab1:
    col1, col2 = st.columns(2)
    col1.number_input("Image Width", key="input_image_width", value=224)
    col2.number_input("Image Height", key="input_image_height", value=224)

    st.toggle("Single Channel", key="single_channel", value=True)

with tab2:
    st.radio("Test Data", ["UTKFace"], key="dataset", index=0, horizontal=True)
    st.number_input("Sample Size", key="sample_size", value=0)
    st.toggle("Shuffle", key="shuffle", value=True)
    st.number_input("Seed", key="seed", value=69)

st.container(height=18, border=False)
st.subheader(":three: Analyze the model")

st.button("Start Analysis", key="start_analysis")
