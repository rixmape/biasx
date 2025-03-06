import streamlit as st
import tempfile
import pandas as pd
import numpy as np
import pandas as pd


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



def configuration_page():
    # Model Configuration
    with st.container(border=True):
        st.markdown("### Model Configuration")
        model_upload, model_config = st.columns(2)
        with model_upload:
            uploaded_file = model_upload.file_uploader("Upload", type=["h5", "keras"], key="uploaded_model", label_visibility="collapsed")

        with model_config.container(border=True):
            c1, c2 = st.columns(2)
            width = c1.number_input("Image Width", key="input_image_width", value=224)
            height = c2.number_input("Image Height", key="input_image_height", value=224)
            color_mode = c1.pills("Color Mode", ["Grayscale", "RGB"], key="color_mode", selection_mode="single")
            channel = c2.pills("Color Mode", ["Single", "Triple"], key="channel", selection_mode="single")
            invert_label = st.toggle("Invert Label?", value=st.session_state.invert_label)

        if uploaded_file is not None:
            # Create a temporary file to store the uploaded model
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as temp_file:
                temp_file.write(uploaded_file.read())  # Save the uploaded file
                temp_path = temp_file.name  # Get the path of the temp file

            st.session_state.model_path = temp_path

    # Other Configuration
    with st.container():
        explainer, other = st.columns(2)

        with explainer.container(border=True):
            st.markdown("### Explainer Configuration")
            max_face = st.number_input("Maximum Number of Face to detect", key="max_face", value=1)

            with st.container(border=True):
                cam_method = st.pills("Class Activation Map Method", ["GradCam", "GradCam++", "ScoreCam"], key="cam_method", selection_mode="single")
                cam_threshold = st.number_input("Class Activation Map Threshold", key="cam_threshold", value=1)
                threshold_method = st.pills("Thresholding Method", ["otsu", "niblack", "sauvola"], key="threshold_method", selection_mode="single")

            with st.container(border=True):
                min_overlap = st.number_input("Minimum Overlap Ratio", key="overlap_ratio", value=1)
                distant_metric = st.pills("Distant Metric", ["Euclidean", "Manhattan"], key="distant_metric", selection_mode="single")

        with other.container():
            with st.container(border=True):
                st.markdown("### Dataset Configuration")
                threshold_method = st.pills("Dataset Selection", ["UTKFace", "FairFace"], key="dataset_selection", selection_mode="multi")
                c1, c2 = st.columns(2)
                sample_size = c1.number_input("Sample Size", key="sample_size", value=1, disabled=st.session_state.select_all)
                use_all_data = c1.toggle("Select All", key = "select_all")
                shuffle_seed = c2.number_input("Seed", key="seed", value=69, disabled = not st.session_state.enable_shuffle)
                enable_shuffle = c2.toggle("Shuffle?", key="enable_shuffle")

            with st.container(border=True):
                st.markdown("### Calculator")
                decimal_place = st.number_input("Number of Decimal Place", key="decimal_place", value=1)

            if st.button("Start Analysis", type="primary", use_container_width=True):
                st.session_state.layout = "wide"
                st.session_state.configuration = False
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


def visualization_page():
    tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Model Performance", "Analysis"])

    # Feature Analysis Tab
    with tab1.container(border=True):
        st.markdown("### Feature Analysis")
        c1, c2 = st.columns(2)
        with c1.container(border=False):
            placeholder()
            placeholder()

        with c2.container(border=False):
            placeholder()
            placeholder()

    # Model Performance
    with tab2.container(border=True):
        st.markdown("### Model Performance")
        c1, c2 = st.columns([1,2])
        with c1.container(border=False):
            placeholder()
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


if __name__ == "__main__":
    if st.session_state.configuration:
        configuration_page()
    else:
        visualization_page()
