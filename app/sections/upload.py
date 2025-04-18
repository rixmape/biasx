import streamlit as st

# isort: off
from utils import create_temp_file, retrieve_model_options, load_hf_model


def display_file_uploader():
    st.write("#### :primary-background[&nbsp;Option 1&nbsp;]&nbsp;&nbsp;&nbsp;Upload your own model")
    st.write("")

    uploaded_file = st.file_uploader("", type=["h5", "keras"], key="uploaded_model", label_visibility="collapsed", accept_multiple_files=False)
    if uploaded_file:
        st.session_state.file_info = uploaded_file.name
        st.session_state.config["model"]["path"] = create_temp_file(uploaded_file)
        st.rerun()


def display_model_selection(model_options):
    st.write("#### :primary-background[&nbsp;Option 2&nbsp;]&nbsp;&nbsp;&nbsp;Select a pre-trained model")
    st.write("")

    selected_model = st.pills("", list(model_options.keys()), key="selected_models", label_visibility="collapsed")
    with st.expander("Model Information", expanded=False):
        if selected_model:
            st.json(model_options[selected_model])
        else:
            st.write("Select a model to view its information.")

    st.write("")

    # TODO: Automatically configure options for the selected model
    if st.button("Continue", type="primary", use_container_width=True, disabled=selected_model is None):
        with st.spinner("Downloading model..."):
            model_path = load_hf_model(model_filename=model_options[selected_model]["path"])
            if model_path:
                with open(model_path, "rb") as file_handler:
                    st.session_state.config["model"]["path"] = create_temp_file(file_handler)
                st.session_state.file_info = selected_model
                st.rerun()


def display_model_upload_page():
    """Displays the page for model selection or upload."""
    model_options = retrieve_model_options()
    if not model_options:
        st.error("Could not retrieve model options.")
        return

    st.write("### Setup the model to analyze")
    st.write("")

    with st.container(border=True):
        display_file_uploader()

    with st.container(border=True):
        display_model_selection(model_options)
