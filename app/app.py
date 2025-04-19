import streamlit as st

# isort: off
from utils import initialize_session_state
from sections.home import display_home_page
from sections.config import display_configuration_page
from sections.results import display_visualization_page
from sections.upload import display_model_upload_page


def main():
    """Initializes the app and routes to the appropriate page."""
    initialize_session_state()

    st.set_page_config(
        page_title="BiasX Framework",
        page_icon=":bar_chart:",
        layout=st.session_state.layout,
        initial_sidebar_state="collapsed",
    )

    st.title("BiasX Framework")

    if st.session_state.show_upload_page:
        display_model_upload_page()
    elif st.session_state.config["model"]["path"] is None:
        display_home_page()
    elif st.session_state.configuration:
        display_configuration_page()
    else:
        display_visualization_page()


if __name__ == "__main__":
    main()
