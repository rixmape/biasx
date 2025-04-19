import streamlit as st


def display_home_page():
    st.subheader("What is BiasX?")
    st.markdown(
        """
        BiasX is a comprehensive analytical framework for detecting, measuring, and explaining gender bias in facial
        classification models. Unlike traditional fairness tools that only quantify bias through statistical metrics,
        :red-background[BiasX reveals why bias occurs] by connecting model decisions to specific facial features.
        """
    )

    with st.expander("Details", expanded=False):
        st.markdown(
            """
            The BiasX framework initially takes facial images from a dataset and the parameters of the AI model being
            evaluated to perform image classification. Based on these classification results, the system then generates
            visual explanations, highlighting the specific facial regions that most influenced the model's decision.
            Finally, combining the classification outcomes with the visual explanations of significant regions, the
            framework conducts a bias analysis, culminating in a comprehensive report for the AI model developer that
            details the model's performance and potential biases.
            """
        )
        st.image("app/assets/dataflow.png")
        st.markdown(
            """
            External Links:&nbsp;&nbsp;
            [GitHub Repository](https://github.com/rixmape/biasx)&nbsp;&nbsp;/&nbsp;
            [PyPI Project](https://pypi.org/project/biasx/)&nbsp;&nbsp;/&nbsp;&nbsp;
            [Streamlit Cloud](https://biasxframework.streamlit.app/)&nbsp;&nbsp;/&nbsp;&nbsp;
            [Project Documentation](https://rixmape.github.io/biasx/)
            """
        )

    st.subheader("How to Use BiasX?")
    c1, c2, c3 = st.columns(3)

    with c1.container(border=True):
        st.write("#### :red-background[&nbsp;&nbsp;1&nbsp;&nbsp;]&nbsp;&nbsp;Setup")  # `&nbsp;` for consecutive spaces
        st.write("Upload your face classification model.")

    with c2.container(border=True):
        st.write("#### :red-background[&nbsp;&nbsp;2&nbsp;&nbsp;]&nbsp;&nbsp;Configure")
        st.write("Specify settings for model, dataset, and analysis.")

    with c3.container(border=True):
        st.write("#### :red-background[&nbsp;&nbsp;3&nbsp;&nbsp;]&nbsp;&nbsp;Analyze")
        st.write("Run BiasX to analyze the model and visualize results.")

    st.subheader("Try BiasX Now!")

    if st.button("Upload or Select Model", type="primary", use_container_width=True):
        st.session_state.show_upload_page = True
        st.rerun()
