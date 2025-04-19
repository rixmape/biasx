import streamlit as st


def display_model_configuration():
    st.write("#### Face Classification Model")
    st.write("")

    with st.container():
        c1, c2, c3, _ = st.columns(4)

        channel = c1.pills(
            "Channel",
            ["Single", "Triple"],
            key="channel",
            selection_mode="single",
            default=st.session_state.model_config["channel"],
            help="Specify whether the model expects input images with a single (grayscale) or triple (color) channel dimension.",
        )
        st.session_state.config["dataset"]["single_channel"] = channel == "Single"

        color_mode = c2.pills(
            "Color Mode",
            ["Grayscale", "RGB"],
            key="color_mode",
            default=st.session_state.model_config["color_mode"],
            help="Select the color format (Grayscale or RGB) to which input images should be converted before model processing.",
        )
        st.session_state.config["dataset"]["color_mode"] = "L" if color_mode == "Grayscale" else "RGB"

        # NOTE: Inverted classes set to True by default, assuming IdentiFace is chosen
        inverted_classes = c3.pills(
            "Inverted Classes",
            ["True", "False"],
            key="inverted_classes",
            selection_mode="single",
            default="True",
            help="Indicate if the model's output labels for gender (e.g., 0 for male, 1 for female) are the reverse of the library's default expectation.",
        )
        st.session_state.config["model"]["inverted_classes"] = inverted_classes == "True"

    with st.container():
        c1, c2 = st.columns(2)

        st.session_state.config["dataset"]["image_width"] = c1.number_input(
            "Image Width",
            key="input_image_width",
            value=st.session_state.model_config["width"],
            help="Define the target width in pixels to which all input images will be resized.",
        )
        st.session_state.config["dataset"]["image_height"] = c2.number_input(
            "Image Height",
            key="input_image_height",
            value=st.session_state.model_config["height"],
            help="Define the target height in pixels to which all input images will be resized.",
        )


def display_dataset_configuration():
    st.write("#### Facial Image Dataset")
    st.write("")

    with st.container():
        c1, c2, *_ = st.columns(4)
        st.session_state.config["dataset"]["source"] = c1.pills(
            "Dataset Selection",
            ["utkface", "fairface"],
            key="dataset_selection",
            selection_mode="single",
            default=st.session_state.config["dataset"]["source"],
            help="Choose the source dataset (utkface or fairface) to use for the bias analysis.",
        )

        shuffle = c2.pills(
            "Shuffle Dataset",
            ["True", "False"],
            key="shuffle_dataset",
            selection_mode="single",
            default="True",
            help="Determine whether to randomly shuffle the dataset records before analysis.",
        )
        st.session_state.config["dataset"]["shuffle"] = shuffle == "True"

    with st.container():
        c1, c2 = st.columns(2)
        st.session_state.config["dataset"]["max_samples"] = c1.number_input(
            "Sample Size",
            key="sample_size",
            value=st.session_state.config["dataset"]["max_samples"],
            help="Specify the maximum number of images to randomly sample from the selected dataset for analysis.",
        )
        st.session_state.config["dataset"]["seed"] = c2.number_input(
            "Seed",
            key="seed",
            value=st.session_state.config["dataset"]["seed"],
            help="Set the random seed for reproducibility when shuffling or sampling the dataset.",
        )


def display_explainer_configuration():
    st.write("#### Visual Explainer")
    st.write("")

    st.session_state.config["explainer"]["distance_metric"] = st.pills(
        "Distance Metric",
        ["euclidean", "cityblock", "cosine"],
        key="distant_metric",
        selection_mode="single",
        default=st.session_state.config["explainer"]["distance_metric"],
        help="Select the method used to calculate the distance between activation box centers and landmark box centers for feature matching.",
    )

    with st.container():
        c1, c2 = st.columns(2)

        st.session_state.config["explainer"]["cam_method"] = c1.pills(
            "Class Activation Map Method",
            ["gradcam", "gradcam++", "scorecam"],
            key="cam_method",
            selection_mode="single",
            default=st.session_state.config["explainer"]["cam_method"],
            help="Choose the algorithm to generate heatmaps highlighting important regions for model predictions.",
        )
        st.session_state.config["explainer"]["threshold_method"] = c2.pills(
            "Thresholding Method",
            ["otsu", "triangle", "sauvola"],
            key="threshold_method",
            selection_mode="single",
            default=st.session_state.config["explainer"]["threshold_method"],
            help="Select the algorithm used to binarize the activation heatmap after initial filtering.",
        )

    st.write("")

    with st.container():
        c1, c2 = st.columns(2)

        st.session_state.config["explainer"]["cutoff_percentile"] = c1.slider(
            "Cutoff Percentile",
            key="cam_threshold",
            value=st.session_state.config["explainer"]["cutoff_percentile"],
            help="Set the percentile threshold below which activation values in the heatmap will be ignored (set to zero).",
        )
        st.session_state.config["explainer"]["overlap_threshold"] = c2.slider(
            "Overlap Threshold",
            key="overlap_ratio",
            value=st.session_state.config["explainer"]["overlap_threshold"],
            help="Define the minimum required Intersection over Area (IoA) between an activation box and a landmark box to label the activation box with the landmark's feature.",
        )


def display_buttons():
    c1, c2 = st.columns(2)

    if c1.button("Change Model", use_container_width=True):
        st.session_state.config["model"]["path"] = None
        st.rerun()

    if c2.button("Start Analysis", type="primary", use_container_width=True):
        st.session_state.start_analysis = True
        st.session_state.configuration = False
        st.rerun()


def display_configuration_page():
    st.write("### Define the pipeline components")
    st.write("")

    with st.container(border=True):
        display_model_configuration()

    with st.container(border=True):
        display_dataset_configuration()

    with st.container(border=True):
        display_explainer_configuration()

    st.write("")

    display_buttons()
