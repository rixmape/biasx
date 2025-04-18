import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import plotly.graph_objects as go

LANDMARKS = [
    "Left Eye",
    "Right Eye",
    "Nose",
    "Lips",
    "Left Cheek",
    "Right Cheek",
    "Chin",
    "Forehead",
    "Left Eyebrow",
    "Right Eyebrow",
]

# TODO: Use more contrasting colors
COLORS = [
    "#6A5ACD",
    "#27AE60",
    "#3498DB",
    "#1ABC9C",
    "#8E44AD",
    "#F39C12",
    "#16A085",
    "#F1C40F",
    "#5D6D7E",
    "#2980B9",
]


def create_image_with_overlays(image, sample, overlay=None, facial_feature=None, color_mode="L"):
    fig, ax = plt.subplots()

    if color_mode == "L":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))

    ax.imshow(image)

    if "Heatmap" in overlay:
        heatmap = cv2.resize(sample.activation_map, (200, 200))
        ax.imshow(heatmap, cmap="jet", alpha=0.3)

    if "Landmark Boxes" in overlay:
        for i, box in enumerate(sample.landmark_boxes):
            if not facial_feature or (box.feature and box.feature.value in facial_feature):
                min_x, min_y, max_x, max_y = box.min_x, box.min_y, box.max_x, box.max_y
                color = COLORS[i % len(COLORS)]
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=3, edgecolor=color, facecolor="none", alpha=0.9)
                ax.add_patch(rect)

    if "Bounding Boxes" in overlay:
        for box in sample.activation_boxes:
            if not facial_feature or (box.feature and box.feature.value in facial_feature):
                min_x, min_y, max_x, max_y = box.min_x, box.min_y, box.max_x, box.max_y
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, linewidth=4, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

    ax.axis("off")
    return fig


def create_legend():
    """Creates a legend for landmark visualization."""
    fig = go.Figure()

    for i, name in enumerate(LANDMARKS):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=12, color=color), name=name, showlegend=True))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="middle", y=0.5, xanchor="center", x=0.5),
    )

    return fig
