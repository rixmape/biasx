from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches

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

def create_spatial_heatmap(
    explanations: list,
    misclassified_only: bool = True,
    gender_focus: int = None  # 0 for Male, 1 for Female, None for all
):
    """Creates a spatial heatmap of activation frequency across facial regions."""
    
    # Filter explanations based on misclassification and gender
    filtered_exps = [
        exp for exp in explanations
        if (not misclassified_only or exp.predicted_gender != exp.image_data.gender.numerator)
        and (gender_focus is None or exp.image_data.gender.numerator == gender_focus)
    ]
    features = [exp.landmark_boxes for exp in filtered_exps]
    flat_features = [box.feature for sublist in features for box in sublist]

    heatmap = np.zeros((48, 48))

    for exp in filtered_exps:
        for box in exp.activation_boxes:
            if not flat_features or box.feature in flat_features:
                # continue  # Skip if the feature is not in our list
            # y_range = (max(0, box.min_y), min(48, box.max_y))
            # x_range = (max(0, box.min_x), min(48, box.max_x))
            # heatmap[y_range[0] : y_range[1], x_range[0] : x_range[1]] += 1
                size = 210
                y_min_scaled = int((box.min_y / size) * 48)
                y_max_scaled = int((box.max_y / size) * 48)
                x_min_scaled = int((box.min_x / size) * 48)
                x_max_scaled = int((box.max_x / size) * 48)

                y_range = (max(0, y_min_scaled), min(size, y_max_scaled))
                x_range = (max(0, x_min_scaled), min(size, x_max_scaled))

                heatmap[y_range[0]: y_range[1], x_range[0]: x_range[1]] += 1


    max_activation = np.max(heatmap)
    normalized_heatmap = heatmap / max_activation if max_activation > 0 else heatmap

    # Create the heatmap figure
    fig = go.Figure(go.Heatmap(z=normalized_heatmap, colorscale="Reds", showscale=False, zmin=0, zmax=1))
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-1, 49]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, range=[49, -1]),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        plot_bgcolor="rgba(0,0,0,0)",
        width=400,  # Adjust to your desired width
        height=400, 
    )

    return fig

def create_confusion_matrix(explanations):
    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_pred = np.array([exp.predicted_gender.numerator for exp in explanations])

    cm = confusion_matrix(y_true, y_pred)

    custom_colorscale = [[0, "#2B2D42"], [1, "#EDF2F4"]]

    labels = ["Male", "Female"]
    fig = px.imshow(cm, 
                    x=labels, y=labels, 
                    color_continuous_scale=custom_colorscale,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    title="Confusion Matrix",
                    text_auto=True,)  # Show values inside heatmap cells
                    
    fig.update_layout(
        autosize=False,
        height= 400,  
        title_font=dict(size=20),  
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14), 
        yaxis=dict(tickangle=-90),
        font=dict(size=14),
        coloraxis_showscale=False,
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
    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_pred = np.array([exp.predicted_gender.numerator for exp in explanations])

    # Compute precision, recall, and f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    labels = ["Male", "Female"]

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

def image_overlays(image, heatmap, landmark_boxes, bboxes, overlay, colors, facial_feature):
    fig, ax = plt.subplots()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))

    ax.imshow(image)
    
    if "Heatmap" in overlay:
        heatmap = cv2.resize(heatmap, (200, 200))
        ax.imshow(heatmap, cmap="jet", alpha=0.3)  # Overlay activation map
    
    if "Landmark Boxes" in overlay:
        for i, box in enumerate(landmark_boxes):
            if not facial_feature or (box.feature and box.feature.value in facial_feature):
                min_x, min_y, max_x, max_y = box.min_x, box.min_y, box.max_x, box.max_y
                color = colors[i % len(colors)]
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                        linewidth=3, edgecolor=color, facecolor="none", alpha=0.9)
                ax.add_patch(rect)
    
    if "Bounding Boxes" in overlay:
        for box in bboxes:
            if not facial_feature or (box.feature and box.feature.value in facial_feature):
                min_x, min_y, max_x, max_y = box.min_x, box.min_y, box.max_x, box.max_y
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                        linewidth=4, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

    ax.axis("off")  # Hide axis


    return fig

def create_legend(landmark_names, colors):    
    fig = go.Figure()
    
    for i, name in enumerate(landmark_names):
        color = colors[i % len(colors)]
        
        # Add an invisible scatter point with a legend entry
        fig.add_trace(go.Scatter(
            x=[0], 
            y=[0],
            mode='markers',
            marker=dict(size=12, color=color),
            name=name,
            showlegend=True
        ))
    
    # Hide the axes and make the plot area transparent
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=100,  # Minimal height, just enough for the legend
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="middle",
            y=0.5,
            xanchor="center",
            x=0.5,
            font=dict(size=11) 
        ),
    )
    
    return fig