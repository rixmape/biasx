import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, precision_recall_fscore_support, roc_curve


def create_confusion_matrix(explanations: list):
    labels = ["Male", "Female"]
    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_pred = np.array([exp.predicted_gender.numerator for exp in explanations])
    cm = confusion_matrix(y_true, y_pred)

    # TODO: Rotate y-tick labels to be vertical
    # TODO: Increase font size of all tick labels
    fig = px.imshow(cm, x=labels, y=labels, color_continuous_scale="Reds", labels=dict(x="Predicted", y="True", color="Count"), text_auto=True)
    fig.update_layout(font=dict(size=16), coloraxis_showscale=False, height=300, margin=dict(l=0, r=0, t=0, b=0))

    return fig


def create_roc_curve(explanations: list):
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
        legend=dict(yanchor="bottom", xanchor="right", x=1, y=1, orientation="h", title=None),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def create_precision_recall_curve(explanations: list):
    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_score = np.array([exp.prediction_confidence for exp in explanations])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    baseline = sum(y_true) / len(y_true)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"AUC = {pr_auc:.2f}", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines", name=f"Baseline ({baseline:.2f})", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        legend=dict(yanchor="bottom", xanchor="right", x=1, y=1, orientation="h", title=None),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig


def create_classwise_performance_chart(explanations: list):
    y_true = np.array([exp.image_data.gender.numerator for exp in explanations])
    y_pred = np.array([exp.predicted_gender.numerator for exp in explanations])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    labels = ["Male", "Female"]

    df = pd.DataFrame(
        {
            "Class": labels * 3,
            "Metric": ["Precision", "Recall", "F1-score"] * len(labels),
            "Value": list(precision) + list(recall) + list(f1),
        }
    )

    fig = px.bar(
        df,
        x="Class",
        y="Value",
        color="Metric",
        barmode="group",
        labels={"Value": "Score", "Class": "Class"},
        color_discrete_map={"Precision": "#2b2d42", "Recall": "#8d99ae", "F1-score": "#edf2f4"},
    )
    fig.update_layout(
        legend=dict(yanchor="bottom", xanchor="right", x=1, y=1, orientation="h", title=None),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig
