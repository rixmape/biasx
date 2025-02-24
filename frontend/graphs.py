import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, roc_curve

from biasx.types import Explanation


def create_radar_chart(feature_scores: dict[str, float]) -> go.Figure:
    """Creates a radar chart comparing bias scores across facial features."""
    features = list(feature_scores.keys())
    scores = list(feature_scores.values())

    fig = go.Figure(
        go.Scatterpolar(
            r=scores + [scores[0]],
            theta=features + [features[0]],
            fill="toself",
            fillcolor="blue",
            line=dict(color="blue"),
        )
    )

    fig.update_layout(polar=dict(radialaxis=dict(range=[0, max(scores) + 0.1], tickformat=".2f")))

    return fig


def create_parallel_coordinates(feature_probabilities: dict[str, dict[int, float]]) -> go.Figure:
    """Creates a parallel coordinates plot showing gender and feature activation relationships."""
    dimensions = [
        dict(
            range=[0, 1],
            label=feature.replace("_", " ").title(),
            values=[probs[0], probs[1]],
            tickformat=".2f",
        )
        for feature, probs in feature_probabilities.items()
    ]

    fig = go.Figure()
    fig.add_trace(go.Parcoords(line=dict(color=[0, 1], colorscale=[[0, "blue"], [1, "red"]]), dimensions=dimensions))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Male", line=dict(color="blue"), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Female", line=dict(color="red"), showlegend=True))

    fig.update_layout(showlegend=True, legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05))

    return fig


def create_confusion_matrix(explanations: list[Explanation]) -> go.Figure:
    """Creates a confusion matrix visualization for gender predictions."""
    y_true = [exp.true_gender for exp in explanations]
    y_pred = [exp.predicted_gender for exp in explanations]
    cm = confusion_matrix(y_true, y_pred, normalize="pred")

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Predicted Male", "Predicted Female"],
            y=["True Male", "True Female"],
            text=[[f"{val:.2%}" for val in row] for row in cm],
            texttemplate="%{text}",
            colorscale="Blues",
            zmax=1,
            showscale=False,
        )
    )

    fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", yaxis=dict(tickangle=-90))

    return fig


def create_roc_curves(explanations: list[Explanation]) -> go.Figure:
    """Creates ROC curves for model performance analysis."""
    y_true = np.array([exp.true_gender for exp in explanations])
    y_score = np.array([exp.prediction_confidence for exp in explanations])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fpr_inv, tpr_inv, _ = roc_curve(1 - y_true, 1 - y_score)
    roc_auc_inv = auc(fpr_inv, tpr_inv)

    fig = go.Figure(
        [
            go.Scatter(x=fpr, y=tpr, name=f"Female (AUC = {roc_auc:.3f})", line=dict(color="red", width=2)),
            go.Scatter(x=fpr_inv, y=tpr_inv, name=f"Male (AUC = {roc_auc_inv:.3f})", line=dict(color="blue", width=2)),
            go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="gray", width=2), showlegend=False),
        ]
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.02],
        legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05),
    )

    return fig


def create_violin_plot(explanations: list[Explanation]) -> go.Figure:
    """Creates a violin plot showing confidence score distributions across prediction outcomes."""
    correct = {"male": [], "female": []}
    incorrect = {"male": [], "female": []}

    for exp in explanations:
        target = correct if exp.predicted_gender == exp.true_gender else incorrect
        gender = "male" if exp.true_gender == 0 else "female"
        target[gender].append(exp.prediction_confidence)

    fig = go.Figure(
        [
            go.Violin(
                x=["Male"] * len(correct["male"]) + ["Female"] * len(correct["female"]),
                y=correct["male"] + correct["female"],
                side="positive",
                fillcolor="blue",
                name="Correct",
            ),
            go.Violin(
                x=["Male"] * len(incorrect["male"]) + ["Female"] * len(incorrect["female"]),
                y=incorrect["male"] + incorrect["female"],
                side="negative",
                fillcolor="red",
                name="Incorrect",
            ),
        ]
    )

    fig.update_layout(
        yaxis_title="Confidence Score",
        yaxis_range=[-0.02, 1.02],
        violinmode="overlay",
        legend=dict(yanchor="bottom", xanchor="right", x=0.95, y=0.05),
    )

    return fig
