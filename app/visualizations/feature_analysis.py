import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_radar_chart(feature_analyses: list):
    categories = list(feature_analyses.keys())
    values = [feature_analyses[feature]["bias_score"] for feature in categories]

    fig = go.Figure()
    trace = go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line=dict(color="blue", width=2),
        marker=dict(size=10, symbol="circle"),
    )
    fig.add_trace(trace)
    fig.update_layout(
        margin=dict(l=0, r=0, b=30, t=30, pad=0),
        height=350,
    )

    return fig


def create_feature_probability_chart(feature_analyses: list):
    data = [
        {
            "Feature": feature,
            "Male Probability": feature_analyses[feature]["male_probability"],
            "Female Probability": feature_analyses[feature]["female_probability"],
        }
        for feature in feature_analyses
    ]

    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Feature",
        y=["Male Probability", "Female Probability"],
        barmode="group",
    )
    fig.update_layout(
        legend=dict(yanchor="bottom", xanchor="right", x=1, y=1, orientation="h", title=None),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        height=400,
    )

    return fig
