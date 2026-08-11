"""Chart components for prediction visualization."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.config import RISK_STYLES


def plot_probability_chart(probabilities: dict) -> go.Figure:
    """Horizontal bar chart of class probabilities."""
    ordered_labels = ["Low", "Medium", "High"]
    df = pd.DataFrame({
        "Risk Level": ordered_labels,
        "Probability": [probabilities.get(label, 0.0) for label in ordered_labels],
    })
    colors = [RISK_STYLES[label]["color"] for label in ordered_labels]

    fig = px.bar(
        df, x="Probability", y="Risk Level", orientation="h",
        color="Risk Level", color_discrete_map={
            "Low": RISK_STYLES["Low"]["color"],
            "Medium": RISK_STYLES["Medium"]["color"],
            "High": RISK_STYLES["High"]["color"],
        },
        text=df["Probability"].apply(lambda v: f"{v:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=280,
        margin=dict(l=10, r=30, t=30, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_feature_importance(importance_data: list[dict], top_n: int = 12) -> go.Figure:
    """Horizontal bar chart of global feature importances from the trained model."""
    df = pd.DataFrame(importance_data[:top_n]).sort_values("importance", ascending=True)

    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=20, t=30, b=10),
        coloraxis_showscale=False,
        xaxis_title="Importance Score",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_risk_gauge(confidence: float, risk_level: str) -> go.Figure:
    """Gauge chart showing confidence in the predicted risk level."""
    style = RISK_STYLES.get(risk_level, RISK_STYLES["Medium"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"Confidence ({risk_level})", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": style["color"]},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40], "color": "#ecfdf5"},
                {"range": [40, 70], "color": "#fffbeb"},
                {"range": [70, 100], "color": "#fef2f2"},
            ],
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def plot_input_profile(inputs: dict) -> go.Figure:
    """Radar chart comparing student inputs against normalized reference values."""
    categories = ["AI Hours", "Study Hours", "Anxiety", "Tool Diversity", "Skill Retention", "GPA"]
    values = [
        inputs["weekly_ai_hours"] / 26.0,
        inputs["study_hours"] / 26.0,
        inputs["anxiety_level"] / 10.0,
        inputs["tool_diversity"] / 5.0,
        inputs["skill_retention"] / 100.0,
        (inputs["gpa"] - 1.8) / (4.0 - 1.8),
    ]
    values = [min(1.0, max(0.0, v)) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(79, 70, 229, 0.15)",
        line=dict(color="#4f46e5", width=2),
        name="Your Profile",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def render_charts(inputs: dict, prediction: dict, importance_data: list[dict]) -> None:
    """Render all dashboard charts in a two-column layout."""
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_probability_chart(prediction["probabilities"]),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            plot_risk_gauge(prediction["confidence"], prediction["risk_level"]),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(
            plot_input_profile(inputs),
            use_container_width=True,
        )

    with col4:
        st.plotly_chart(
            plot_feature_importance(importance_data),
            use_container_width=True,
        )
