"""Utility helpers for the Streamlit application."""

from pathlib import Path

import streamlit as st

from app.config import CSS_PATH


def load_custom_css() -> None:
    """Inject custom CSS into the Streamlit page."""
    if CSS_PATH.exists():
        with open(CSS_PATH, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_header() -> None:
    """Render the application hero banner."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-badge">AI Student Learning Analytics</div>
            <h1>Burnout Risk Predictor</h1>
            <p>
                Machine-learning powered assessment of student burnout risk based on
                academic performance, AI usage patterns, and wellbeing indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render the application footer."""
    st.markdown(
        """
        <div class="app-footer">
            Burnout Risk Prediction System &mdash; Trained on 46,735 student records
            &nbsp;|&nbsp; Model: Logistic Regression (GridSearchCV tuned)
        </div>
        """,
        unsafe_allow_html=True,
    )


def configure_page() -> None:
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="Burnout Risk Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
