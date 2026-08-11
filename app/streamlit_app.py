"""
Burnout Risk Predictor – Streamlit Application
===============================================
Run from the project root:

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for `app.*` and `src.*` imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.results_panel import render_prediction_results, render_welcome_state
from app.components.sidebar_inputs import render_sidebar
from app.services.model_loader import get_feature_importance, load_model_package
from app.services.predictor import predict_burnout
from app.utils import configure_page, load_custom_css, render_footer, render_header


def main() -> None:
    configure_page()
    load_custom_css()
    render_header()

    # Verify model is available
    try:
        load_model_package()
        importance_data = get_feature_importance()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    inputs = render_sidebar()

    if inputs is None:
        render_welcome_state()
    else:
        with st.spinner("Analyzing student profile..."):
            prediction = predict_burnout(inputs)
        render_prediction_results(inputs, prediction, importance_data)

    render_footer()


if __name__ == "__main__":
    main()
