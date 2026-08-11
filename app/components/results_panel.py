"""Main results panel: prediction summary, metrics, and recommendations."""

import streamlit as st

from app.components.charts import render_charts
from app.config import RISK_STYLES
from app.services.recommendations import build_prediction_summary, generate_recommendations


def render_welcome_state() -> None:
    """Display onboarding content before the first prediction."""
    st.markdown(
        """
        <div class="metric-card">
            <h3>How it works</h3>
            <p style="color:#64748b; margin:0; line-height:1.7;">
                1. Fill in the <strong>Student Profile</strong> form in the sidebar.<br>
                2. Click <strong>Predict Burnout Risk</strong>.<br>
                3. Review the predicted risk level, probability distribution, feature importance, and personalized recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-card"><h3>7 Models Trained</h3><p class="value">Logistic Regression</p><p style="color:#64748b;font-size:0.85rem;margin:0;">Best performing classifier</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card"><h3>Dataset Size</h3><p class="value">46,735</p><p style="color:#64748b;font-size:0.85rem;margin:0;">Cleaned student records</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card"><h3>Target Classes</h3><p class="value">3 Levels</p><p style="color:#64748b;font-size:0.85rem;margin:0;">Low · Medium · High</p></div>',
            unsafe_allow_html=True,
        )


def render_prediction_results(
    inputs: dict,
    prediction: dict,
    importance_data: list[dict],
) -> None:
    """Render the full prediction dashboard after inference."""
    summary = build_prediction_summary(inputs, prediction)

    st.markdown(
        f"""
        <div class="risk-panel" style="background:{summary['risk_bg']}; border-color:{summary['risk_border']};">
            <h2 style="color:{summary['risk_color']};">{summary['headline']}</h2>
            <p style="color:#334155;">
                Model confidence: <strong>{summary['confidence_pct']}%</strong> &nbsp;|&nbsp;
                Highest probability class: <strong>{summary['top_class']}</strong> ({summary['top_probability_pct']}%)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    probs = prediction["probabilities"]
    with m1:
        st.markdown(
            f'<div class="metric-card"><h3>Low Risk</h3>'
            f'<p class="value" style="color:{RISK_STYLES["Low"]["color"]}">{probs.get("Low", 0):.1%}</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h3>Medium Risk</h3>'
            f'<p class="value" style="color:{RISK_STYLES["Medium"]["color"]}">{probs.get("Medium", 0):.1%}</p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h3>High Risk</h3>'
            f'<p class="value" style="color:{RISK_STYLES["High"]["color"]}">{probs.get("High", 0):.1%}</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><h3>Confidence</h3><p class="value">{summary["confidence_pct"]}%</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-title">Analytics Dashboard</p>', unsafe_allow_html=True)
    render_charts(inputs, prediction, importance_data)

    # Prediction summary table
    st.markdown('<p class="section-title">Prediction Summary</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Input Profile**")
        st.dataframe(
            {
                "Field": [
                    "Major Category", "Year of Study", "GPA",
                    "Weekly AI Hours", "Study Hours", "Anxiety Level",
                    "Prompt Skill", "Paid Subscription", "Tool Diversity", "Skill Retention",
                ],
                "Value": [
                    inputs["major_category"], inputs["year_of_study"], f"{inputs['gpa']:.2f}",
                    f"{inputs['weekly_ai_hours']:.1f}", f"{inputs['study_hours']:.1f}",
                    str(inputs["anxiety_level"]), inputs["prompt_skill"],
                    "Yes" if inputs["paid_subscription"] else "No",
                    str(inputs["tool_diversity"]), f"{inputs['skill_retention']:.1f}",
                ],
            },
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        st.markdown("**Probability Distribution**")
        st.dataframe(
            {
                "Risk Level": list(prediction["probabilities"].keys()),
                "Probability": [f"{v:.2%}" for v in prediction["probabilities"].values()],
            },
            use_container_width=True,
            hide_index=True,
        )

    # Recommendations
    st.markdown('<p class="section-title">Personalized Recommendations</p>', unsafe_allow_html=True)
    recommendations = generate_recommendations(inputs, prediction)

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="rec-card {rec['priority']}">
                <h4>{rec['title']}</h4>
                <p>{rec['detail']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
