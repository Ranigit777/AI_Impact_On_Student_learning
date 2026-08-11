"""Sidebar input form for student profile data."""

import streamlit as st

from app.config import (
    INPUT_BOUNDS,
    MAJOR_CATEGORIES,
    PROMPT_SKILLS,
    YEARS_OF_STUDY,
)


def render_sidebar() -> dict | None:
    """
    Render sidebar inputs and return a dict of values when the user clicks Predict.

    Returns None if the predict button has not been clicked.
    """
    with st.sidebar:
        st.markdown('<p class="sidebar-header">Student Profile</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-subtext">Enter academic and wellbeing indicators to assess burnout risk.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Academic Background")
        major_category = st.selectbox("Major Category", MAJOR_CATEGORIES, index=0)
        year_of_study = st.selectbox("Year of Study", YEARS_OF_STUDY, index=2)

        gpa_min, gpa_max, gpa_default = INPUT_BOUNDS["gpa"]
        gpa = st.slider("GPA", min_value=float(gpa_min), max_value=float(gpa_max),
                        value=float(gpa_default), step=0.05)

        st.markdown("##### AI & Study Habits")
        ai_min, ai_max, ai_default = INPUT_BOUNDS["weekly_ai_hours"]
        weekly_ai_hours = st.slider(
            "Weekly AI Hours", min_value=float(ai_min), max_value=float(ai_max),
            value=float(ai_default), step=0.5,
            help="Average hours per week spent using Generative AI tools."
        )

        study_min, study_max, study_default = INPUT_BOUNDS["study_hours"]
        study_hours = st.slider(
            "Study Hours (Traditional)", min_value=float(study_min), max_value=float(study_max),
            value=float(study_default), step=0.5,
            help="Weekly hours spent studying without AI assistance."
        )

        td_min, td_max, td_default = INPUT_BOUNDS["tool_diversity"]
        tool_diversity = st.slider(
            "Tool Diversity", min_value=int(td_min), max_value=int(td_max),
            value=int(td_default), step=1,
            help="Number of distinct AI tools used regularly."
        )

        prompt_skill = st.selectbox("Prompt Engineering Skill", PROMPT_SKILLS, index=1)

        paid_subscription = st.toggle("Paid Subscription", value=False,
                                      help="Whether the student subscribes to paid AI services.")

        st.markdown("##### Wellbeing")
        anx_min, anx_max, anx_default = INPUT_BOUNDS["anxiety_level"]
        anxiety_level = st.slider(
            "Anxiety Level (Exams)", min_value=int(anx_min), max_value=int(anx_max),
            value=int(anx_default), step=1,
            help="Self-reported exam anxiety on a scale of 1 (low) to 10 (high)."
        )

        sr_min, sr_max, sr_default = INPUT_BOUNDS["skill_retention"]
        skill_retention = st.slider(
            "Skill Retention Score", min_value=float(sr_min), max_value=float(sr_max),
            value=float(sr_default), step=0.5,
            help="Evaluated knowledge retention score (0–100)."
        )

        st.divider()
        predict_clicked = st.button("Predict Burnout Risk", type="primary", use_container_width=True)

    if not predict_clicked:
        return None

    return {
        "major_category": major_category,
        "year_of_study": year_of_study,
        "weekly_ai_hours": weekly_ai_hours,
        "study_hours": study_hours,
        "anxiety_level": anxiety_level,
        "prompt_skill": prompt_skill,
        "paid_subscription": paid_subscription,
        "tool_diversity": tool_diversity,
        "skill_retention": skill_retention,
        "gpa": gpa,
    }
