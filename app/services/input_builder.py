"""Build a model-ready feature row from Streamlit sidebar inputs."""

import pandas as pd

from app.config import DEFAULT_INSTITUTIONAL_POLICY, DEFAULT_PRIMARY_USE_CASE


def estimate_ai_dependency(weekly_ai_hours: float, tool_diversity: int) -> int:
    """
    Estimate perceived AI dependency when not directly collected.
    Heuristic based on usage intensity and tool diversity.
    """
    raw_score = 1.0 + (weekly_ai_hours * 0.32) + (tool_diversity * 0.45)
    return int(min(10, max(1, round(raw_score))))


def build_feature_dataframe(inputs: dict) -> pd.DataFrame:
    """
    Convert sidebar inputs into a single-row DataFrame matching training schema.

    The model expects engineered features and categorical columns that are
    transformed by the saved ColumnTransformer at inference time.
    """
    gpa = inputs["gpa"]
    weekly_ai = inputs["weekly_ai_hours"]
    study_hours = inputs["study_hours"]
    perceived_dependency = estimate_ai_dependency(weekly_ai, inputs["tool_diversity"])

    gpa_improvement = 0.0
    ai_efficiency = gpa_improvement / (weekly_ai + 1.0)
    study_balance = study_hours / (weekly_ai + 1.0)
    dependency_index = (perceived_dependency * weekly_ai) / (study_hours + 1.0)

    row = {
        "Pre_Semester_GPA": gpa,
        "Weekly_GenAI_Hours": weekly_ai,
        "Tool_Diversity": inputs["tool_diversity"],
        "Traditional_Study_Hours": study_hours,
        "Perceived_AI_Dependency": perceived_dependency,
        "Anxiety_Level_During_Exams": inputs["anxiety_level"],
        "Post_Semester_GPA": gpa,
        "Skill_Retention_Score": inputs["skill_retention"],
        "GPA_Improvement": gpa_improvement,
        "AI_Efficiency": ai_efficiency,
        "Study_Balance": study_balance,
        "Dependency_Index": dependency_index,
        "Major_Category": inputs["major_category"],
        "Year_of_Study": inputs["year_of_study"],
        "Primary_Use_Case": inputs.get("primary_use_case", DEFAULT_PRIMARY_USE_CASE),
        "Prompt_Engineering_Skill": inputs["prompt_skill"],
        "Institutional_Policy": inputs.get("institutional_policy", DEFAULT_INSTITUTIONAL_POLICY),
    }

    return pd.DataFrame([row])
