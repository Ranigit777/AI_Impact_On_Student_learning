"""Application configuration and input option constants."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "burnout_prediction_model.pkl"
CSS_PATH = Path(__file__).resolve().parent / "styles" / "custom.css"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Categorical options (aligned with training data)
MAJOR_CATEGORIES = ["STEM", "Business", "Humanities", "Medical", "Arts"]
YEARS_OF_STUDY = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
PROMPT_SKILLS = ["Beginner", "Intermediate", "Advanced"]
PRIMARY_USE_CASES = [
    "Summarizing_Reading",
    "Ideation",
    "Copywriting/Drafting",
    "Debugging/Troubleshooting",
    "Direct_Answer_Generation",
]
INSTITUTIONAL_POLICIES = [
    "Allowed_With_Citation",
    "Actively_Encouraged",
    "Strict_Ban",
]

# Defaults for fields not collected in the primary sidebar
DEFAULT_PRIMARY_USE_CASE = "Summarizing_Reading"
DEFAULT_INSTITUTIONAL_POLICY = "Allowed_With_Citation"

# Numeric input bounds (from cleaned dataset)
INPUT_BOUNDS = {
    "weekly_ai_hours": (0.0, 26.0, 5.4),
    "study_hours": (1.0, 26.0, 11.3),
    "anxiety_level": (1, 10, 4),
    "tool_diversity": (1, 5, 3),
    "skill_retention": (40.0, 100.0, 76.5),
    "gpa": (1.8, 4.0, 3.2),
}

# Burnout risk display styling
RISK_STYLES = {
    "Low": {"color": "#10b981", "bg": "#ecfdf5", "border": "#6ee7b7", "icon": "🟢"},
    "Medium": {"color": "#f59e0b", "bg": "#fffbeb", "border": "#fcd34d", "icon": "🟡"},
    "High": {"color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "icon": "🔴"},
}
