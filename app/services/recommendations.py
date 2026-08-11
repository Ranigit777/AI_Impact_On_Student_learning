"""Generate personalized wellness recommendations from prediction results."""

from app.config import RISK_STYLES


def generate_recommendations(inputs: dict, prediction: dict) -> list[dict]:
    """
    Build actionable recommendations based on risk level and input profile.

    Returns a list of dicts: {title, detail, priority}
    """
    risk = prediction["risk_level"]
    recommendations: list[dict] = []

    weekly_ai = inputs["weekly_ai_hours"]
    study_hours = inputs["study_hours"]
    anxiety = inputs["anxiety_level"]
    skill_retention = inputs["skill_retention"]
    paid_sub = inputs.get("paid_subscription", False)
    prompt_skill = inputs["prompt_skill"]

    if risk == "High":
        recommendations.append({
            "title": "Prioritize recovery and workload reduction",
            "detail": (
                "Your profile indicates elevated burnout risk. Reduce weekly AI reliance, "
                "schedule structured breaks, and speak with academic advising or counseling services."
            ),
            "priority": "high",
        })
    elif risk == "Medium":
        recommendations.append({
            "title": "Monitor stress signals closely",
            "detail": (
                "You are in a moderate-risk zone. Track sleep, mood, and focus weekly "
                "and adjust study habits before stress compounds."
            ),
            "priority": "medium",
        })
    else:
        recommendations.append({
            "title": "Maintain healthy study-AI balance",
            "detail": (
                "Your current habits appear sustainable. Keep balancing traditional study "
                "with mindful, purposeful AI use."
            ),
            "priority": "low",
        })

    if weekly_ai > 12:
        recommendations.append({
            "title": "Cap weekly GenAI usage",
            "detail": (
                f"You reported {weekly_ai:.1f} AI hours/week. Try limiting sessions to "
                "focused tasks and reserving at least 60% of study time for active learning."
            ),
            "priority": "high" if weekly_ai > 18 else "medium",
        })

    if study_hours < 6:
        recommendations.append({
            "title": "Increase offline study time",
            "detail": (
                "Low traditional study hours can reduce deep learning and increase AI dependency. "
                "Aim for at least 8–10 focused offline hours per week."
            ),
            "priority": "medium",
        })

    if anxiety >= 7:
        recommendations.append({
            "title": "Address exam anxiety proactively",
            "detail": (
                f"Anxiety level {anxiety}/10 is a strong burnout driver. Practice retrieval-based "
                "revision, mock exams, and mindfulness techniques before assessment periods."
            ),
            "priority": "high",
        })

    if skill_retention < 65:
        recommendations.append({
            "title": "Strengthen knowledge retention",
            "detail": (
                "Skill retention appears below the cohort median. Use spaced repetition, "
                "self-quizzing, and teaching concepts to peers without AI assistance."
            ),
            "priority": "medium",
        })

    if prompt_skill == "Beginner" and weekly_ai > 8:
        recommendations.append({
            "title": "Upgrade prompt engineering skills",
            "detail": (
                "Higher AI usage with beginner prompt skills can produce shallow learning. "
                "Complete a short prompt-engineering workshop to use AI as a tutor, not a shortcut."
            ),
            "priority": "medium",
        })

    if paid_sub and weekly_ai > 15:
        recommendations.append({
            "title": "Audit paid AI subscription value",
            "detail": (
                "Paid tools can encourage overuse. Set weekly usage budgets and evaluate whether "
                "each subscription meaningfully improves learning outcomes."
            ),
            "priority": "low",
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda item: priority_order[item["priority"]])
    return recommendations


def build_prediction_summary(inputs: dict, prediction: dict) -> dict:
    """Create a human-readable summary of the prediction."""
    risk = prediction["risk_level"]
    style = RISK_STYLES.get(risk, RISK_STYLES["Medium"])
    top_prob = max(prediction["probabilities"].items(), key=lambda item: item[1])

    return {
        "headline": f"{style['icon']} Predicted Burnout Risk: {risk}",
        "confidence_pct": round(prediction["confidence"] * 100, 1),
        "top_class": top_prob[0],
        "top_probability_pct": round(top_prob[1] * 100, 1),
        "risk_color": style["color"],
        "risk_bg": style["bg"],
        "risk_border": style["border"],
    }
