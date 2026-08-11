"""
Automated tests for the Burnout Risk Prediction pipeline.

Run from the project root:
    python tests/test_prediction.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_loader import get_feature_importance, load_model_package
from app.services.predictor import predict_burnout


SAMPLE_INPUTS = {
    "major_category": "STEM",
    "year_of_study": "Junior",
    "weekly_ai_hours": 10.0,
    "study_hours": 12.0,
    "anxiety_level": 5,
    "prompt_skill": "Intermediate",
    "paid_subscription": False,
    "tool_diversity": 3,
    "skill_retention": 75.0,
    "gpa": 3.2,
}


def test_model_file_exists():
    model_path = PROJECT_ROOT / "models" / "burnout_prediction_model.pkl"
    assert model_path.exists(), f"Model not found at {model_path}. Run: python main_phase2.py"


def test_model_package_loads():
    package = load_model_package()
    assert "model" in package
    assert "preprocessor" in package
    assert "label_encoder" in package
    assert len(package["feature_names"]) == 33


def test_prediction_returns_valid_output():
    result = predict_burnout(SAMPLE_INPUTS)
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == {"Low", "Medium", "High"}
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}, expected ~1.0"


def test_feature_importance_available():
    importance = get_feature_importance()
    assert len(importance) > 0
    assert "feature" in importance[0]
    assert "importance" in importance[0]
    assert importance[0]["importance"] >= importance[-1]["importance"]


def test_high_risk_profile():
    """High AI usage + high anxiety should tend toward higher burnout risk."""
    high_risk_inputs = {
        **SAMPLE_INPUTS,
        "weekly_ai_hours": 22.0,
        "study_hours": 4.0,
        "anxiety_level": 9,
        "skill_retention": 45.0,
        "gpa": 2.2,
    }
    result = predict_burnout(high_risk_inputs)
    assert result["risk_level"] in {"Medium", "High"}


if __name__ == "__main__":
    tests = [
        test_model_file_exists,
        test_model_package_loads,
        test_prediction_returns_valid_output,
        test_feature_importance_available,
        test_high_risk_profile,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print(" BURNOUT RISK PREDICTION – TEST SUITE")
    print("=" * 60)

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1

    print("=" * 60)
    print(f" Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
