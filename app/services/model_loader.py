"""Load and cache the trained burnout prediction model package."""

from functools import lru_cache

import joblib
import numpy as np

from app.config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model_package() -> dict:
    """Load the serialized model artifact from disk (cached)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python main_phase2.py` to train and export the model first."
        )
    return joblib.load(MODEL_PATH)


def get_feature_importance() -> list[dict]:
    """Extract feature importance scores from the trained model."""
    package = load_model_package()
    model = package["model"]
    feature_names = package["feature_names"]

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.mean(np.abs(model.coef_), axis=0)
    else:
        return []

    importance = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(feature_names, scores)
    ]
    importance.sort(key=lambda item: item["importance"], reverse=True)
    return importance
