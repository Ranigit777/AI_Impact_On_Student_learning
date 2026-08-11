"""Run burnout risk predictions using the trained model package."""

import numpy as np

from app.services.input_builder import build_feature_dataframe
from app.services.model_loader import load_model_package


def predict_burnout(inputs: dict) -> dict:
    """
    Predict burnout risk level and class probabilities for a student profile.

    Returns
    -------
    dict with keys: risk_level, risk_class_idx, probabilities, feature_df, model_inputs
    """
    package = load_model_package()
    preprocessor = package["preprocessor"]
    label_encoder = package["label_encoder"]
    model = package["model"]

    feature_df = build_feature_dataframe(inputs)
    X = preprocessor.transform(feature_df)

    class_indices = model.predict(X)
    risk_class_idx = int(class_indices[0])
    risk_level = label_encoder.inverse_transform([risk_class_idx])[0]

  # Probability distribution across Low / Medium / High
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        proba = proba[0]
    else:
        proba = np.zeros(len(label_encoder.classes_))
        proba[risk_class_idx] = 1.0

    probabilities = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, proba)
    }

    confidence = float(probabilities[risk_level])

    return {
        "risk_level": risk_level,
        "risk_class_idx": risk_class_idx,
        "probabilities": probabilities,
        "confidence": confidence,
        "feature_df": feature_df,
        "target_names": label_encoder.classes_.tolist(),
    }
