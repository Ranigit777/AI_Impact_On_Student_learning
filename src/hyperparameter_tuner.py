import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class HyperparameterTuner:
    """
    HyperparameterTuner performs GridSearchCV hyperparameter optimization on the best performing model.
    """

    def __init__(self, best_model_name: str, base_model, param_grid: dict = None):
        self.best_model_name = best_model_name
        self.base_model = base_model
        self.param_grid = param_grid if param_grid is not None else self._default_param_grid(best_model_name)

    def _default_param_grid(self, model_name: str) -> dict:
        """
        Returns default hyperparameter search space for key algorithms.
        """
        if "Random Forest" in model_name:
            return {
                "n_estimators": [100, 150],
                "max_depth": [12, 20, None],
                "min_samples_split": [2, 5]
            }
        elif "XGBoost" in model_name:
            return {
                "n_estimators": [100, 150],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1]
            }
        elif "Gradient Boosting" in model_name:
            return {
                "max_iter": [100, 150],
                "max_depth": [6, 10, None],
                "learning_rate": [0.05, 0.1]
            }
        elif "Decision Tree" in model_name:
            return {
                "max_depth": [8, 12, 20, None],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"]
            }
        elif "Logistic Regression" in model_name:
            return {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        elif "K-Nearest Neighbors" in model_name:
            return {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            }
        else:
            return {}

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             cv: int = 3, scoring: str = "f1_weighted") -> tuple:
        """
        Executes GridSearchCV to find optimal hyperparameters.
        """
        print("\n" + "=" * 65, flush=True)
        print(f" HYPERPARAMETER TUNING (GridSearchCV) FOR: {self.best_model_name}", flush=True)
        print("=" * 65, flush=True)

        if not self.param_grid:
            print("[Tuner] No hyperparameter grid specified. Returning base model.", flush=True)
            self.base_model.fit(X_train, y_train)
            return self.base_model, {}

        print(f"[Tuner] Search Space: {self.param_grid}", flush=True)
        print(f"[Tuner] Running GridSearchCV (cv={cv}, scoring='{scoring}')...", flush=True)

        grid_search = GridSearchCV(
            estimator=self.base_model,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print("\n[Tuner] Optimization Complete!", flush=True)
        print(f"  - Best Hyperparameters: {best_params}", flush=True)
        print(f"  - Best CV Weighted F1: {best_cv_score:.4f}", flush=True)

        # Evaluate tuned model on test set
        y_pred_tuned = best_model.predict(X_test)
        tuned_acc = accuracy_score(y_test, y_pred_tuned)
        tuned_prec = precision_score(y_test, y_pred_tuned, average='weighted', zero_division=0)
        tuned_rec = recall_score(y_test, y_pred_tuned, average='weighted', zero_division=0)
        tuned_f1 = f1_score(y_test, y_pred_tuned, average='weighted', zero_division=0)

        tuned_metrics = {
            "Tuned Model": self.best_model_name + " (Tuned)",
            "Best Parameters": best_params,
            "Accuracy": tuned_acc,
            "Precision (Weighted)": tuned_prec,
            "Recall (Weighted)": tuned_rec,
            "F1 Score (Weighted)": tuned_f1,
            "Best CV Score": best_cv_score
        }

        print("\n[Tuner] Tuned Model Test Performance:", flush=True)
        print(f"  -> Accuracy: {tuned_acc:.4f} | Precision: {tuned_prec:.4f} | Recall: {tuned_rec:.4f} | F1 Score: {tuned_f1:.4f}", flush=True)

        return best_model, tuned_metrics
