import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class ModelTrainer:
    """
    ModelTrainer trains 7 candidate ML classifiers, performs 5-fold cross-validation,
    evaluates test set metrics, plots confusion matrices, and selects the winning model.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def get_models(self) -> dict:
        """
        Initializes the 7 mandatory baseline classification algorithms.
        """
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            "Support Vector Machine": LinearSVC(dual=False, max_iter=2000, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=-1),
            "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss', n_jobs=-1)
        }
        return models

    def evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        target_names: list = None) -> tuple:
        """
        Trains and evaluates all models. Computes Accuracy, Precision, Recall, F1 Score,
        5-fold Cross-Validation score, Confusion Matrix, and Classification Reports.
        """
        if target_names is None:
            target_names = ["Low", "Medium", "High"]

        models = self.get_models()
        results = []
        fitted_models = {}
        confusion_matrices = {}
        classification_reports = {}

        cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        print("\n" + "=" * 65, flush=True)
        print(" TRAINING & EVALUATING CLASSIFICATION MODELS", flush=True)
        print("=" * 65, flush=True)

        for name, model in models.items():
            print(f"\n[Model] Training '{name}'...", flush=True)
            start_time = time.time()

            # 5-Fold Cross-Validation on Train set
            if name == "K-Nearest Neighbors" and len(X_train) > 10000:
                cv_scores = cross_val_score(model, X_train[:10000], y_train[:10000], cv=cv_fold, scoring='f1_weighted', n_jobs=1)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_fold, scoring='f1_weighted', n_jobs=1)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Fit on full Train set
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            fitted_models[name] = model

            # Test Set Predictions
            y_pred = model.predict(X_test)

            # Compute Evaluation Metrics (Weighted for multi-class)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            clf_rep = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

            confusion_matrices[name] = cm
            classification_reports[name] = classification_report(y_test, y_pred, target_names=target_names)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision (Weighted)": prec,
                "Recall (Weighted)": rec,
                "F1 Score (Weighted)": f1,
                "5-Fold CV F1 (Mean)": cv_mean,
                "5-Fold CV F1 (Std)": cv_std,
                "Training Time (s)": train_time
            })

            print(f"  -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f} | CV F1: {cv_mean:.4f} (±{cv_std:.4f})")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results).sort_values(by="F1 Score (Weighted)", ascending=False).reset_index(drop=True)

        print("\n" + "=" * 65)
        print(" MODEL BENCHMARK COMPARISON TABLE")
        print("=" * 65)
        print(results_df.to_string(index=False))

        # Save Visualizations
        self.plot_model_comparison(results_df)
        self.plot_confusion_matrices(confusion_matrices, target_names)

        return results_df, fitted_models, confusion_matrices, classification_reports

    def plot_model_comparison(self, results_df: pd.DataFrame, filename: str = "model_comparison_metrics.png"):
        """Plots grouped bar plot of model metrics."""
        plt.figure(figsize=(14, 7))
        melted_df = results_df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1 Score (Weighted)", "5-Fold CV F1 (Mean)"],
            var_name="Metric", value_name="Score"
        )
        
        ax = sns.barplot(
            data=melted_df, x="Model", y="Score", hue="Metric",
            palette="Set2"
        )
        plt.title("Classification Model Performance Metrics Comparison", pad=15, weight='bold')
        plt.xlabel("Algorithm")
        plt.ylabel("Score (0.0 - 1.0)")
        plt.ylim(0.0, 1.05)
        plt.xticks(rotation=20)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # Add data values above bars
        for p in ax.patches:
            h = p.get_height()
            if h > 0.1:
                ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8, rotation=90, xytext=(0, 3),
                            textcoords='offset points')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved Comparison Chart: {save_path}")

    def plot_confusion_matrices(self, cm_dict: dict, target_names: list, filename: str = "confusion_matrices.png"):
        """Plots confusion matrices for all 7 algorithms in a 3x3 subplot grid."""
        n_models = len(cm_dict)
        fig, axes = plt.subplots(3, 3, figsize=(16, 15))
        axes = axes.flatten()

        for idx, (name, cm) in enumerate(cm_dict.items()):
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=target_names, yticklabels=target_names, cbar=False
            )
            axes[idx].set_title(f"{name}", weight='bold')
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("True Label")

        # Hide empty subplot grid slots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle("Confusion Matrix Grid for All 7 Classifiers", fontsize=18, weight='bold', y=0.99)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved Confusion Matrix Grid: {save_path}")
