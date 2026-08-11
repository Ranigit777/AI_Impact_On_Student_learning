import os
import joblib
import pandas as pd
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.hyperparameter_tuner import HyperparameterTuner
from src.feature_importance import FeatureImportanceAnalyzer


def main():
    print("=" * 70, flush=True)
    print(" AI IMPACT ON STUDENT LEARNING - PHASE 2 MODELING & EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_data_path = os.path.join(base_dir, "data", "cleaned_student_data.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Step 1: Load Cleaned Dataset
    print(f"\n[Load] Loading cleaned dataset from: {cleaned_data_path}", flush=True)
    df = pd.read_csv(cleaned_data_path)
    print(f"[Load] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

    # Step 2: Preprocessing (Categorical Encoding, Numerical Scaling, Train-Test Split)
    preprocessor = DataPreprocessor(target_col="Burnout_Risk_Level")
    X_proc, y_proc = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_proc, y_proc, test_size=0.2, random_state=42)

    target_names = preprocessor.label_encoder.classes_.tolist()
    feature_names = preprocessor.feature_names

    # Step 3: Train and Evaluate Baseline 7 Models
    trainer = ModelTrainer(output_dir=figures_dir)
    results_df, fitted_models, confusion_matrices, classification_reports = trainer.evaluate_models(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        target_names=target_names
    )

    # Step 4: Model Selection
    best_model_name = results_df.iloc[0]["Model"]
    best_base_model = fitted_models[best_model_name]
    best_f1 = results_df.iloc[0]["F1 Score (Weighted)"]

    print("\n" + "=" * 70)
    print(f" WINNING BASELINE MODEL SELECTED: '{best_model_name}' (F1 Score: {best_f1:.4f})")
    print("=" * 70)

    # Step 5: Hyperparameter Tuning via GridSearchCV
    tuner = HyperparameterTuner(best_model_name=best_model_name, base_model=best_base_model)
    tuned_model, tuned_metrics = tuner.tune(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        cv=5, scoring="f1_weighted"
    )

    # Step 6: Feature Importance Analysis
    fi_analyzer = FeatureImportanceAnalyzer(output_dir=figures_dir)
    fi_df = fi_analyzer.analyze(
        model=tuned_model,
        feature_names=feature_names,
        X_test=X_test,
        y_test=y_test,
        top_n=15,
        filename="feature_importance.png"
    )

    # Step 7: Save Trained Model Package
    model_save_path = os.path.join(models_dir, "burnout_prediction_model.pkl")
    model_package = {
        "preprocessor": preprocessor.preprocessor,
        "label_encoder": preprocessor.label_encoder,
        "model": tuned_model,
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "target_names": target_names,
        "tuned_metrics": tuned_metrics
    }
    joblib.dump(model_package, model_save_path)
    print(f"\n[Export] Trained Model Package saved to:")
    print(f"  -> {model_save_path}")

    # Step 8: Generate Detailed Model Evaluation Report
    report_path = os.path.join(reports_dir, "model_evaluation_report.md")
    generate_markdown_report(
        report_path=report_path,
        results_df=results_df,
        best_model_name=best_model_name,
        tuned_metrics=tuned_metrics,
        classification_reports=classification_reports,
        fi_df=fi_df
    )
    print(f"  -> Evaluation Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print(" PHASE 2 MODELING PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 70)


def generate_markdown_report(report_path, results_df, best_model_name, tuned_metrics, classification_reports, fi_df):
    """Generates comprehensive markdown report for Phase 2."""
    with open(report_path, "w") as f:
        f.write("# Model Evaluation & Selection Report\n\n")
        f.write("**Project Title**: AI Impact on Student Learning – Burnout Risk Prediction\n")
        f.write("**Phase**: Phase 2 - Model Training, Evaluation & Selection\n\n")

        f.write("## 1. Model Comparison Summary Table\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n")

        f.write(f"## 2. Best Performing Model: `{best_model_name}`\n\n")
        f.write(f"- **Baseline F1 Score**: `{results_df.iloc[0]['F1 Score (Weighted)']:.4f}`\n")
        f.write(f"- **Tuned F1 Score**: `{tuned_metrics['F1 Score (Weighted)']:.4f}`\n")
        f.write(f"- **Best Parameters**: `{tuned_metrics['Best Parameters']}`\n\n")

        f.write("### Classification Report for Best Model\n\n")
        f.write("```text\n")
        f.write(classification_reports[best_model_name])
        f.write("```\n\n")

        f.write("## 3. Top 15 Feature Importances\n\n")
        if not fi_df.empty:
            f.write(fi_df.head(15).to_markdown(index=False))
        f.write("\n\n")

        f.write("## 4. Model Artifact Saved\n\n")
        f.write("- Trained Pipeline artifact path: `models/burnout_prediction_model.pkl`\n")


if __name__ == "__main__":
    main()
