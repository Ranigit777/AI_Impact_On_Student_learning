import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance


class FeatureImportanceAnalyzer:
    """
    FeatureImportanceAnalyzer extracts and plots feature importance for the best model.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def analyze(self, model, feature_names: list, X_test: np.ndarray = None, y_test: np.ndarray = None,
                top_n: int = 15, filename: str = "feature_importance.png") -> pd.DataFrame:
        """
        Extracts feature importances and exports a horizontal bar plot.
        """
        print("\n" + "=" * 65, flush=True)
        print(" FEATURE IMPORTANCE ANALYSIS", flush=True)
        print("=" * 65, flush=True)

        importances = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            print("  [+] Extracted tree-based feature_importances_.", flush=True)
        elif hasattr(model, "coef_"):
            importances = np.mean(np.abs(model.coef_), axis=0)
            print("  [+] Extracted coefficient weights from linear model.", flush=True)
        elif X_test is not None and y_test is not None:
            print("  [+] Model does not natively expose importances. Computing Permutation Importance...", flush=True)
            perm_result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            importances = perm_result.importances_mean

        if importances is None:
            print("  [-] Unable to extract feature importances for this model type.", flush=True)
            return pd.DataFrame()

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        print(f"\nTop {top_n} Most Important Features:", flush=True)
        print(fi_df.head(top_n).to_string(index=False), flush=True)

        # Plot Top N Features
        plt.figure(figsize=(12, 7))
        top_fi = fi_df.head(top_n)
        ax = sns.barplot(
            data=top_fi, x="Importance", y="Feature",
            hue="Feature", palette="viridis", legend=False
        )
        plt.title(f"Top {top_n} Feature Importances for Burnout Risk Prediction", pad=15, weight='bold')
        plt.xlabel("Importance Score")
        plt.ylabel("Feature Name")

        # Annotate importance scores
        for p in ax.patches:
            w = p.get_width()
            if w > 0:
                ax.annotate(f"{w:.4f}", (w, p.get_y() + p.get_height() / 2.),
                            ha='left', va='center', fontsize=9, xytext=(5, 0),
                            textcoords='offset points')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n  [+] Saved Feature Importance Chart: {save_path}")

        return fi_df
