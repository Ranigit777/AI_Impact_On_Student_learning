import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDAVisualizer:
    """
    EDAVisualizer generates comprehensive statistical figures for data analysis.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply modern aesthetic style
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['figure.titlesize'] = 16

    def plot_correlation_heatmap(self, df: pd.DataFrame, filename: str = "correlation_heatmap.png"):
        """Generates correlation heatmap for numerical features."""
        plt.figure(figsize=(12, 9))
        numeric_df = df.select_dtypes(include=[np.number])
        if 'Student_ID' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['Student_ID'])

        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title("Correlation Heatmap of Academic & AI Features", pad=20, weight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_burnout_risk_distribution(self, df: pd.DataFrame, filename: str = "burnout_risk_distribution.png"):
        """Plots distribution of target variable: Burnout Risk Level."""
        plt.figure(figsize=(8, 5))
        order = ['Low', 'Medium', 'High']
        counts = df['Burnout_Risk_Level'].value_counts()
        palette = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
        
        ax = sns.countplot(
            data=df, x='Burnout_Risk_Level', order=order,
            palette=palette, hue='Burnout_Risk_Level', legend=False
        )
        plt.title("Burnout Risk Level Distribution (Target Variable)", pad=15, weight='bold')
        plt.xlabel("Burnout Risk Level")
        plt.ylabel("Count")

        # Annotate bars with counts and percentages
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                pct = (height / total) * 100
                ax.annotate(f'{int(height)}\n({pct:.1f}%)',
                            (p.get_x() + p.get_width() / 2., height / 2),
                            ha='center', va='center', fontsize=11, color='white', weight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_ai_usage_distribution(self, df: pd.DataFrame, filename: str = "ai_usage_distribution.png"):
        """Plots weekly GenAI hours distribution across Burnout Risk Levels."""
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df, x='Weekly_GenAI_Hours', hue='Burnout_Risk_Level',
            hue_order=['Low', 'Medium', 'High'], palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
            kde=True, element="step", common_norm=False
        )
        plt.title("Weekly GenAI Usage Hours Distribution by Burnout Risk", pad=15, weight='bold')
        plt.xlabel("Weekly GenAI Hours")
        plt.ylabel("Density / Count")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_gpa_distribution(self, df: pd.DataFrame, filename: str = "gpa_distribution.png"):
        """Plots pre vs post semester GPA distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(df['Pre_Semester_GPA'], kde=True, ax=axes[0], color='#3498db', bins=25)
        axes[0].set_title("Pre-Semester GPA Distribution", weight='bold')
        axes[0].set_xlabel("Pre-Semester GPA")
        
        sns.histplot(df['Post_Semester_GPA'], kde=True, ax=axes[1], color='#9b59b6', bins=25)
        axes[1].set_title("Post-Semester GPA Distribution", weight='bold')
        axes[1].set_xlabel("Post-Semester GPA")

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_study_hours_distribution(self, df: pd.DataFrame, filename: str = "study_hours_distribution.png"):
        """Plots Traditional Study Hours vs GenAI Study Hours."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Traditional_Study_Hours', y='Weekly_GenAI_Hours',
            hue='Burnout_Risk_Level', hue_order=['Low', 'Medium', 'High'],
            palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
            alpha=0.6, s=50
        )
        plt.title("Traditional Study Hours vs Weekly GenAI Hours", pad=15, weight='bold')
        plt.xlabel("Traditional Study Hours")
        plt.ylabel("Weekly GenAI Hours")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_anxiety_distribution(self, df: pd.DataFrame, filename: str = "anxiety_distribution.png"):
        """Plots Exam Anxiety Level breakdown by Burnout Risk Level."""
        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=df, x='Burnout_Risk_Level', y='Anxiety_Level_During_Exams',
            order=['Low', 'Medium', 'High'], palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
        )
        plt.title("Exam Anxiety Level by Burnout Risk Level", pad=15, weight='bold')
        plt.xlabel("Burnout Risk Level")
        plt.ylabel("Exam Anxiety Level (1-10)")
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_histograms(self, df: pd.DataFrame, filename: str = "histograms.png"):
        """Plots histograms for all numerical features."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if 'Student_ID' in num_cols:
            num_cols = num_cols.drop('Student_ID')

        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(num_cols):
            sns.histplot(df[col], kde=True, ax=axes[idx], color='#1f77b4', bins=20)
            axes[idx].set_title(f"Histogram: {col}", weight='bold')

        # Hide extra subplots
        for idx in range(len(num_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_boxplots(self, df: pd.DataFrame, filename: str = "boxplots.png"):
        """Plots boxplots for numerical features to visualize outliers."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        if 'Student_ID' in num_cols:
            num_cols = num_cols.drop('Student_ID')

        n_cols = 3
        n_rows = (len(num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(num_cols):
            sns.boxplot(y=df[col], ax=axes[idx], color='#e74c3c')
            axes[idx].set_title(f"Boxplot: {col}", weight='bold')

        for idx in range(len(num_cols), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_pairplots(self, df: pd.DataFrame, filename: str = "pairplots.png"):
        """Plots pairplots of key features colored by Burnout Risk Level."""
        key_cols = [
            'Pre_Semester_GPA', 'Post_Semester_GPA', 'Weekly_GenAI_Hours',
            'Traditional_Study_Hours', 'Anxiety_Level_During_Exams', 'Burnout_Risk_Level'
        ]
        sub_df = df[key_cols].dropna()
        if len(sub_df) > 2000:
            sub_df = sub_df.sample(n=2000, random_state=42)

        g = sns.pairplot(
            sub_df, hue='Burnout_Risk_Level',
            palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
            corner=True, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 20}
        )
        g.fig.suptitle("Pairplot of Key Academic & Wellbeing Features", y=1.02, weight='bold')
        save_path = os.path.join(self.output_dir, filename)
        g.savefig(save_path, dpi=200)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def plot_countplots(self, df: pd.DataFrame, filename: str = "countplots.png"):
        """Plots countplots for categorical features."""
        cat_cols = ['Major_Category', 'Year_of_Study', 'Primary_Use_Case',
                    'Prompt_Engineering_Skill', 'Institutional_Policy']

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.flatten()

        for idx, col in enumerate(cat_cols):
            if col in df.columns:
                sns.countplot(
                    data=df, x=col, hue='Burnout_Risk_Level',
                    palette={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'},
                    ax=axes[idx]
                )
                axes[idx].set_title(f"Countplot: {col}", weight='bold')
                axes[idx].tick_params(axis='x', rotation=30)

        # Remove extra subplot
        fig.delaxes(axes[-1])

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [+] Saved EDA plot: {save_path}")

    def run_full_eda(self, df: pd.DataFrame):
        """Executes all EDA plotting functions."""
        print("\n" + "=" * 50)
        print(" RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 50)
        self.plot_correlation_heatmap(df)
        self.plot_burnout_risk_distribution(df)
        self.plot_ai_usage_distribution(df)
        self.plot_gpa_distribution(df)
        self.plot_study_hours_distribution(df)
        self.plot_anxiety_distribution(df)
        self.plot_histograms(df)
        self.plot_boxplots(df)
        self.plot_pairplots(df)
        self.plot_countplots(df)
        print(f"[EDA] All figures successfully saved to '{self.output_dir}'")
