import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    FeatureEngineer creates engineered features:
    - GPA Improvement: Post_Semester_GPA - Pre_Semester_GPA
    - AI Efficiency: GPA_Improvement / (Weekly_GenAI_Hours + 1.0)
    - Study Balance: Traditional_Study_Hours / (Weekly_GenAI_Hours + 1.0)
    - Dependency Index: (Perceived_AI_Dependency * Weekly_GenAI_Hours) / (Traditional_Study_Hours + 1.0)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_gpa_improvement(self) -> pd.DataFrame:
        """
        Calculates the change in GPA across the semester.
        """
        self.df['GPA_Improvement'] = self.df['Post_Semester_GPA'] - self.df['Pre_Semester_GPA']
        return self.df

    def create_ai_efficiency(self) -> pd.DataFrame:
        """
        Calculates GPA improvement per weekly GenAI hour spent.
        """
        self.df['AI_Efficiency'] = self.df['GPA_Improvement'] / (self.df['Weekly_GenAI_Hours'] + 1.0)
        return self.df

    def create_study_balance(self) -> pd.DataFrame:
        """
        Calculates the ratio of traditional study hours to weekly GenAI hours.
        """
        self.df['Study_Balance'] = self.df['Traditional_Study_Hours'] / (self.df['Weekly_GenAI_Hours'] + 1.0)
        return self.df

    def create_dependency_index(self) -> pd.DataFrame:
        """
        Calculates a composite index of AI dependency weighted by usage relative to traditional study.
        """
        self.df['Dependency_Index'] = (self.df['Perceived_AI_Dependency'] * self.df['Weekly_GenAI_Hours']) / (self.df['Traditional_Study_Hours'] + 1.0)
        return self.df

    def engineer_all_features(self) -> pd.DataFrame:
        """
        Executes all feature engineering transformations.
        """
        print("\n" + "=" * 50)
        print(" RUNNING FEATURE ENGINEERING")
        print("=" * 50)
        self.create_gpa_improvement()
        print("  [+] Created feature: GPA_Improvement")
        self.create_ai_efficiency()
        print("  [+] Created feature: AI_Efficiency")
        self.create_study_balance()
        print("  [+] Created feature: Study_Balance")
        self.create_dependency_index()
        print("  [+] Created feature: Dependency_Index")
        print(f"[Feature Engineer] Updated dataset shape: {self.df.shape}")
        return self.df
