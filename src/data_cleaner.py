import pandas as pd
import numpy as np


class DataCleaner:
    """
    DataCleaner performs data cleaning operations:
    - Missing value imputation
    - Duplicate removal
    - Data type correction
    - Outlier detection and removal via IQR
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Imputes missing values:
        - Numerical columns: Median
        - Categorical columns: Mode
        """
        initial_missing = self.df.isnull().sum().sum()
        if initial_missing == 0:
            print("[Clean] No missing values found.")
            return self.df

        print(f"[Clean] Handling {initial_missing} missing values...")
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    print(f"  - Imputed missing in numerical '{col}' with median: {median_val}")
                else:
                    mode_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_val)
                    print(f"  - Imputed missing in categorical '{col}' with mode: {mode_val}")

        return self.df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Identifies and drops duplicate rows.
        """
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            print(f"[Clean] Removed {dup_count} duplicate rows.")
        else:
            print("[Clean] No duplicate rows found.")
        return self.df

    def correct_data_types(self) -> pd.DataFrame:
        """
        Ensures appropriate data types for categorical and numerical features.
        """
        print("[Clean] Correcting column data types...")
        categorical_cols = [
            'Major_Category', 'Year_of_Study', 'Primary_Use_Case',
            'Prompt_Engineering_Skill', 'Paid_Subscription',
            'Institutional_Policy', 'Burnout_Risk_Level'
        ]

        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')

        # Numeric conversions
        numeric_cols = [
            'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Tool_Diversity',
            'Traditional_Study_Hours', 'Perceived_AI_Dependency',
            'Anxiety_Level_During_Exams', 'Post_Semester_GPA', 'Skill_Retention_Score'
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        return self.df

    def detect_and_remove_outliers(self, factor: float = 1.5, columns: list = None) -> pd.DataFrame:
        """
        Detects and removes outliers using the Interquartile Range (IQR) method.
        """
        if columns is None:
            columns = [
                'Weekly_GenAI_Hours', 'Traditional_Study_Hours',
                'Pre_Semester_GPA', 'Post_Semester_GPA',
                'Skill_Retention_Score'
            ]

        initial_rows = len(self.df)
        outlier_indices = set()

        print(f"[Clean] Detecting outliers using IQR (factor={factor})...")
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                col_outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].index
                outlier_count = len(col_outliers)
                if outlier_count > 0:
                    print(f"  - Column '{col}': found {outlier_count} outliers (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
                    outlier_indices.update(col_outliers)

        if outlier_indices:
            self.df = self.df.drop(index=list(outlier_indices)).reset_index(drop=True)
            removed_rows = initial_rows - len(self.df)
            print(f"[Clean] Removed total {removed_rows} outlier rows ({removed_rows / initial_rows * 100:.2f}% of dataset).")
        else:
            print("[Clean] No outliers detected outside boundaries.")

        return self.df

    def run_cleaning_pipeline(self, remove_outliers_flag: bool = True) -> pd.DataFrame:
        """
        Executes the full cleaning pipeline.
        """
        print("\n" + "=" * 50)
        print(" RUNNING DATA CLEANING PIPELINE")
        print("=" * 50)
        self.handle_missing_values()
        self.remove_duplicates()
        self.correct_data_types()
        if remove_outliers_flag:
            self.detect_and_remove_outliers()
        print(f"[Clean] Final cleaned dataset shape: {self.df.shape}")
        return self.df
