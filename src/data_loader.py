import os
import pandas as pd
import numpy as np


class DataLoader:
    """
    DataLoader handles dataset loading, initial inspection, and reporting statistics.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from CSV file.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at path: {self.data_path}")

        print(f"Loading dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Successfully loaded dataset with shape: {df.shape}")
        return df

    def inspect_data(self, df: pd.DataFrame) -> dict:
        """
        Prints and returns key dataset characteristics:
        - Shape
        - Data types
        - Missing values
        - Duplicate rows
        - Summary statistics
        """
        print("\n" + "=" * 50)
        print(" DATASET INITIAL INSPECTION")
        print("=" * 50)

        shape = df.shape
        print(f"\n1. Dataset Shape: {shape[0]} rows, {shape[1]} columns")

        print("\n2. Data Types:")
        print(df.dtypes)

        print("\n3. Missing Values:")
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing Percentage (%)': missing_pct})
        print(missing_df[missing_df['Missing Count'] > 0] if (missing_counts > 0).any() else "No missing values found.")

        print("\n4. Duplicate Rows:")
        duplicate_count = df.duplicated().sum()
        print(f"Total duplicate rows: {duplicate_count}")

        print("\n5. Numerical Summary Statistics:")
        num_summary = df.describe().T
        print(num_summary)

        print("\n6. Categorical Summary Statistics:")
        cat_summary = df.describe(include=['object', 'category']).T if len(df.select_dtypes(include=['object', 'category']).columns) > 0 else "No categorical columns."
        print(cat_summary)

        inspection_report = {
            "shape": shape,
            "dtypes": df.dtypes.to_dict(),
            "missing_values": missing_counts.to_dict(),
            "duplicates": int(duplicate_count),
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary
        }
        return inspection_report
