import os
import pandas as pd
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.eda import EDAVisualizer


def main():
    print("=" * 65)
    print(" AI IMPACT ON STUDENT LEARNING - PHASE 1 PIPELINE EXECUTION")
    print("=" * 65)

    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "data", "ai_student_learning_raw.csv")
    cleaned_data_path = os.path.join(base_dir, "data", "cleaned_student_data.csv")
    figures_dir = os.path.join(base_dir, "reports", "figures")

    # Step 1: Load and inspect dataset
    loader = DataLoader(data_path=raw_data_path)
    raw_df = loader.load_data()
    loader.inspect_data(raw_df)

    # Step 2: Data cleaning pipeline
    cleaner = DataCleaner(df=raw_df)
    cleaned_df = cleaner.run_cleaning_pipeline(remove_outliers_flag=True)

    # Step 3: Feature engineering
    engineer = FeatureEngineer(df=cleaned_df)
    final_df = engineer.engineer_all_features()

    # Step 4: Save cleaned dataset
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    final_df.to_csv(cleaned_data_path, index=False)
    print(f"\n[Export] Cleaned & feature-engineered dataset successfully saved to:")
    print(f"  -> {cleaned_data_path}")
    print(f"  -> Final Dataset Shape: {final_df.shape[0]} rows, {final_df.shape[1]} columns")

    # Step 5: Exploratory Data Analysis (EDA)
    visualizer = EDAVisualizer(output_dir=figures_dir)
    visualizer.run_full_eda(final_df)

    print("\n" + "=" * 65)
    print(" PHASE 1 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
