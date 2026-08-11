import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """
    DataPreprocessor handles dataset splitting, encoding of categorical variables,
    scaling of numerical features, and target variable encoding.
    """

    def __init__(self, target_col: str = "Burnout_Risk_Level"):
        self.target_col = target_col
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame):
        """
        Fits preprocessor on full dataset features X and target y.
        Encodes target variable and applies OneHotEncoder + StandardScaler.
        """
        print("\n" + "=" * 50)
        print(" DATA PREPROCESSING & FEATURE SCALING")
        print("=" * 50)

        # Drop ID if present
        data = df.copy()
        if 'Student_ID' in data.columns:
            data = data.drop(columns=['Student_ID'])

        # Separate X and y
        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]

        # Target encoding (Low: 0, Medium: 1, High: 2)
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"[Preprocess] Target Variable '{self.target_col}' encoded classes:")
        for idx, cls_name in enumerate(self.label_encoder.classes_):
            print(f"  - Class {idx}: {cls_name}")

        # Identify numerical and categorical columns
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"[Preprocess] Identified {len(self.numerical_cols)} numerical features and {len(self.categorical_cols)} categorical features.")

        # Build ColumnTransformer
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

        # Fit & transform features
        X_transformed = self.preprocessor.fit_transform(X)

        # Retrieve feature names
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names = self.numerical_cols + encoded_cat_names

        print(f"[Preprocess] Processed feature matrix shape: {X_transformed.shape}")
        return X_transformed, y_encoded

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new raw features using fitted preprocessor.
        """
        data = df.copy()
        if 'Student_ID' in data.columns:
            data = data.drop(columns=['Student_ID'])
        if self.target_col in data.columns:
            data = data.drop(columns=[self.target_col])

        return self.preprocessor.transform(data)

    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
        """
        Splits features and target into stratified Train and Test sets.
        """
        print(f"[Preprocess] Splitting data into {int((1-test_size)*100)}% Train and {int(test_size*100)}% Test (Stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"  - Training Set: {X_train.shape[0]} samples")
        print(f"  - Testing Set : {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test
