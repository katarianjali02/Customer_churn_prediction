"""
data_preprocessing.py
----------------------
Handles loading and cleaning of the Telco Customer Churn dataset.

Steps performed:
1. Load raw CSV
2. Drop identifier column (customerID)
3. Fix TotalCharges (stored as string, has blank values for new customers)
4. Encode target variable (Churn -> 0/1)
5. Encode categorical features (Label Encoding for binary, One-Hot for multi-class)
6. Scale numerical features
7. Split into train/test sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

RAW_DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
PROCESSED_DATA_DIR = "data"
MODELS_DIR = "models"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw churn dataset from CSV."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe: fix types, drop unneeded columns, handle missing values."""
    df = df.copy()

    # customerID is just an identifier, not predictive
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has some blank strings for customers with 0 tenure -> convert & fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Standardize the SeniorCitizen column to a readable categorical (0/1 already fine as-is)
    return df


def encode_features(df: pd.DataFrame, fit_encoders: bool = True, encoders: dict = None):
    """
    Encode categorical variables.
    Binary categorical columns -> Label Encoding
    Multi-class categorical columns -> One-Hot Encoding
    Returns the encoded dataframe and the dictionary of fitted encoders (for reuse at inference time).
    """
    df = df.copy()
    encoders = encoders or {}

    categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)

    if fit_encoders:
        # Determine binary vs multi-class columns from the training data's cardinality
        binary_cols = [c for c in categorical_cols if df[c].nunique() == 2]
        multi_cols = [c for c in categorical_cols if df[c].nunique() > 2]
    else:
        # At inference time a single row (or small batch) may not reflect true
        # cardinality, so rely on which columns were fitted with a LabelEncoder.
        binary_cols = [c for c in categorical_cols if c in encoders]
        multi_cols = [c for c in categorical_cols if c not in encoders]

    # Label encode binary columns
    for col in binary_cols:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col])

    # One-hot encode multi-class columns
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df, encoders


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode target Churn column: Yes -> 1, No -> 0."""
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def scale_numeric(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None):
    """Scale numeric columns with StandardScaler."""
    df = df.copy()
    if fit_scaler:
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df, scaler


def preprocess_pipeline(path: str = RAW_DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline: load -> clean -> encode -> scale -> split.
    Saves the fitted encoders/scaler/feature columns to models/ so the same
    transformations can be applied to new data at inference time.
    """
    df = load_raw_data(path)
    df = clean_data(df)
    df = encode_target(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_encoded, encoders = encode_features(X, fit_encoders=True)
    X_scaled, scaler = scale_numeric(X_encoded, fit_scaler=True)

    feature_columns = X_scaled.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Churn rate (train):", y_train.mean().round(3))
