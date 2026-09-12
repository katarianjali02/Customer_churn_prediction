"""
predict.py
----------
Load the trained churn model and run predictions on new customer records.

Usage (as a script, predicts on a sample input built into this file):
    python src/predict.py

Usage (as a module):
    from predict import predict_churn
    predict_churn({"gender": "Female", "SeniorCitizen": 0, ...})
"""

import os
import joblib
import pandas as pd

from data_preprocessing import clean_data, encode_features, scale_numeric, TARGET_COL

MODELS_DIR = "models"


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_model.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, encoders, scaler, feature_columns


def predict_churn(customer: dict) -> dict:
    """
    Predict churn probability for a single customer, given as a dict of raw
    (unprocessed) feature values matching the original dataset's columns
    (excluding customerID and Churn).
    """
    model, encoders, scaler, feature_columns = load_artifacts()

    df = pd.DataFrame([customer])
    df = clean_data(df)

    X_encoded, _ = encode_features(df, fit_encoders=False, encoders=encoders)

    # Align columns with training-time feature set (handles missing one-hot columns)
    for col in feature_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[feature_columns]

    X_scaled, _ = scale_numeric(X_encoded, fit_scaler=False, scaler=scaler)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0,
        "TotalCharges": "450.0",
    }

    result = predict_churn(sample_customer)
    print("Sample prediction:", result)
