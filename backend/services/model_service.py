import joblib
import numpy as np
import pandas as pd

from backend.config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_PATH
)

print("Loading trained model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_PATH)

print("Model loaded successfully.")


# ==========================================
# SINGLE PREDICTION
# ==========================================

def predict(features: dict):

    # Convert JSON to DataFrame
    df = pd.DataFrame([features])

    # Remove spaces in column names
    df.columns = df.columns.str.strip()

    # Keep only training columns
    df = df[feature_names]

    # Scale
    X = scaler.transform(df)

    # Predict
    prediction = model.predict(X)

    # Decode label
    prediction = encoder.inverse_transform(prediction)

    return prediction[0]


# ==========================================
# CSV PREDICTION
# ==========================================

def predict_dataframe(df):

    # -----------------------------
    # Clean column names
    # -----------------------------
    df.columns = df.columns.str.strip()

    # -----------------------------
    # Remove identifier columns
    # -----------------------------
    identifier_columns = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Label"
    ]

    df.drop(
        columns=identifier_columns,
        errors="ignore",
        inplace=True
    )

    # -----------------------------
    # Remove invalid values
    # -----------------------------
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(inplace=True)

    # -----------------------------
    # Arrange columns exactly like training
    # -----------------------------
    df = df[feature_names]

    # -----------------------------
    # Scale
    # -----------------------------
    X = scaler.transform(df)

    # -----------------------------
    # Predict
    # -----------------------------
    predictions = model.predict(X)

    # Decode predictions
    predictions = encoder.inverse_transform(predictions)

    # -----------------------------
    # Save prediction column
    # -----------------------------
    df["Prediction"] = predictions

    # -----------------------------
    # Save CSV
    # -----------------------------
    df.to_csv(
        "predicted_output.csv",
        index=False
    )

    # -----------------------------
    # Prediction Summary
    # -----------------------------
    summary = (
        df["Prediction"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_records": len(df),
        "summary": summary
    }