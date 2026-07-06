from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import joblib


def train_logistic(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

import joblib
import os



def save_model(model, scaler, encoder, feature_names):

    print("\n========== SAVING FILES ==========")

    print("Current Working Directory:")
    print(os.getcwd())

    # Save Random Forest Model
    joblib.dump(
        model,
        "models/random_forest_model.pkl"
    )
    print("Random Forest Model Saved")

    # Save Scaler
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )
    print("Scaler Saved")

    # Save Label Encoder
    joblib.dump(
        encoder,
        "models/label_encoder.pkl"
    )
    print("Label Encoder Saved")

    # Save Feature Names
    joblib.dump(
        feature_names,
        "models/feature_names.pkl"
    )
    print("Feature Names Saved")

    print("\nFiles currently inside models folder:\n")

    for file in os.listdir("models"):
        print(file)

    print("\n========== SAVE COMPLETE ==========\n")