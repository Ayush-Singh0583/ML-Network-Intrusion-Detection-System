from sklearn.linear_model import LogisticRegression

import joblib


def train_logistic(X_train, y_train):

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, scaler, encoder):

    joblib.dump(model, "models/logistic_model.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    joblib.dump(encoder, "models/label_encoder.pkl")