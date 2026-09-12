import joblib


def load_saved_model():

    model = joblib.load("models/logistic_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    encoder = joblib.load("models/label_encoder.pkl")

    return model, scaler, encoder