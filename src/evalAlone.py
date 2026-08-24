import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

# Ensure project root and src/ are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing import (
    load_all_datasets,
    clean_dataset,
    remove_identifier_columns,
    split_features_target,
    split_dataset
)
from evaluation import evaluate_model, feature_importance


def get_default_paths():
    return {
        "data_dir": os.path.join(PROJECT_ROOT, "data"),
        "models_dir": os.path.join(PROJECT_ROOT, "models"),
        "default_model": os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"),
    }


def load_artifacts(models_dir, model_path):
    print("\n========== LOADING ARTIFACTS ==========")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    scaler_path = os.path.join(models_dir, "scaler.pkl")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    for path, name in [(scaler_path, "Scaler"), (encoder_path, "Label Encoder"), (features_path, "Feature Names")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at: {path}")

    print(f"Loading Model          : {os.path.basename(model_path)}")
    model = joblib.load(model_path)

    print("Loading Scaler         : scaler.pkl")
    scaler = joblib.load(scaler_path)

    print("Loading Label Encoder  : label_encoder.pkl")
    encoder = joblib.load(encoder_path)

    print("Loading Feature Names  : feature_names.pkl")
    feature_names = joblib.load(features_path)

    print("Artifacts loaded successfully!")
    return model, scaler, encoder, feature_names


def plot_and_save_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Label values in matrix
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix plot saved to: {output_path}")


def evaluate_alone(
    model_path=None,
    data_dir=None,
    models_dir=None,
    sample_size=None
):
    defaults = get_default_paths()
    data_dir = data_dir or defaults["data_dir"]
    models_dir = models_dir or defaults["models_dir"]
    model_path = model_path or defaults["default_model"]

    # 1. Load artifacts
    model, scaler, encoder, feature_names = load_artifacts(models_dir, model_path)

    # 2. Load & preprocess dataset
    print(f"\n========== PREPARING TEST DATA FROM '{data_dir}' ==========")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    df = load_all_datasets(data_dir)
    df = clean_dataset(df)
    df = remove_identifier_columns(df)

    if sample_size and sample_size < len(df):
        print(f"\nSampling {sample_size} rows for rapid evaluation...")
        df = df.sample(n=sample_size, random_state=42)

    X, y = split_features_target(df)

    # Align columns with feature_names expected by the model
    missing_cols = set(feature_names) - set(X.columns)
    if missing_cols:
        print(f"Warning: {len(missing_cols)} expected features missing in data. Filling with 0.")
        for col in missing_cols:
            X[col] = 0
    X = X[feature_names]

    # Encode labels using the saved encoder
    y_encoded = encoder.transform(y)

    # Split to get the standard test set (20% split with seed 42)
    _, X_test, _, y_test = split_dataset(X, y_encoded)

    # Scale features using saved scaler
    X_test_scaled = scaler.transform(X_test)

    print(f"\nEvaluating on {len(X_test_scaled)} test samples with {len(feature_names)} features...")

    # 3. Perform model evaluation
    y_pred = evaluate_model(model, X_test_scaled, y_test)

    # 4. Generate & Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, encoder.classes_, cm_plot_path)

    # 5. Feature Importance (if applicable)
    if hasattr(model, "feature_importances_"):
        feature_importance(model, feature_names)
    elif hasattr(model, "coef_"):
        print("\nModel uses linear coefficients instead of tree-based feature_importances_.")

    print("\n========================================")
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("========================================\n")


def main():
    defaults = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Standalone Model Evaluation for Network Intrusion Detection System"
    )
    parser.add_argument(
        "--model",
        default=defaults["default_model"],
        help=f"Path to trained model .pkl (Default: {defaults['default_model']})"
    )
    parser.add_argument(
        "--data_dir",
        default=defaults["data_dir"],
        help=f"Directory containing CSV datasets (Default: {defaults['data_dir']})"
    )
    parser.add_argument(
        "--models_dir",
        default=defaults["models_dir"],
        help=f"Directory containing model artifacts (Default: {defaults['models_dir']})"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: number of samples to evaluate on for faster execution"
    )

    args = parser.parse_args()

    evaluate_alone(
        model_path=args.model,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()
