import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_integer_dtype
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test,
    encoder=None,
    unknown_threshold=None,
    model_name="model"
):

    # --------------------------------------
    # Ground Truth Labels
    # --------------------------------------

    if is_integer_dtype(y_test) and encoder is not None:
        y_true = np.array(encoder.inverse_transform(y_test)).astype(str)
    else:
        y_true = np.array(y_test).astype(str)

    # Convert unseen Friday attacks into Unknown_Attack
    known_classes = set(encoder.classes_)

    y_true_open = np.array([
        label if label in known_classes else "Unknown_Attack"
        for label in y_true
    ])

    # --------------------------------------
    # Prediction Probabilities
    # --------------------------------------
    probabilities = model.predict_proba(X_test)

    # Highest confidence for every sample
    confidence = probabilities.max(axis=1)

    # Predicted class index
    y_pred = probabilities.argmax(axis=1)

    # Convert to class names
    if encoder is not None:
        y_pred_labels = np.array(encoder.inverse_transform(y_pred)).astype(str)
    else:
        y_pred_labels = y_pred.astype(str)

    # --------------------------------------
    # Unknown Attack Detection
    # --------------------------------------
    num_unknown = 0

    if unknown_threshold is not None:
        unknown_mask = confidence < unknown_threshold
        num_unknown = int(np.sum(unknown_mask))
        y_pred_labels[unknown_mask] = "Unknown_Attack"

    # All labels appearing in either true or predicted
    labels = sorted(list(set(y_true) | set(y_pred_labels)))

    print("\n========== MODEL EVALUATION ==========\n")

    if unknown_threshold is not None:
        print("========== UNKNOWN ATTACK DETECTION ==========")
        print(f"Confidence Threshold : {unknown_threshold:.2f}")
        print(f"Average Confidence   : {confidence.mean():.4f}")
        print(f"Unknown Detections   : {num_unknown}")
        print(f"Unknown Rate         : {(num_unknown/len(y_pred_labels))*100:.2f}%\n")

    # --------------------------------------
    # Metrics
    # --------------------------------------
    acc = accuracy_score(y_true_open, y_pred_labels)
    macro_p = precision_score(y_true, y_pred_labels, labels=labels,
                              average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred_labels, labels=labels,
                           average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_labels, labels=labels,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred_labels, labels=labels,
                           average="weighted", zero_division=0)

    print(f"Accuracy        : {acc:.6f}")
    print(f"Macro Precision : {macro_p:.6f}")
    print(f"Macro Recall    : {macro_r:.6f}")
    print(f"Macro F1        : {macro_f1:.6f}")
    print(f"Weighted F1     : {weighted_f1:.6f}")

    # --------------------------------------
    # Confusion Matrix
    # --------------------------------------
    cm = confusion_matrix(
        y_true_open,
        y_pred_labels,
        labels=labels
    )

    print("\nConfusion Matrix")
    print(cm)

    # --------------------------------------
    # Classification Report
    # --------------------------------------
    report = classification_report(
        y_true_open,
        y_pred_labels,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    print(classification_report(
        y_true_open,
        y_pred_labels,
        labels=labels,
        zero_division=0
    ))

    # --------------------------------------
    # Save Results
    # --------------------------------------
    os.makedirs("results", exist_ok=True)

    pd.DataFrame(report).transpose().to_csv(
        "results/classification_report.csv"
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )
    plt.close()

    # --------------------------------------
    # Summary
    # --------------------------------------
    print("\n========== SUMMARY ==========")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    if unknown_threshold is not None:
        print(f"Unknown Rate    : {(num_unknown/len(y_pred_labels))*100:.2f}%")

    return y_pred_labels


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def feature_importance(model, feature_names, model_name="model"):

    if not hasattr(model, "feature_importances_"):
        print("\nThis model does not support feature importance.")
        return

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========================================")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("========================================\n")
    print(importance.head(10))

    os.makedirs("results", exist_ok=True)

    csv_path = f"results/{model_name}_feature_importance.csv"
    img_path = f"results/{model_name}_feature_importance.png"

    importance.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    top10 = importance.head(10)

    plt.barh(
        top10["Feature"][::-1],
        top10["Importance"][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 10 Features - {model_name}")

    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"\nFeature importance saved to: {csv_path}")
    print(f"Feature importance graph saved to: {img_path}")

    return importance