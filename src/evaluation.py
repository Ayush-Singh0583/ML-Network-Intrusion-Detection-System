import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        digits=6
    )

    print("\n========== MODEL EVALUATION ==========")

    print("\nAccuracy")
    print(accuracy)

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)

    return y_pred


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

def feature_importance(model, feature_names, model_name):

    if not hasattr(model, "feature_importances_"):
        print("\nThis model does not support feature importance.")
        return None

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========================================")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("========================================\n")
    print(importance.head(10))

    # Create results folder
    os.makedirs("results", exist_ok=True)

    csv_path = f"results/{model_name}_feature_importance.csv"
    img_path = f"results/{model_name}_feature_importance.png"

    # Save CSV
    importance.to_csv(
        csv_path,
        index=False
    )

    # Plot Top 10
    plt.figure(figsize=(10, 6))

    top10 = importance.head(10)

    plt.barh(
        top10["Feature"][::-1],
        top10["Importance"][::-1]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(f"Top 10 Features - {model_name}")

    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    print("\nFeature importance saved to:")
    print(csv_path)

    print("Feature Importance Graph saved to:")
    print(img_path)

    return importance