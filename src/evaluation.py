import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


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
import pandas as pd
import matplotlib.pyplot as plt


def feature_importance(model, feature_names):

    # Create DataFrame
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    # Sort by Importance
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    # Display Top 10
    print("\n========================================")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("========================================\n")

    print(importance.head(10))

    # Save CSV
    importance.to_csv(
        "models/random_forest_feature_importance.csv",
        index=False
    )

    # Plot Top 10 Features
    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["Feature"].head(10)[::-1],
        importance["Importance"].head(10)[::-1]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Top 10 Most Important Features")

    plt.tight_layout()

    plt.savefig(
        "models/feature_importance.png",
        dpi=300
    )

    plt.close()

    print("\nFeature importance saved to:")
    print("models/random_forest_feature_importance.csv")

    print("Feature Importance Graph saved to:")
    print("models/feature_importance.png")

    return importance