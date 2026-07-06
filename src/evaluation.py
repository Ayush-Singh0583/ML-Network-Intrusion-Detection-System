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


def feature_importance(model, feature_names):

    coef = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.feature_importances_[0]
    })

    # Absolute importance
    coef["Absolute Coefficient"] = coef["Coefficient"].abs()

    # Most influential features
    importance = coef.sort_values(
        by="Absolute Coefficient",
        ascending=False
    )

    print("\n========================================")
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("========================================\n")
    print(importance.head(10))

    print("\n========================================")
    print("TOP 10 FEATURES INDICATING DDoS")
    print("========================================\n")
    print(
        coef.sort_values(
            by="Coefficient",
            ascending=False
        ).head(10)
    )

    print("\n========================================")
    print("TOP 10 FEATURES INDICATING BENIGN")
    print("========================================\n")
    print(
        coef.sort_values(
            by="Coefficient"
        ).head(10)
    )

    # Save for report and analysis
    importance.to_csv(
        "models/logistic_feature_importance.csv",
        index=False
    )

    print("\nFeature importance saved to:")
    print("models/logistic_feature_importance.csv")

    return importance