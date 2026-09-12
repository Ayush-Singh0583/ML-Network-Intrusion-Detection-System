from preprocessing import *
from training import *

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================
# LOAD & PREPROCESS DATA
# ============================

print("\n========== PREPARING DATA ==========\n")

df = load_all_datasets("data")
df = clean_dataset(df)
df = remove_identifier_columns(df)

X, y = split_features_target(df)
y, encoder = encode_labels(y)

X_train, X_test, y_train, y_test = split_dataset(X, y)
X_train, X_test, scaler = scale_dataset(X_train, X_test)

# ============================
# MODELS TO COMPARE
# ============================

models = {
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm
}

results = []

# ============================
# TRAIN & EVALUATE
# ============================

for name, trainer in models.items():

    print(f"\nTraining {name}...")

    model = trainer(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    })

# ============================
# RESULTS TABLE
# ============================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========\n")
print(results_df.round(6))

# ============================
# SAVE RESULTS
# ============================

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nResults saved to results/model_comparison.csv")