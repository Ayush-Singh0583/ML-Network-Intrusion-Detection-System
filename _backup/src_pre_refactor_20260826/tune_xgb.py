from preprocessing import *

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

# ============================
# LOAD & PREPROCESS DATA
# ============================

print("\n========== LOADING DATA ==========\n")

df = load_all_datasets("data")
df = clean_dataset(df)
df = remove_identifier_columns(df)

X, y = split_features_target(df)
y, encoder = encode_labels(y)

X_train, X_test, y_train, y_test = split_dataset(X, y)
X_train, X_test, scaler = scale_dataset(X_train, X_test)

# ============================
# BASE MODEL
# ============================

xgb = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss"
)

# ============================
# PARAMETER GRID
# ============================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8]
}

# ============================
# GRID SEARCH
# ============================

print("Starting Grid Search...\n")

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=1,
    verbose=2
)

grid.fit(X_train, y_train)

# ============================
# BEST MODEL
# ============================

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== BEST PARAMETERS ==========")
print(grid.best_params_)

print(f"\nBest CV Score : {grid.best_score_:.6f}")
print(f"Test Accuracy : {accuracy:.6f}")

# ============================
# SAVE BEST MODEL
# ============================

os.makedirs("saved_models/xgboost_tuned", exist_ok=True)

joblib.dump(
    best_model,
    "saved_models/xgboost_tuned/xgboost_tuned.pkl"
)

results = pd.DataFrame(grid.cv_results_)
results.to_csv(
    "results/xgb_grid_search.csv",
    index=False
)

print("\nTuned model saved.")
print("Grid search results saved.")