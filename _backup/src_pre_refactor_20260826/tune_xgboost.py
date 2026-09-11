from preprocessing import prepare_data
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ==========================================
# LOAD DATA
# ==========================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    feature_names
) = prepare_data(
    save_csv=False,
    split_by_day=True
)

# ==========================================
# BASE MODEL
# ==========================================

xgb = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

# ==========================================
# SEARCH SPACE
# ==========================================

param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5]
}

# ==========================================
# RANDOM SEARCH
# ==========================================

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,          # 10 candidates is enough
    scoring="accuracy",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=1            # IMPORTANT: prevents RAM explosion
)

print("\n========== STARTING TUNING ==========\n")

search.fit(X_train, y_train)

print("\n========== BEST PARAMETERS ==========\n")
print(search.best_params_)

best_model = search.best_estimator_

# ==========================================
# TEST ON FRIDAY
# ==========================================

y_pred = best_model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== FRIDAY RESULT ==========\n")
print(f"Accuracy : {accuracy:.6f}")

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/xgb_tuned.pkl")

print("\nTuned model saved successfully.")