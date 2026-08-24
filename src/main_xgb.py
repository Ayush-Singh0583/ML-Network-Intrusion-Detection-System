from preprocessing import *
from training import *
from evaluation import *


# ============================
# MODEL SELECTION
# ============================

MODEL = "xgboost"


# ============================
# PREPARE DATA
# ============================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    feature_names
) = prepare_data(save_csv=True)


# ============================
# TRAIN MODEL
# ============================

model = train_xgboost(
    X_train,
    y_train
)


# ============================
# EVALUATION
# ============================

y_pred = evaluate_model(
    model,
    X_test,
    y_test
)


# ============================
# FEATURE IMPORTANCE
# ============================

feature_importance(
    model,
    feature_names,
    MODEL
)


# ============================
# SAVE MODEL
# ============================

save_model(
    model,
    scaler,
    encoder,
    feature_names.tolist(),
    MODEL
)


# ============================
# MODEL INFORMATION
# ============================

print("\n========== MODEL INFORMATION ==========")

print("Model Type :", MODEL)
print("Number of Trees :", model.n_estimators)
print("Maximum Depth :", model.max_depth)
print("Learning Rate :", model.learning_rate)

train_accuracy = model.score(
    X_train,
    y_train
)

test_accuracy = model.score(
    X_test,
    y_test
)

print(f"Training Accuracy : {train_accuracy:.6f}")
print(f"Testing Accuracy  : {test_accuracy:.6f}")