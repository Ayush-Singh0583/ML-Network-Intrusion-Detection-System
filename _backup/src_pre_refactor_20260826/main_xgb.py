from sklearn.metrics import accuracy_score

from preprocessing import *
from training import *
from evaluation import *


# ============================
# MODEL SELECTION
# ============================

MODEL = "xgboost"


# ============================
# PREPARE DATA
# Cross-Day Validation
# Train : Monday–Thursday
# Test  : Friday
# ============================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    feature_names
) = prepare_data(
    save_csv=True,
    split_by_day=True
)


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

# Confidence threshold for open-set detection
UNKNOWN_THRESHOLD = 0.90

y_pred_labels = evaluate_model(
    model,
    X_test,
    y_test,
    encoder,
    unknown_threshold=UNKNOWN_THRESHOLD,
    model_name=MODEL
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

print("Validation Type : Cross-Day")
print("Training Days   : Monday–Thursday")
print("Testing Day     : Friday")

print("Model Type      :", MODEL)
print("Number of Trees :", model.n_estimators)
print("Maximum Depth   :", model.max_depth)
print("Learning Rate   :", model.learning_rate)

# Training accuracy (encoded labels)
train_accuracy = model.score(
    X_train,
    y_train
)

# Testing accuracy (string labels)
test_accuracy = accuracy_score(
    y_test,
    y_pred_labels
)

print(f"Training Accuracy : {train_accuracy:.6f}")
print(f"Testing Accuracy  : {test_accuracy:.6f}")