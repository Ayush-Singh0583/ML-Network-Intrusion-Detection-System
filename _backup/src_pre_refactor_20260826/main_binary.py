from preprocessing import prepare_binary_data
from training import train_xgboost, save_model
from evaluation import evaluate_model

MODEL = "binary_xgboost"

# ============================
# PREPARE BINARY DATA
# ============================

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    feature_names
) = prepare_binary_data(
    save_csv=False
)

# ============================
# TRAIN
# ============================

model = train_xgboost(
    X_train,
    y_train
)

# ============================
# EVALUATE
# ============================

evaluate_model(
    model,
    X_test,
    y_test,
    encoder=encoder,
    model_name=MODEL
)

# ============================
# SAVE
# ============================

save_model(
    model,
    scaler,
    encoder,
    feature_names.tolist(),
    MODEL
)

print("\n========== BINARY MODEL COMPLETE ==========")
print("Classes :", list(encoder.classes_))
print("Train Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)