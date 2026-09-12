from preprocessing import prepare_binary_crossday
from training import train_xgboost, save_model
from evaluation import evaluate_model

MODEL = "binary_xgb_crossday"

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    feature_names
) = prepare_binary_crossday()

model = train_xgboost(X_train, y_train)

evaluate_model(
    model,
    X_test,
    y_test,
    encoder=encoder,
    model_name=MODEL
)

save_model(
    model,
    scaler,
    encoder,
    feature_names.tolist(),
    MODEL
)

print("\n========== BINARY CROSS-DAY COMPLETE ==========")