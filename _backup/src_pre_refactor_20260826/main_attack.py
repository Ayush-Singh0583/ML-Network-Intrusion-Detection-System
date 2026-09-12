from preprocessing import prepare_attack_data
from training import train_xgboost, save_model
from evaluation import evaluate_model

MODEL = "attack_xgboost"

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    encoder,
    feature_names
) = prepare_attack_data(save_csv=False)

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

print("\n========== ATTACK MODEL COMPLETE ==========")
print("Attack Classes:", len(encoder.classes_))
print(encoder.classes_)