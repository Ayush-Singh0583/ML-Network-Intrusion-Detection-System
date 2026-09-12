from preprocessing import *
from training import *
from evaluation import *

# ============================
# LOAD DATASET
# ============================

df = load_all_datasets("data")
print("After Loading:", df.shape)

# ============================
# CLEAN DATASET
# ============================

df = clean_dataset(df)
print("After Cleaning:", df.shape)

df = remove_identifier_columns(df)
print("After Removing Columns:", df.shape)

# ============================
# FEATURE / TARGET
# ============================

X, y = split_features_target(df)
feature_names = X.columns

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ============================
# ENCODE LABELS
# ============================

y, encoder = encode_labels(y)

# ============================
# SPLIT
# ============================

X_train, X_test, y_train, y_test = split_dataset(X, y)

# ============================
# SCALE
# ============================

X_train, X_test, scaler = scale_dataset(X_train, X_test)

# ============================
# TRAIN
# ============================

MODEL = "lightgbm"

model = train_lightgbm(
    X_train,
    y_train
)

# ============================
# EVALUATE
# ============================

y_pred = evaluate_model(
    model,
    X_test,
    y_test,
    encoder,
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
# SAVE
# ============================

save_model(
    model,
    scaler,
    encoder,
    feature_names.tolist(),
    MODEL
)

# ============================
# INFO
# ============================

print("\n========== MODEL INFORMATION ==========")

print("Model Type :", MODEL)
print("Number of Trees :", model.n_estimators)
print("Maximum Depth :", model.max_depth)
print("Learning Rate :", model.learning_rate)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy : {train_accuracy:.6f}")
print(f"Testing Accuracy  : {test_accuracy:.6f}")