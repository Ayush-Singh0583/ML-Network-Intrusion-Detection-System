from preprocessing import *

from training import *

from evaluation import *


# Load Dataset

df = load_dataset(
    "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)


# Cleaning

df = clean_dataset(df)

df = remove_identifier_columns(df)


# Feature Target Split

X, y = split_features_target(df)


# Label Encoding

y, encoder = encode_labels(y)


# Train Test Split

X_train, X_test, y_train, y_test = split_dataset(X, y)
# Feature Target Split

X, y = split_features_target(df)

feature_names = X.columns

# Scaling

X_train, X_test, scaler = scale_dataset(
    X_train,
    X_test
)


# Train Model

model = train_logistic(
    X_train,
    y_train
)


# Evaluation

# Evaluate the model
y_pred = evaluate_model(
    model,
    X_test,
    y_test
)

# Display feature importance
coef = feature_importance(
    model,
    feature_names
)

# Save Model

save_model(
    model,
    scaler,
    encoder
)