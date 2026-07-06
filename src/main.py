from preprocessing import *

from training import *

from evaluation import *


# Load Dataset

df = load_all_datasets("data")
print("After Loading:", df.shape)

df = clean_dataset(df)
print("After Cleaning:", df.shape)

df = remove_identifier_columns(df)
print("After Removing Columns:", df.shape)

X, y = split_features_target(df)
print("X Shape:", X.shape)
print("y Shape:", y.shape)


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

#MODEL = "decision_tree"
# MODEL = "logistic"
MODEL = "random_forest"

if MODEL == "logistic":
    model = train_logistic(X_train, y_train)

elif MODEL == "decision_tree":
    model = train_decision_tree(X_train, y_train)

elif MODEL == "random_forest":
    model = train_random_forest(X_train, y_train)

else:
    raise ValueError("Invalid model selected.")



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

print("\n========== MODEL INFORMATION ==========")

if MODEL == "decision_tree":

    print("Tree Depth :", model.get_depth())
    print("Number of Leaves :", model.get_n_leaves())

elif MODEL == "random_forest":

    depths = [tree.get_depth() for tree in model.estimators_]
    leaves = [tree.get_n_leaves() for tree in model.estimators_]

    print("Number of Trees :", len(model.estimators_))
    print("Average Tree Depth :", sum(depths) / len(depths))
    print("Average Number of Leaves :", sum(leaves) / len(leaves))
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy : {train_accuracy:.6f}")
print(f"Testing Accuracy  : {test_accuracy:.6f}")