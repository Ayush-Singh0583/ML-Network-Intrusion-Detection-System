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
# FEATURE / TARGET SPLIT
# ============================

X, y = split_features_target(df)

feature_names = X.columns

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# ============================
# LABEL ENCODING
# ============================

y, encoder = encode_labels(y)


# ============================
# TRAIN TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = split_dataset(
    X,
    y
)


# ============================
# FEATURE SCALING
# ============================

X_train, X_test, scaler = scale_dataset(
    X_train,
    X_test
)


# ============================
# MODEL SELECTION
# ============================

# MODEL = "logistic"
# MODEL = "decision_tree"
MODEL = "random_forest"


# ============================
# TRAIN MODEL
# ============================

if MODEL == "logistic":

    model = train_logistic(
        X_train,
        y_train
    )

elif MODEL == "decision_tree":

    model = train_decision_tree(
        X_train,
        y_train
    )

elif MODEL == "random_forest":

    model = train_random_forest(
        X_train,
        y_train
    )

else:

    raise ValueError("Invalid model selected.")


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
    feature_names
)


# ============================
# SAVE MODEL
# ============================

save_model(
    model,
    scaler,
    encoder,
    feature_names.tolist()
)


# ============================
# MODEL INFORMATION
# ============================

print("\n========== MODEL INFORMATION ==========")

if MODEL == "decision_tree":

    print("Tree Depth :", model.get_depth())
    print("Number of Leaves :", model.get_n_leaves())


elif MODEL == "random_forest":

    depths = [
        tree.get_depth()
        for tree in model.estimators_
    ]

    leaves = [
        tree.get_n_leaves()
        for tree in model.estimators_
    ]

    print("Number of Trees :", len(model.estimators_))
    print("Average Tree Depth :", sum(depths) / len(depths))
    print("Average Number of Leaves :", sum(leaves) / len(leaves))


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