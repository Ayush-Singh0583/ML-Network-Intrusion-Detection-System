from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight

import numpy as np
import joblib
import os


# ============================
# LOGISTIC REGRESSION
# ============================

def train_logistic(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


# ============================
# DECISION TREE
# ============================

def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


# ============================
# RANDOM FOREST
# ============================

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


from sklearn.utils.class_weight import compute_sample_weight


# ============================
# XGBOOST
# ============================

def train_xgboost(X_train, y_train):

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )
    return model


# ============================
# LIGHTGBM
# ============================
def train_lightgbm(X_train, y_train):

    num_classes = len(np.unique(y_train))

    model = LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,

        n_estimators=300,
        learning_rate=0.05,

        num_leaves=255,
        max_depth=-1,

        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,

        class_weight="balanced",

        random_state=42,
        force_col_wise=True,
        verbose=-1
    )

    model.fit(X_train, y_train)
    return model

# ============================
# SAVE MODEL
# ============================

def save_model(model,
               scaler,
               encoder,
               feature_names,
               model_name):

    print("\n========== SAVING FILES ==========")

    save_dir = os.path.join("saved_models", model_name)
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(
        model,
        os.path.join(save_dir, f"{model_name}_model.pkl")
    )

    joblib.dump(
        scaler,
        os.path.join(save_dir, "scaler.pkl")
    )

    joblib.dump(
        encoder,
        os.path.join(save_dir, "label_encoder.pkl")
    )

    joblib.dump(
        feature_names,
        os.path.join(save_dir, "feature_names.pkl")
    )

    print(f"{model_name} model saved successfully.\n")

    print("Files:")
    for file in os.listdir(save_dir):
        print(" -", file)

    print("\n========== SAVE COMPLETE ==========\n")