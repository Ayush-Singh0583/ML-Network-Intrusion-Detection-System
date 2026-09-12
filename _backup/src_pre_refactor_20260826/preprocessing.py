import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ==========================================
# LOAD ALL DATASETS
# ==========================================

def load_all_datasets(folder_path):

    csv_files = sorted([
        file for file in os.listdir(folder_path)
        if file.endswith(".csv")
    ])

    print("\n========== LOADING DATASETS ==========\n")

    dataframes = []

    for file in csv_files:

        print(f"Loading {file}")

        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)

        # ----------------------------------
        # Add Day Column
        # ----------------------------------

        if "Monday" in file:
            df["Day"] = "Monday"

        elif "Tuesday" in file:
            df["Day"] = "Tuesday"

        elif "Wednesday" in file:
            df["Day"] = "Wednesday"

        elif "Thursday" in file:
            df["Day"] = "Thursday"

        elif "Friday" in file:
            df["Day"] = "Friday"

        else:
            df["Day"] = "Custom"

        # Clean column names
        df.columns = df.columns.str.strip()

        # Clean labels
        df["Label"] = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.replace("\uFFFD", "-", regex=False)
            .str.replace("–", "-", regex=False)
        )

        # Standardize Web Attack labels
        df["Label"] = df["Label"].replace({
            "Web Attack - Brute Force": "WebAttack_BruteForce",
            "Web Attack - XSS": "WebAttack_XSS",
            "Web Attack - Sql Injection": "WebAttack_SQLInjection"
        })

        dataframes.append(df)

    merged_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("\n========================================")
    print("DATASET SUMMARY")
    print("========================================")
    print(f"Datasets Loaded : {len(csv_files)}")
    print(f"Rows            : {merged_df.shape[0]}")
    print(f"Columns         : {merged_df.shape[1]}")

    return merged_df


# ==========================================
# CLEAN DATASET
# ==========================================

def clean_dataset(df):

    print("\n========== CLEANING DATASET ==========\n")
    print("Initial Shape :", df.shape)

    # Remove invalid flow duration
    if "Flow Duration" in df.columns:
        df = df[df["Flow Duration"] >= 0]

    # Replace infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns with >50% missing values
    threshold = len(df) * 0.5

    df.dropna(
        axis=1,
        thresh=threshold,
        inplace=True
    )

    # Fill numeric values with median
    numeric_columns = df.select_dtypes(include=np.number).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(df[numeric_columns].median())
    )

    # Remove missing labels
    df.dropna(subset=["Label"], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("\nFinal Shape :", df.shape)

    print("\n========== LABEL DISTRIBUTION ==========\n")
    print(df["Label"].value_counts())

    return df


# ==========================================
# REMOVE IDENTIFIER + CONSTANT COLUMNS
# ==========================================

def remove_identifier_columns(df):

    identifier_columns = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Timestamp"
    ]

    df = df.drop(
        columns=identifier_columns,
        errors="ignore"
    )

    constant_columns = [
        col for col in df.columns
        if col not in ["Label", "Day"]
        and df[col].nunique() == 1
    ]

    print("\n========== REMOVING CONSTANT COLUMNS ==========")
    print(constant_columns)

    df = df.drop(columns=constant_columns)

    print(f"Removed {len(constant_columns)} constant columns")
    print(f"Remaining Columns : {df.shape[1]}")

    return df


# ==========================================
# FEATURE / TARGET SPLIT
# ==========================================

def split_features_target(df):

    X = df.drop(columns=["Label", "Day"], errors="ignore")
    y = df["Label"]

    return X, y


# ==========================================
# LABEL ENCODING
# ==========================================

def encode_labels(y):

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    print("\n========== ENCODED CLASSES ==========\n")

    for i, label in enumerate(encoder.classes_):
        print(f"{i} -> {label}")

    return y, encoder


# ==========================================
# RANDOM STRATIFIED SPLIT
# ==========================================

def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


# ==========================================    
# CROSS-DAY SPLIT
# Train : Monday–Thursday
# Test  : Friday
# ==========================================

def split_dataset_by_day(df):

    train_df = df[df["Day"].isin([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday"
    ])]

    test_df = df[df["Day"] == "Friday"]

    X_train = train_df.drop(columns=["Label", "Day"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label", "Day"])
    y_test = test_df["Label"]

    return X_train, X_test, y_train, y_test


# ==========================================
# FEATURE SCALING
# ==========================================

def scale_dataset(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler


# ==========================================
# COMPLETE PREPROCESSING PIPELINE
# ==========================================

def prepare_data(
    folder_path="data",
    save_csv=False,
    split_by_day=False
):

    # Load
    df = load_all_datasets(folder_path)
    print("After Loading:", df.shape)

    # Clean
    df = clean_dataset(df)
    print("After Cleaning:", df.shape)

    # Remove unwanted columns
    df = remove_identifier_columns(df)
    print("After Removing Columns:", df.shape)

    # Save cleaned dataset
    if save_csv:

        os.makedirs("results", exist_ok=True)

        df.to_csv(
            "results/preprocessed_dataset.csv",
            index=False
        )

        print("Preprocessed CSV saved to results/preprocessed_dataset.csv")

    # --------------------------------------
    # Cross-Day Split
    # --------------------------------------
    if split_by_day:

        X_train, X_test, y_train, y_test = split_dataset_by_day(df)

        feature_names = X_train.columns

        print("\n========== CROSS-DAY SPLIT ==========")
        print("Training : Monday–Thursday")
        print("Testing  : Friday")

        # Global encoder (all classes)
        global_encoder = LabelEncoder()
        global_encoder.fit(df["Label"])

        # Training encoder (only train classes)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)

        print("\nTraining Classes:")
        for i, cls in enumerate(encoder.classes_):
            print(f"{i} -> {cls}")

    # --------------------------------------
    # Random Stratified Split
    # --------------------------------------

    else:

        X, y = split_features_target(df)

        feature_names = X.columns

        print("X Shape:", X.shape)
        print("y Shape:", y.shape)

        y, encoder = encode_labels(y)

        X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Scale features
    X_train, X_test, scaler = scale_dataset(
        X_train,
        X_test
    )

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
        feature_names
    )

# ==========================================
# BINARY DATA PREPARATION
# BENIGN vs ATTACK
# ==========================================

def prepare_binary_data(folder_path="data", save_csv=False):

    df = load_all_datasets(folder_path)
    print("After Loading:", df.shape)

    df = clean_dataset(df)
    print("After Cleaning:", df.shape)

    df = remove_identifier_columns(df)
    print("After Removing Columns:", df.shape)

    # Convert every attack into ATTACK
    df["Label"] = df["Label"].apply(
        lambda x: "BENIGN" if x == "BENIGN" else "ATTACK"
    )

    if save_csv:
        os.makedirs("results", exist_ok=True)
        df.to_csv(
            "results/binary_dataset.csv",
            index=False
        )

    X = df.drop(columns=["Label", "Day"])
    y = df["Label"]

    feature_names = X.columns

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_test, scaler = scale_dataset(
        X_train,
        X_test
    )

    print("\n========== BINARY SPLIT ==========")
    print("Classes :", list(encoder.classes_))
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
        feature_names
    )

def prepare_attack_data(folder_path="data", save_csv=False):

    df = load_all_datasets(folder_path)
    df = clean_dataset(df)
    df = remove_identifier_columns(df)

    # Keep only attacks
    df = df[df["Label"] != "BENIGN"]

    if save_csv:
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/attack_dataset.csv", index=False)

    X = df.drop(columns=["Label", "Day"])
    y = df["Label"]

    feature_names = X.columns

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_test, scaler = scale_dataset(X_train, X_test)

    print("\n========== ATTACK SPLIT ==========")
    print("Classes :", len(encoder.classes_))
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
        feature_names
    )
# ==========================================
# BINARY CROSS-DAY
# Train : Monday–Thursday
# Test  : Friday
# ==========================================

def prepare_binary_crossday(folder_path="data"):

    df = load_all_datasets(folder_path)
    df = clean_dataset(df)
    df = remove_identifier_columns(df)

    # BENIGN vs ATTACK
    df["Label"] = df["Label"].apply(
        lambda x: "BENIGN" if x == "BENIGN" else "ATTACK"
    )

    train_df = df[df["Day"].isin([
        "Monday", "Tuesday", "Wednesday", "Thursday"
    ])]

    test_df = df[df["Day"] == "Friday"]

    X_train = train_df.drop(columns=["Label", "Day"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label", "Day"])
    y_test = test_df["Label"]

    # SAVE FEATURE NAMES BEFORE SCALING
    feature_names = X_train.columns

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    X_train, X_test, scaler = scale_dataset(X_train, X_test)

    print("\n========== BINARY CROSS-DAY ==========")
    print("Classes :", list(encoder.classes_))
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
        feature_names
    )


# ==========================================
# ATTACK CROSS-DAY
# Train : Monday–Thursday
# Test  : Friday
# ==========================================

def prepare_attack_crossday(folder_path="data"):

    df = load_all_datasets(folder_path)
    df = clean_dataset(df)
    df = remove_identifier_columns(df)

    # Remove BENIGN traffic
    df = df[df["Label"] != "BENIGN"]

    train_df = df[df["Day"].isin([
        "Monday", "Tuesday", "Wednesday", "Thursday"
    ])]

    test_df = df[df["Day"] == "Friday"]

    X_train = train_df.drop(columns=["Label", "Day"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label", "Day"])
    y_test = test_df["Label"]

    # SAVE FEATURE NAMES BEFORE SCALING
    feature_names = X_train.columns

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    X_train, X_test, scaler = scale_dataset(X_train, X_test)

    print("\n========== ATTACK CROSS-DAY ==========")
    print("Classes :", len(encoder.classes_))
    print("Train Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        encoder,
        feature_names
    )