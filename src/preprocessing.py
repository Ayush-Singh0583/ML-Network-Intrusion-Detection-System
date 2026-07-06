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

        # Remove leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Clean label column
        df["Label"] = (
            df["Label"]
            .astype(str)
            .str.strip()
        )

        # Replace strange unicode characters
        df["Label"] = (
            df["Label"]
            .str.replace("\uFFFD", "-", regex=False)
            .str.replace("–", "-", regex=False)
        )

        # Standardize Web Attack labels
        df["Label"] = df["Label"].replace({
            "Web Attack - Brute Force": "WebAttack_BruteForce",
            "Web Attack - XSS": "WebAttack_XSS",
            "Web Attack - Sql Injection": "WebAttack_SQLInjection",
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

    # Remove negative Flow Duration
    if "Flow Duration" in df.columns:
        df = df[df["Flow Duration"] >= 0]

    # Replace infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove columns with >50% missing values
    threshold = len(df) * 0.5

    df.dropna(
        axis=1,
        thresh=threshold,
        inplace=True
    )

    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=np.number).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    # Remove rows where Label is missing
    df.dropna(subset=["Label"], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("\nFinal Shape :", df.shape)

    print("\n========== LABEL DISTRIBUTION ==========\n")
    print(df["Label"].value_counts())

    return df


# ==========================================
# REMOVE IDENTIFIER COLUMNS
# ==========================================

def remove_identifier_columns(df):

    identifier_columns = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ]

    df.drop(
        columns=identifier_columns,
        inplace=True,
        errors="ignore"
    )

    return df


# ==========================================
# FEATURE / TARGET SPLIT
# ==========================================

def split_features_target(df):

    X = df.drop(columns="Label")

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
# TRAIN TEST SPLIT
# ==========================================

def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ==========================================
# FEATURE SCALING
# ==========================================

def scale_dataset(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler