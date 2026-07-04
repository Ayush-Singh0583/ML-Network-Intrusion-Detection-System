import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_dataset(df):

    # Remove negative Flow Duration
    df = df[df["Flow Duration"] >= 0]

    # Replace infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    return df


def remove_identifier_columns(df):

    identifier_columns = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ]

    df.drop(columns=identifier_columns,
            inplace=True,
            errors="ignore")

    return df


def split_features_target(df):

    X = df.drop(columns="Label")
    y = df["Label"]

    return X, y


def encode_labels(y):

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    return y, encoder


def split_dataset(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def scale_dataset(X_train, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler