import pandas as pd
import numpy as np

# Load preprocessed CSV
df = pd.read_csv("results/preprocessed_dataset.csv")

print("=" * 50)
print("DATASET VALIDATION REPORT")
print("=" * 50)

# 1. Shape
print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# 2. Missing values
missing = df.isna().sum().sum()
print(f"\nMissing Values : {missing}")

# 3. Infinite values
numeric = df.select_dtypes(include=np.number)
inf_count = np.isinf(numeric).sum().sum()
print(f"Infinite Values : {inf_count}")

# 4. Duplicate rows
duplicates = df.duplicated().sum()
print(f"Duplicate Rows : {duplicates}")

# 5. Label check
print("\nLabels Found:")
print(df["Label"].value_counts())

# 6. Constant columns
constant = [c for c in df.columns if df[c].nunique() == 1]
print(f"\nConstant Columns ({len(constant)}):")
print(constant)

# 7. All-zero columns
zero_cols = []
for col in numeric.columns:
    if (df[col] == 0).all():
        zero_cols.append(col)

print(f"\nAll Zero Columns ({len(zero_cols)}):")
print(zero_cols)

# 8. Data types
print("\nData Types:")
print(df.dtypes.value_counts())

print("\nValidation Complete.")