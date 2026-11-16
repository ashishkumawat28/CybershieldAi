
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
)

parquet_files = sorted(DATA_PATH.rglob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError("No Parquet files found!")

total_rows = 0
total_missing = 0
total_infinite = 0
labels = Counter()
all_columns = None

print("=" * 65)
print("       CYBERSHIELD AI - CLEAN DATASET VALIDATION")
print("=" * 65)

for file in parquet_files:

    df = pd.read_parquet(file)

    # Check consistent columns
    if all_columns is None:
        all_columns = df.columns.tolist()
    elif df.columns.tolist() != all_columns:
        raise ValueError(f"Column mismatch: {file}")

    if "Label" not in df.columns:
        raise ValueError(f"Label column missing: {file}")

    feature_columns = [
        col for col in df.columns if col != "Label"
    ]

    # Count records and labels
    total_rows += len(df)
    labels.update(df["Label"].value_counts().to_dict())

    # Count missing values
    total_missing += int(
        df[feature_columns].isna().to_numpy().sum()
    )

    # Count remaining infinite values
    numeric_df = df[feature_columns].select_dtypes(
        include=[np.number]
    )

    total_infinite += int(
        np.isinf(numeric_df.to_numpy()).sum()
    )

print("\nVALIDATION SUMMARY")
print("-" * 65)

print(f"Parquet files: {len(parquet_files)}")
print(f"Total records: {total_rows:,}")
print(f"Total columns: {len(all_columns)}")
print(f"Input features: {len(all_columns) - 1}")
print(f"Unique labels: {len(labels)}")
print(f"Missing feature values: {total_missing:,}")
print(f"Remaining infinite values: {total_infinite:,}")

print("\nLABEL DISTRIBUTION")
print("-" * 65)

for label, count in labels.most_common():
    print(f"{label:30s} {count:12,}")

print("\nVALIDATION CHECKS")
print("-" * 65)

print(
    "Record count correct:",
    total_rows == 2_830_743
)

print(
    "Expected 15 labels:",
    len(labels) == 15
)

print(
    "No infinite values:",
    total_infinite == 0
)

print(
    "All columns consistent:",
    all_columns is not None
)

print("\nValidation completed.")

