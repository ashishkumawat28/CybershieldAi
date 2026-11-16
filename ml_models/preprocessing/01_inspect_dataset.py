
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "raw"

print("=" * 60)
print("       CYBERSHIELD AI - DATASET INSPECTION")
print("=" * 60)

# Find CSV files
csv_files = list(DATASET_PATH.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {DATASET_PATH}"
    )

print(f"\nTotal CSV files found: {len(csv_files)}")

# Display file information
print("\nDATASET FILES")
print("-" * 60)

for file in csv_files:
    size_mb = file.stat().st_size / (1024 * 1024)

    print(f"{file.name}")
    print(f"Size: {size_mb:.2f} MB")
    print("-" * 40)

# Inspect the first CSV file
selected_file = csv_files[0]

print("\nINSPECTING FILE:")
print(selected_file.name)

# Read only a small sample initially
df = pd.read_csv(
    selected_file,
    nrows=10000,
    low_memory=False
)

# Clean column-name whitespace
df.columns = df.columns.str.strip()

print("\n1. DATASET SAMPLE SHAPE")
print(df.shape)

print("\n2. FIRST FIVE RECORDS")
print(df.head())

print("\n3. COLUMN NAMES")
for column in df.columns:
    print(column)

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\n6. DUPLICATE RECORDS IN SAMPLE")
print(df.duplicated().sum())

# Identify label column
label_columns = [
    col for col in df.columns
    if col.strip().lower() == "label"
]

if label_columns:
    label_col = label_columns[0]

    print("\n7. ATTACK LABEL DISTRIBUTION (SAMPLE)")
    print(df[label_col].value_counts(dropna=False))
else:
    print("\nLabel column not found!")

print("\n8. NUMERICAL SUMMARY")
print(df.describe().T)

print("\nINSPECTION COMPLETED!")
