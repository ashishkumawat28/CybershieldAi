
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "datasets" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
REPORT_PATH = PROJECT_ROOT / "reports"

CHUNK_SIZE = 25_000

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RAW_PATH.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No raw CSV files found!")

# --------------------------------------------------
# 2. LABEL NORMALIZATION
# --------------------------------------------------

def normalize_label(label):
    label = str(label).strip()

    # Replace corrupted character in Web Attack labels
    label = label.replace("\ufffd", "-")

    # Normalize repeated spaces
    label = re.sub(r"\s+", " ", label)

    return label


# --------------------------------------------------
# 3. PROCESS DATASET
# --------------------------------------------------

expected_columns = None

total_rows = 0
total_infinite = 0
total_missing_after_cleaning = 0

global_labels = Counter()
file_summaries = []

print("=" * 70)
print("       CYBERSHIELD AI - DATA CLEANING")
print("=" * 70)

for csv_file in csv_files:

    print(f"\nProcessing: {csv_file.name}")

    # Read header first
    current_columns = pd.read_csv(
        csv_file,
        nrows=0
    ).columns.str.strip().tolist()

    if "Label" not in current_columns:
        raise ValueError(f"Label column missing: {csv_file.name}")

    # Verify consistent feature columns across files
    if expected_columns is None:
        expected_columns = current_columns
    elif current_columns != expected_columns:
        raise ValueError(
            f"Column mismatch detected in {csv_file.name}. "
            "Inspect the CSV headers before continuing."
        )

    file_output = OUTPUT_PATH / csv_file.stem
    file_output.mkdir(parents=True, exist_ok=True)

    file_rows = 0
    file_labels = Counter()
    chunk_number = 0

    for chunk in pd.read_csv(
        csv_file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        # Clean labels
        chunk["Label"] = (
            chunk["Label"]
            .astype("string")
            .str.strip()
            .map(normalize_label)
        )

        if chunk["Label"].isna().any():
            raise ValueError(
                f"Missing target labels found in {csv_file.name}"
            )

        # Count labels
        labels = chunk["Label"].tolist()
        file_labels.update(labels)
        global_labels.update(labels)

        # Convert every feature to numeric
        feature_columns = [
            col for col in chunk.columns if col != "Label"
        ]

        chunk[feature_columns] = chunk[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace positive and negative infinity with NaN
        numeric_values = chunk[feature_columns]

        infinite_count = int(
            np.isinf(numeric_values).to_numpy().sum()
        )

        total_infinite += infinite_count

        chunk[feature_columns] = numeric_values.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Count remaining missing feature values
        missing_count = int(
            chunk[feature_columns].isna().to_numpy().sum()
        )

        total_missing_after_cleaning += missing_count

        # Save each chunk as a separate Parquet file
        output_file = (
            file_output /
            f"part-{chunk_number:05d}.parquet"
        )

        chunk.to_parquet(
            output_file,
            engine="pyarrow",
            index=False
        )

        file_rows += len(chunk)
        chunk_number += 1

        print(
            f"  Saved chunk {chunk_number} | "
            f"Rows processed: {file_rows:,}"
        )

    total_rows += file_rows

    file_summaries.append({
        "file": csv_file.name,
        "rows": file_rows,
        "chunks": chunk_number
    })

    print(f"Completed: {file_rows:,} rows")
    print(f"Labels: {dict(file_labels)}")


# --------------------------------------------------
# 4. SAVE CLEANING REPORTS
# --------------------------------------------------

label_report = pd.DataFrame(
    global_labels.items(),
    columns=["Label", "Count"]
)

label_report["Percentage"] = (
    label_report["Count"] / total_rows * 100
).round(4)

label_report = label_report.sort_values(
    "Count",
    ascending=False
)

label_report.to_csv(
    REPORT_PATH / "cleaned_attack_distribution.csv",
    index=False
)

pd.DataFrame(file_summaries).to_csv(
    REPORT_PATH / "cleaning_file_summary.csv",
    index=False
)


# --------------------------------------------------
# 5. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("                 CLEANING COMPLETED")
print("=" * 70)

print(f"Total files processed: {len(csv_files)}")
print(f"Total rows processed: {total_rows:,}")
print(f"Infinite values replaced: {total_infinite:,}")
print(
    f"Missing feature values remaining: "
    f"{total_missing_after_cleaning:,}"
)

print("\nFINAL LABEL DISTRIBUTION")

print(label_report.to_string(index=False))

print("\nCleaned dataset saved at:")
print(OUTPUT_PATH)

print("\nReports saved:")
print(REPORT_PATH / "cleaned_attack_distribution.csv")
print(REPORT_PATH / "cleaning_file_summary.csv")

