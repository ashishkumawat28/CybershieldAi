
import json
import numpy as np
import pandas as pd
import duckdb

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "datasets" / "processed"
TRAIN_DIR = PROCESSED_DIR / "cicids2017_split" / "train"
TEST_DIR = PROCESSED_DIR / "cicids2017_split" / "test"

OUTPUT_DIR = PROCESSED_DIR / "cicids2017_model_ready"
OUTPUT_TRAIN = OUTPUT_DIR / "train"
OUTPUT_TEST = OUTPUT_DIR / "test"

REPORT_DIR = ROOT / "reports"

CONSTANT_REPORT = REPORT_DIR / "train_feature_quality.csv"

FEATURE_REPORT = REPORT_DIR / "model_feature_list.json"
LABEL_REPORT = REPORT_DIR / "label_mapping.csv"
SUMMARY_REPORT = REPORT_DIR / "preprocessing_summary.csv"

for folder in [OUTPUT_TRAIN, OUTPUT_TEST]:
    if folder.exists() and any(folder.rglob("*.parquet")):
        raise FileExistsError(
            f"Output folder already contains Parquet files: {folder}. "
            "Move the old output before rerunning."
        )

OUTPUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUTPUT_TEST.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print(" CYBERSHIELD AI - MODEL DATA PREPROCESSING")
print("=" * 65)

# --------------------------------------------------
# STEP 1: Read feature audit and identify constants
# --------------------------------------------------

audit = pd.read_csv(CONSTANT_REPORT)

constant_features = audit.loc[
    audit["Is_Constant"].astype(str).str.lower() == "true",
    "Feature"
].tolist()

print(f"\nConstant features to remove: {len(constant_features)}")

for feature in constant_features:
    print(" -", feature)

# --------------------------------------------------
# STEP 2: Find source files and feature columns
# --------------------------------------------------

train_files = sorted(TRAIN_DIR.rglob("*.parquet"))
test_files = sorted(TEST_DIR.rglob("*.parquet"))

if not train_files or not test_files:
    raise RuntimeError("Training or testing Parquet files are missing.")

sample = pd.read_parquet(train_files[0])

all_features = [
    col for col in sample.columns if col != "Label"
]

feature_columns = [
    col for col in all_features
    if col not in constant_features
]

if len(feature_columns) != 70:
    raise RuntimeError(
        f"Expected 70 features after removing constants, "
        f"found {len(feature_columns)}."
    )

print(f"\nOriginal features: {len(all_features)}")
print(f"Remaining features: {len(feature_columns)}")

# --------------------------------------------------
# STEP 3: Calculate medians using TRAINING data only
# --------------------------------------------------

print("\nCalculating training-only median values...")

def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'

train_glob = (TRAIN_DIR / "**" / "*.parquet").as_posix()
train_glob_sql = train_glob.replace("'", "''")

median_expressions = []

for col in feature_columns:
    q = quote_identifier(col)

    median_expressions.append(
        f"median(CASE WHEN isfinite({q}) "
        f"THEN {q} ELSE NULL END) AS {q}"
    )

query = f"""
    SELECT {", ".join(median_expressions)}
    FROM read_parquet('{train_glob_sql}', union_by_name = true)
"""

con = duckdb.connect()
median_result = con.execute(query).fetchdf().iloc[0].to_dict()
con.close()

medians = {}

for col in feature_columns:
    value = median_result[col]

    if pd.isna(value) or not np.isfinite(value):
        raise RuntimeError(
            f"Cannot calculate a valid training median for {col}"
        )

    medians[col] = float(value)

print("Training medians calculated successfully.")

for col in feature_columns:
    print(f"{col}: {medians[col]}")

# --------------------------------------------------
# STEP 4: Create deterministic label encoding
# --------------------------------------------------

print("\nCreating label mapping...")

labels = set()

for file in train_files:
    df = pd.read_parquet(file, columns=["Label"])
    labels.update(df["Label"].dropna().astype(str).unique())

label_list = sorted(labels)

if len(label_list) != 15:
    raise RuntimeError(
        f"Expected 15 training classes, found {len(label_list)}."
    )

label_mapping = {
    label: index
    for index, label in enumerate(label_list)
}

mapping_df = pd.DataFrame({
    "Label": label_list,
    "Label_Encoded": range(len(label_list))
})

mapping_df.to_csv(LABEL_REPORT, index=False)

print(mapping_df.to_string(index=False))

# --------------------------------------------------
# STEP 5: Transform train and test datasets
# --------------------------------------------------

def transform_split(files, source_dir, output_dir, split_name):

    total_rows = 0
    total_missing_filled = 0

    for i, file in enumerate(files, start=1):

        df = pd.read_parquet(file)

        if "Label" not in df.columns:
            raise RuntimeError(f"Label column missing in {file}")

        if set(all_features) - set(df.columns):
            raise RuntimeError(f"Feature columns missing in {file}")

        original_labels = df["Label"].astype(str)

        unknown_labels = set(original_labels.unique()) - set(label_mapping)

        if unknown_labels:
            raise RuntimeError(
                f"Unknown labels in {file}: {unknown_labels}"
            )

        # Remove constant features.
        df = df.drop(columns=constant_features)

        # Convert non-finite values into missing values.
        feature_data = df[feature_columns].replace(
            [np.inf, -np.inf], np.nan
        )

        missing_before = int(feature_data.isna().sum().sum())

        # Apply medians learned from training data.
        feature_data = feature_data.fillna(medians)

        if feature_data.isna().any().any():
            raise RuntimeError(
                f"Missing values remain in {split_name}: {file}"
            )

        if not np.isfinite(
            feature_data.to_numpy(dtype=np.float64)
        ).all():
            raise RuntimeError(
                f"Infinite values remain in {split_name}: {file}"
            )

        result = feature_data.copy()
        result["Label"] = original_labels.to_numpy()
        result["Label_Encoded"] = (
            original_labels.map(label_mapping).to_numpy()
        )

        relative_path = file.relative_to(source_dir)
        output_file = output_dir / relative_path

        output_file.parent.mkdir(parents=True, exist_ok=True)

        result.to_parquet(
            output_file,
            index=False,
            compression="zstd"
        )

        total_rows += len(result)
        total_missing_filled += missing_before

        print(
            f"{split_name}: processed {i}/{len(files)} files"
        )

    return total_rows, total_missing_filled


print("\nProcessing training dataset...")

train_rows, train_filled = transform_split(
    train_files,
    TRAIN_DIR,
    OUTPUT_TRAIN,
    "TRAIN"
)

print("\nProcessing testing dataset...")

test_rows, test_filled = transform_split(
    test_files,
    TEST_DIR,
    OUTPUT_TEST,
    "TEST"
)

# --------------------------------------------------
# STEP 6: Save feature list and preprocessing details
# --------------------------------------------------

with open(FEATURE_REPORT, "w", encoding="utf-8") as f:
    json.dump(feature_columns, f, indent=4)

with open(
    OUTPUT_DIR / "training_medians.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(medians, f, indent=4)

summary = pd.DataFrame([
    {"Metric": "Original features", "Value": len(all_features)},
    {"Metric": "Constant features removed", "Value": len(constant_features)},
    {"Metric": "Final model features", "Value": len(feature_columns)},
    {"Metric": "Number of classes", "Value": len(label_list)},
    {"Metric": "Training records", "Value": train_rows},
    {"Metric": "Testing records", "Value": test_rows},
    {"Metric": "Training missing values filled", "Value": train_filled},
    {"Metric": "Testing missing values filled", "Value": test_filled}
])

summary.to_csv(SUMMARY_REPORT, index=False)

# --------------------------------------------------
# STEP 7: Final results
# --------------------------------------------------

print("\n" + "=" * 65)
print("                 FINAL RESULTS")
print("=" * 65)

print(f"Original features: {len(all_features)}")
print(f"Constant features removed: {len(constant_features)}")
print(f"Final features: {len(feature_columns)}")
print(f"Number of classes: {len(label_list)}")

print(f"\nTraining records: {train_rows:,}")
print(f"Testing records: {test_rows:,}")

print(f"\nTraining missing values filled: {train_filled:,}")
print(f"Testing missing values filled: {test_filled:,}")

print("\nModel-ready folders:")
print(OUTPUT_TRAIN)
print(OUTPUT_TEST)

print("\nReports:")
print(FEATURE_REPORT)
print(LABEL_REPORT)
print(SUMMARY_REPORT)

print("\nStep 13 completed successfully.")
