
import pandas as pd
import numpy as np
import duckdb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]

TEMP_DIR = ROOT / "datasets" / "processed" / "baseline_temp"
REPORT_DIR = ROOT / "reports"

INDEX_DIR = ROOT / "datasets" / "processed" / "split_index"
TRAIN_DIR = ROOT / "datasets" / "processed" / "cicids2017_split" / "train"
TEST_DIR = ROOT / "datasets" / "processed" / "cicids2017_split" / "test"

GROUP_PARTS = INDEX_DIR / "parts"
GROUP_FILE = INDEX_DIR / "unique_feature_groups.parquet"

REPORT_DIR.mkdir(exist_ok=True)

for folder in [TRAIN_DIR, TEST_DIR]:
    if folder.exists() and any(folder.rglob("*.parquet")):
        raise FileExistsError(
            f"{folder} already contains Parquet files. "
            "Move the previous output before rerunning."
        )

if GROUP_FILE.exists():
    raise FileExistsError(
        f"{GROUP_FILE} already exists. "
        "Move the previous split index before rerunning."
    )

GROUP_PARTS.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(" CYBERSHIELD AI - GROUP-AWARE TRAIN TEST SPLIT")
print("=" * 65)

# --------------------------------------------------
# STEP 1: Build feature-group metadata
# --------------------------------------------------

files = sorted(TEMP_DIR.rglob("*.parquet"))

if not files:
    raise RuntimeError("No source Parquet files found in baseline_temp.")

print(f"\nSource files: {len(files)}")

for i, file in enumerate(files, start=1):

    df = pd.read_parquet(file)

    feature_columns = [
        col for col in df.columns if col != "Label"
    ]

    hashes = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False,
        categorize=True
    ).to_numpy(dtype=np.uint64)

    metadata = pd.DataFrame({
        "Feature_Hash": hashes,
        "Label": df["Label"].astype(str).to_numpy()
    })

    grouped = (
        metadata.groupby(["Feature_Hash", "Label"])
        .size()
        .reset_index(name="Record_Count")
    )

    output = GROUP_PARTS / f"part_{i:03d}.parquet"
    grouped.to_parquet(output, index=False)

    print(f"Indexed {i}/{len(files)} files")

# --------------------------------------------------
# STEP 2: Combine groups across all source files
# --------------------------------------------------

print("\nCombining feature groups with DuckDB...")

con = duckdb.connect()

parts_glob = (GROUP_PARTS / "*.parquet").as_posix()
group_path = GROUP_FILE.as_posix()

con.execute(f"""
    COPY (
        SELECT
            Feature_Hash,
            ANY_VALUE(Label) AS Label,
            SUM(Record_Count) AS Record_Count,
            COUNT(DISTINCT Label) AS Label_Count
        FROM read_parquet('{parts_glob}')
        GROUP BY Feature_Hash
    )
    TO '{group_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

group_df = con.execute(f"""
    SELECT Feature_Hash, Label, Record_Count, Label_Count
    FROM read_parquet('{group_path}')
    ORDER BY Feature_Hash
""").fetchdf()

con.close()

if not (group_df["Label_Count"] == 1).all():
    raise RuntimeError(
        "A feature hash has multiple labels. "
        "Stop and investigate before splitting."
    )

print(f"Unique feature groups: {len(group_df):,}")

# --------------------------------------------------
# STEP 3: Stratified split of unique feature groups
# --------------------------------------------------

class_group_counts = group_df["Label"].value_counts()

print("\nUnique feature groups per class:")
print(class_group_counts.to_string())

minimum_groups = int(class_group_counts.min())

if minimum_groups < 2:
    raise RuntimeError(
        "At least one class has fewer than 2 unique feature groups. "
        "A reliable stratified train/test split needs review."
    )

n_splits = min(5, minimum_groups)

print(f"\nUsing {n_splits}-fold stratification.")

splitter = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

X = np.zeros((len(group_df), 1), dtype=np.uint8)
y = group_df["Label"].to_numpy()

train_indices, test_indices = next(
    splitter.split(X, y)
)

test_hashes = np.sort(
    group_df.iloc[test_indices]["Feature_Hash"]
    .to_numpy(dtype=np.uint64)
)

print(f"Training feature groups: {len(train_indices):,}")
print(f"Testing feature groups: {len(test_indices):,}")

# --------------------------------------------------
# STEP 4: Assign original records to train/test
# --------------------------------------------------

total_train = 0
total_test = 0

for i, file in enumerate(files, start=1):

    df = pd.read_parquet(file)

    feature_columns = [
        col for col in df.columns if col != "Label"
    ]

    hashes = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False,
        categorize=True
    ).to_numpy(dtype=np.uint64)

    test_mask = np.isin(
        hashes,
        test_hashes,
        assume_unique=False
    )

    train_part = df.loc[~test_mask]
    test_part = df.loc[test_mask]

    relative_path = file.relative_to(TEMP_DIR)

    if not train_part.empty:
        out = TRAIN_DIR / relative_path
        out.parent.mkdir(parents=True, exist_ok=True)
        train_part.to_parquet(out, index=False)
        total_train += len(train_part)

    if not test_part.empty:
        out = TEST_DIR / relative_path
        out.parent.mkdir(parents=True, exist_ok=True)
        test_part.to_parquet(out, index=False)
        total_test += len(test_part)

    print(f"Split {i}/{len(files)} files")

# --------------------------------------------------
# STEP 5: Validate split
# --------------------------------------------------

print("\n" + "=" * 65)
print("                 SPLIT RESULTS")
print("=" * 65)

total = total_train + total_test

print(f"Training records: {total_train:,}")
print(f"Testing records:  {total_test:,}")
print(f"Total records:    {total:,}")

print(f"Training ratio: {total_train / total * 100:.2f}%")
print(f"Testing ratio:  {total_test / total * 100:.2f}%")

if total != 2_823_723:
    raise RuntimeError(
        f"Record count mismatch. Expected 2,823,723, got {total:,}"
    )

# Verify that no feature hash occurs in both splits.
train_hash_set = set()
test_hash_set = set()

for file in sorted(TRAIN_DIR.rglob("*.parquet")):
    df = pd.read_parquet(file)
    features = df.drop(columns=["Label"])
    hashes = pd.util.hash_pandas_object(
        features, index=False, categorize=True
    )
    train_hash_set.update(hashes.to_numpy(dtype=np.uint64))

for file in sorted(TEST_DIR.rglob("*.parquet")):
    df = pd.read_parquet(file)
    features = df.drop(columns=["Label"])
    hashes = pd.util.hash_pandas_object(
        features, index=False, categorize=True
    )
    test_hash_set.update(hashes.to_numpy(dtype=np.uint64))

overlap = train_hash_set.intersection(test_hash_set)

print(f"\nTrain unique feature groups: {len(train_hash_set):,}")
print(f"Test unique feature groups:  {len(test_hash_set):,}")
print(f"Feature-group overlap:       {len(overlap):,}")

if overlap:
    raise RuntimeError("Feature leakage detected between splits.")

# Save class distributions.
con = duckdb.connect()

train_glob = (TRAIN_DIR / "**" / "*.parquet").as_posix()
test_glob = (TEST_DIR / "**" / "*.parquet").as_posix()

train_distribution = con.execute(f"""
    SELECT Label, COUNT(*) AS Record_Count
    FROM read_parquet('{train_glob}')
    GROUP BY Label
    ORDER BY Record_Count DESC
""").fetchdf()

test_distribution = con.execute(f"""
    SELECT Label, COUNT(*) AS Record_Count
    FROM read_parquet('{test_glob}')
    GROUP BY Label
    ORDER BY Record_Count DESC
""").fetchdf()

con.close()

train_distribution.to_csv(
    REPORT_DIR / "train_class_distribution.csv",
    index=False
)

test_distribution.to_csv(
    REPORT_DIR / "test_class_distribution.csv",
    index=False
)

print("\nTRAIN CLASS DISTRIBUTION")
print(train_distribution.to_string(index=False))

print("\nTEST CLASS DISTRIBUTION")
print(test_distribution.to_string(index=False))

print("\nReports saved in:", REPORT_DIR)
print("\nStep 11 completed successfully.")
