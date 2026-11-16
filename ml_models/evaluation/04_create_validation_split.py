
import shutil
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_TRAIN = (
    ROOT / "datasets" / "processed" /
    "cicids2017_model_ready" / "train"
)

OUTPUT_ROOT = (
    ROOT / "datasets" / "processed" /
    "cicids2017_validation_split"
)

NEW_TRAIN_DIR = OUTPUT_ROOT / "train"
VALIDATION_DIR = OUTPUT_ROOT / "validation"

REPORT_DIR = ROOT / "reports"

NEW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = REPORT_DIR / "model_feature_list.json"

LABEL_COLUMN = "Label_Encoded"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15


# ============================================================
# 2. LOAD FEATURES
# ============================================================

import json

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

train_files = list(SOURCE_TRAIN.rglob("*.parquet"))

if not train_files:
    raise FileNotFoundError(
        f"No Parquet files found in {SOURCE_TRAIN}"
    )

train_glob = (
    str(SOURCE_TRAIN / "**" / "*.parquet")
    .replace("\\", "/")
)

quoted_features = ", ".join(
    '"' + feature.replace('"', '""') + '"'
    for feature in FEATURES
)

hash_expression = f"hash({quoted_features})"


# ============================================================
# 3. INITIALIZE DUCKDB
# ============================================================

con = duckdb.connect()

print("\n" + "=" * 65)
print("CYBERSHIELD AI - VALIDATION SPLIT CREATION")
print("=" * 65)

print(f"Source training files: {len(train_files)}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# 4. CREATE UNIQUE FEATURE GROUPS
# ============================================================

print("\nGenerating unique feature groups...")

group_query = f"""
SELECT DISTINCT
    CAST({hash_expression} AS VARCHAR) AS group_id,
    "{LABEL_COLUMN}" AS label
FROM read_parquet('{train_glob}', union_by_name=true)
"""

groups = con.execute(group_query).fetchdf()

print(f"Unique feature groups: {len(groups):,}")


# ============================================================
# 5. VERIFY GROUP LABEL CONSISTENCY
# ============================================================

conflicts = (
    groups.groupby("group_id")["label"]
    .nunique()
)

if (conflicts > 1).any():
    raise ValueError(
        "A feature hash has multiple labels. "
        "Stop and investigate before splitting."
    )

class_counts = groups["label"].value_counts()

if (class_counts < 2).any():
    raise ValueError(
        "At least one class has fewer than two feature groups. "
        "A stratified validation split is not possible."
    )

print("\nFeature groups per class:")
print(class_counts.sort_index().to_string())


# ============================================================
# 6. STRATIFIED GROUP SPLIT
# ============================================================

print("\nCreating 85:15 training-validation split...")

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE
)

train_indices, validation_indices = next(
    splitter.split(groups["group_id"], groups["label"])
)

train_groups = groups.iloc[train_indices][["group_id"]].copy()
validation_groups = groups.iloc[validation_indices][["group_id"]].copy()

print(f"Training feature groups: {len(train_groups):,}")
print(f"Validation feature groups: {len(validation_groups):,}")


# ============================================================
# 7. SAVE GROUP IDS
# ============================================================

train_group_path = OUTPUT_ROOT / "train_groups.parquet"
validation_group_path = OUTPUT_ROOT / "validation_groups.parquet"

for path in [train_group_path, validation_group_path]:
    if path.exists():
        path.unlink()

train_groups.to_parquet(train_group_path, index=False)
validation_groups.to_parquet(validation_group_path, index=False)


# ============================================================
# 8. CREATE TEMPORARY SOURCE VIEW
# ============================================================

con.execute(f"""
CREATE OR REPLACE TEMP VIEW source_data AS
SELECT *,
       CAST({hash_expression} AS VARCHAR) AS group_id
FROM read_parquet('{train_glob}', union_by_name=true)
""")


# ============================================================
# 9. WRITE NEW TRAINING DATA
# ============================================================

train_output = NEW_TRAIN_DIR / "train.parquet"

if train_output.exists():
    train_output.unlink()

print("\nWriting new training dataset...")

con.execute(f"""
COPY (
    SELECT s.* EXCLUDE (group_id)
    FROM source_data s
    INNER JOIN read_parquet('{str(train_group_path).replace(chr(92), '/')}')
        g USING (group_id)
)
TO '{str(train_output).replace(chr(92), '/')}' 
(FORMAT PARQUET, COMPRESSION ZSTD)
""")


# ============================================================
# 10. WRITE VALIDATION DATA
# ============================================================

validation_output = VALIDATION_DIR / "validation.parquet"

if validation_output.exists():
    validation_output.unlink()

print("Writing validation dataset...")

con.execute(f"""
COPY (
    SELECT s.* EXCLUDE (group_id)
    FROM source_data s
    INNER JOIN read_parquet('{str(validation_group_path).replace(chr(92), '/')}')
        g USING (group_id)
)
TO '{str(validation_output).replace(chr(92), '/')}' 
(FORMAT PARQUET, COMPRESSION ZSTD)
""")


# ============================================================
# 11. VALIDATE ROW COUNTS
# ============================================================

new_train_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{str(train_output).replace(chr(92), '/')}')"
).fetchone()[0]

validation_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{str(validation_output).replace(chr(92), '/')}')"
).fetchone()[0]

original_count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{train_glob}', union_by_name=true)"
).fetchone()[0]

if new_train_count + validation_count != original_count:
    raise ValueError("Row count mismatch after splitting.")

overlap = con.execute(f"""
SELECT COUNT(*)
FROM (
    SELECT CAST({hash_expression} AS VARCHAR) AS group_id
    FROM read_parquet('{str(train_output).replace(chr(92), '/')}')
) a
INNER JOIN (
    SELECT CAST({hash_expression} AS VARCHAR) AS group_id
    FROM read_parquet('{str(validation_output).replace(chr(92), '/')}')
) b USING (group_id)
""").fetchone()[0]

if overlap != 0:
    raise ValueError("Feature-group leakage detected!")


# ============================================================
# 12. SAVE CLASS DISTRIBUTIONS
# ============================================================

train_distribution = con.execute(f"""
SELECT "{LABEL_COLUMN}" AS label, COUNT(*) AS records
FROM read_parquet('{str(train_output).replace(chr(92), '/')}')
GROUP BY 1 ORDER BY 1
""").fetchdf()

validation_distribution = con.execute(f"""
SELECT "{LABEL_COLUMN}" AS label, COUNT(*) AS records
FROM read_parquet('{str(validation_output).replace(chr(92), '/')}')
GROUP BY 1 ORDER BY 1
""").fetchdf()

train_distribution.to_csv(
    REPORT_DIR / "validation_split_train_distribution.csv",
    index=False
)

validation_distribution.to_csv(
    REPORT_DIR / "validation_split_distribution.csv",
    index=False
)

con.close()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION SPLIT SUMMARY")
print("=" * 65)

print(f"Original training records: {original_count:,}")
print(f"New training records:      {new_train_count:,}")
print(f"Validation records:        {validation_count:,}")
print(f"Feature group overlap:     {overlap}")

print(
    f"New training percentage: "
    f"{new_train_count / original_count * 100:.2f}%"
)

print(
    f"Validation percentage: "
    f"{validation_count / original_count * 100:.2f}%"
)

print("\nNew dataset locations:")
print(NEW_TRAIN_DIR)
print(VALIDATION_DIR)

print("\nSTEP 19.1 COMPLETED SUCCESSFULLY")
print("=" * 65)
