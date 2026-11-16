
import pandas as pd
import sqlite3
from pathlib import Path

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
)

REPORT_PATH = PROJECT_ROOT / "reports"
REPORT_PATH.mkdir(exist_ok=True)

DB_PATH = REPORT_PATH / "duplicate_audit.sqlite"

parquet_files = sorted(DATA_PATH.rglob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError("No Parquet files found!")

# --------------------------------------------------
# 2. INITIALIZE SQLITE DATABASE
# --------------------------------------------------

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Start a fresh audit every time the script runs
cursor.execute("DROP TABLE IF EXISTS seen_rows")
cursor.execute("DROP TABLE IF EXISTS feature_labels")

cursor.execute("""
    CREATE TABLE seen_rows (
        row_hash TEXT PRIMARY KEY
    )
""")

cursor.execute("""
    CREATE TABLE feature_labels (
        feature_hash TEXT PRIMARY KEY,
        first_label TEXT NOT NULL,
        conflict_rows INTEGER DEFAULT 0
    )
""")

conn.commit()

# --------------------------------------------------
# 3. ANALYZE DUPLICATES
# --------------------------------------------------

total_rows = 0
exact_duplicate_rows = 0

print("=" * 65)
print("       CYBERSHIELD AI - DUPLICATE ANALYSIS")
print("=" * 65)

for file_number, file in enumerate(parquet_files, start=1):

    df = pd.read_parquet(file)

    if "Label" not in df.columns:
        raise ValueError(f"Label column missing: {file}")

    feature_columns = [
        col for col in df.columns if col != "Label"
    ]

    # Hash complete rows, including labels
    row_hashes = (
        pd.util.hash_pandas_object(
            df,
            index=False,
            categorize=True
        )
        .astype("uint64")
        .astype(str)
    )

    # Hash features only, excluding labels
    feature_hashes = (
        pd.util.hash_pandas_object(
            df[feature_columns],
            index=False,
            categorize=True
        )
        .astype("uint64")
        .astype(str)
    )

    labels = df["Label"].astype(str).tolist()

    # Count exact duplicate rows across all files
    before = conn.total_changes

    cursor.executemany(
        "INSERT OR IGNORE INTO seen_rows (row_hash) VALUES (?)",
        [(h,) for h in row_hashes]
    )

    inserted = conn.total_changes - before

    exact_duplicate_rows += len(df) - inserted

    # Detect identical features with different labels
    cursor.executemany(
        """
        INSERT INTO feature_labels (
            feature_hash, first_label, conflict_rows
        )
        VALUES (?, ?, 0)

        ON CONFLICT(feature_hash) DO UPDATE SET
            conflict_rows = feature_labels.conflict_rows +
            CASE
                WHEN feature_labels.first_label <> excluded.first_label
                THEN 1
                ELSE 0
            END
        """,
        list(zip(feature_hashes, labels))
    )

    conn.commit()

    total_rows += len(df)

    print(
        f"Processed {file_number}/{len(parquet_files)} files | "
        f"Rows: {total_rows:,}"
    )

# --------------------------------------------------
# 4. FINAL RESULTS
# --------------------------------------------------

cursor.execute("SELECT COUNT(*) FROM seen_rows")
unique_complete_rows = cursor.fetchone()[0]

cursor.execute("""
    SELECT
        COUNT(*),
        COALESCE(SUM(conflict_rows), 0)
    FROM feature_labels
    WHERE conflict_rows > 0
""")

conflict_groups, conflict_rows = cursor.fetchone()

print("\n" + "=" * 65)
print("                 DUPLICATE ANALYSIS RESULTS")
print("=" * 65)

print(f"Total records: {total_rows:,}")
print(f"Unique complete rows: {unique_complete_rows:,}")
print(f"Exact duplicate occurrences: {exact_duplicate_rows:,}")

print("\nFEATURE-LABEL CONFLICT ANALYSIS")
print(f"Feature groups with different labels: {conflict_groups:,}")
print(f"Additional conflicting-label rows: {conflict_rows:,}")

print("\nAUDIT SUMMARY")
print(
    "Duplicate records found:",
    exact_duplicate_rows > 0
)

print(
    "Conflicting feature labels found:",
    conflict_groups > 0
)

print("\nNo records were deleted.")
print(f"Audit database: {DB_PATH}")

conn.close()

