
import pandas as pd
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
REPORT_PATH = PROJECT_ROOT / "reports"

CONFLICT_REPORT = REPORT_PATH / "detailed_conflict_frequency.csv"

BASELINE_DIR = PROJECT_ROOT / "datasets" / "processed" / "cicids2017_baseline"
QUARANTINE_DIR = PROJECT_ROOT / "datasets" / "processed" / "cicids2017_quarantine"
TEMP_DIR = PROJECT_ROOT / "datasets" / "processed" / "baseline_temp"

BASELINE_FILE = BASELINE_DIR / "baseline_unique.parquet"

REPORT_PATH.mkdir(exist_ok=True)

# Avoid accidentally mixing results from an earlier run.
if BASELINE_FILE.exists():
    raise FileExistsError(
        f"{BASELINE_FILE} already exists. Rename or move it before rerunning."
    )

if TEMP_DIR.exists() and any(TEMP_DIR.rglob("*.parquet")):
    raise FileExistsError(
        f"{TEMP_DIR} contains previous temporary files. "
        "Move or clear this generated temporary folder before rerunning."
    )

BASELINE_DIR.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("   CYBERSHIELD AI - BASELINE DATASET CREATION")
print("=" * 65)

# --------------------------------------------------
# STEP 1: Load verified conflicting feature hashes
# --------------------------------------------------

conflict_df = pd.read_csv(
    CONFLICT_REPORT,
    dtype={"Feature_Hash": "string"}
)

conflict_hashes = set(
    conflict_df["Feature_Hash"].dropna()
)

print(f"Verified conflicting groups: {len(conflict_hashes):,}")

# --------------------------------------------------
# STEP 2: Exclude all records in conflicting groups
# --------------------------------------------------

parquet_files = sorted(DATA_PATH.rglob("*.parquet"))

total_original = 0
total_quarantined = 0
total_remaining = 0

original_label_counts = {}
remaining_label_counts = {}

for i, file in enumerate(parquet_files, start=1):

    df = pd.read_parquet(file)
    total_original += len(df)

    for label, count in df["Label"].value_counts().items():
        original_label_counts[label] = (
            original_label_counts.get(label, 0) + int(count)
        )

    feature_columns = [
        column for column in df.columns if column != "Label"
    ]

    hashes = (
        pd.util.hash_pandas_object(
            df[feature_columns],
            index=False,
            categorize=True
        )
        .astype("uint64")
        .astype(str)
    )

    conflict_mask = hashes.isin(conflict_hashes)

    excluded = df.loc[conflict_mask]
    remaining = df.loc[~conflict_mask]

    total_quarantined += len(excluded)
    total_remaining += len(remaining)

    # Preserve excluded records separately.
    if not excluded.empty:
        quarantine_file = (
            QUARANTINE_DIR / file.relative_to(DATA_PATH)
        )
        quarantine_file.parent.mkdir(parents=True, exist_ok=True)
        excluded.to_parquet(quarantine_file, index=False)

    # Save non-conflicting records for global deduplication.
    if not remaining.empty:
        temp_file = TEMP_DIR / file.relative_to(DATA_PATH)
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        remaining.to_parquet(temp_file, index=False)

        for label, count in remaining["Label"].value_counts().items():
            remaining_label_counts[label] = (
                remaining_label_counts.get(label, 0) + int(count)
            )

    print(f"Processed {i}/{len(parquet_files)} files")

print("\nConflict exclusion completed.")

print(f"Original records: {total_original:,}")
print(f"Quarantined records: {total_quarantined:,}")
print(f"Records remaining: {total_remaining:,}")

if total_quarantined != 7020:
    raise RuntimeError(
        "Expected 7,020 quarantined records, but found "
        f"{total_quarantined:,}. Stop and investigate."
    )

# --------------------------------------------------
# STEP 3: Remove exact duplicate rows globally
# --------------------------------------------------

print("\nStarting global exact deduplication using DuckDB...")

temp_glob = (TEMP_DIR / "**" / "*.parquet").as_posix()
output_path = BASELINE_FILE.as_posix()

# Escape single quotes in SQL paths.
temp_glob_sql = temp_glob.replace("'", "''")
output_path_sql = output_path.replace("'", "''")

connection = duckdb.connect()

connection.execute(f"""
    COPY (
        SELECT DISTINCT *
        FROM read_parquet(
            '{temp_glob_sql}',
            union_by_name = true
        )
    )
    TO '{output_path_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

final_count = connection.execute(
    f"SELECT COUNT(*) FROM read_parquet('{output_path_sql}')"
).fetchone()[0]

final_distribution = connection.execute(f"""
    SELECT Label, COUNT(*) AS Record_Count
    FROM read_parquet('{output_path_sql}')
    GROUP BY Label
    ORDER BY Record_Count DESC
""").fetchdf()

connection.close()

duplicates_removed = total_remaining - final_count

# --------------------------------------------------
# STEP 4: Save reports
# --------------------------------------------------

summary = pd.DataFrame([
    {"Metric": "Original records", "Count": total_original},
    {"Metric": "Quarantined conflict records", "Count": total_quarantined},
    {"Metric": "Records after conflict exclusion", "Count": total_remaining},
    {"Metric": "Exact duplicate rows removed", "Count": duplicates_removed},
    {"Metric": "Final unique baseline records", "Count": final_count},
    {"Metric": "Original feature groups in conflict report",
     "Count": len(conflict_hashes)}
])

summary.to_csv(
    REPORT_PATH / "baseline_cleaning_summary.csv",
    index=False
)

final_distribution.to_csv(
    REPORT_PATH / "baseline_class_distribution.csv",
    index=False
)

# --------------------------------------------------
# STEP 5: Print final results
# --------------------------------------------------

print("\n" + "=" * 65)
print("                 FINAL RESULTS")
print("=" * 65)

print(f"Original records:              {total_original:,}")
print(f"Quarantined conflict records:  {total_quarantined:,}")
print(f"After conflict exclusion:      {total_remaining:,}")
print(f"Exact duplicates removed:      {duplicates_removed:,}")
print(f"Final baseline records:        {final_count:,}")

print("\nFINAL CLASS DISTRIBUTION")
print(final_distribution.to_string(index=False))

print("\nSaved baseline dataset:")
print(BASELINE_FILE)

print("\nSaved quarantine records:")
print(QUARANTINE_DIR)

print("\nSaved reports:")
print(REPORT_PATH / "baseline_cleaning_summary.csv")
print(REPORT_PATH / "baseline_class_distribution.csv")

print("\nStep 9 completed successfully.")
