import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
REPORT_PATH = PROJECT_ROOT / "reports"

CONFLICT_FILE = REPORT_PATH / "detailed_conflict_frequency.csv"
OUTPUT_FILE = REPORT_PATH / "verified_conflict_groups.csv"

REPORT_PATH.mkdir(exist_ok=True)

print("=" * 65)
print("   CYBERSHIELD AI - EXACT CONFLICT VERIFICATION")
print("=" * 65)

# Read hashes as strings to avoid integer precision or type issues.
conflict_df = pd.read_csv(
    CONFLICT_FILE,
    dtype={"Feature_Hash": "string"}
)

candidate_hashes = set(conflict_df["Feature_Hash"].dropna())

print(f"Candidate conflicting hashes: {len(candidate_hashes):,}")

parquet_files = sorted(DATA_PATH.rglob("*.parquet"))
matching_parts = []

# Collect only records belonging to candidate hashes.
for i, file in enumerate(parquet_files, start=1):
    df = pd.read_parquet(file)

    feature_columns = [c for c in df.columns if c != "Label"]

    hashes = pd.util.hash_pandas_object(
        df[feature_columns],
        index=False,
        categorize=True
    ).astype("uint64").astype(str)

    mask = hashes.isin(candidate_hashes)

    if mask.any():
        selected = df.loc[mask].copy()
        selected.insert(0, "Feature_Hash", hashes.loc[mask].values)
        matching_parts.append(selected)

    print(f"Scanned {i}/{len(parquet_files)} files")

if not matching_parts:
    raise RuntimeError("No matching records found. Check the input report.")

candidates = pd.concat(matching_parts, ignore_index=True)

feature_columns = [
    c for c in candidates.columns
    if c not in ["Feature_Hash", "Label"]
]

print(f"\nCandidate records collected: {len(candidates):,}")

results = []

# Verify actual feature values within every hash group.
for feature_hash, group in candidates.groupby("Feature_Hash", sort=False):

    # drop_duplicates treats matching missing values consistently.
    unique_feature_rows = group[feature_columns].drop_duplicates()

    label_counts = group["Label"].value_counts()

    if len(unique_feature_rows) == 1 and len(label_counts) > 1:
        status = "VERIFIED_TRUE_LABEL_CONFLICT"
    elif len(unique_feature_rows) == 1:
        status = "IDENTICAL_FEATURES_NO_LABEL_CONFLICT"
    else:
        status = "MULTIPLE_FEATURE_PATTERNS_REVIEW_REQUIRED"

    results.append({
        "Feature_Hash": feature_hash,
        "Total_Records": len(group),
        "Unique_Feature_Patterns": len(unique_feature_rows),
        "Number_of_Labels": len(label_counts),
        "Label_Counts": " | ".join(
            f"{label}: {count}"
            for label, count in label_counts.items()
        ),
        "Verification_Status": status
    })

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("                 VERIFICATION RESULTS")
print("=" * 65)

print(
    result_df["Verification_Status"]
    .value_counts()
    .to_string()
)

print("\nTotal candidate records:", f"{len(candidates):,}")
print("Total candidate hash groups:", f"{len(result_df):,}")

print("\nDetailed report saved:")
print(OUTPUT_FILE)