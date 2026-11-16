
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
)

REPORT_PATH = PROJECT_ROOT / "reports"
REPORT_PATH.mkdir(exist_ok=True)

parquet_files = sorted(DATA_PATH.rglob("*.parquet"))

# Store label combinations for each feature hash
feature_labels = defaultdict(set)
feature_examples = {}

print("=" * 65)
print("       CYBERSHIELD AI - LABEL CONFLICT INVESTIGATION")
print("=" * 65)

total_rows = 0

for i, file in enumerate(parquet_files, start=1):

    df = pd.read_parquet(file)

    features = df.drop(columns=["Label"])

    hashes = pd.util.hash_pandas_object(
        features,
        index=False,
        categorize=True
    ).astype("uint64")

    for row_hash, label in zip(hashes, df["Label"]):

        key = int(row_hash)
        feature_labels[key].add(str(label))

        # if key not in feature_examples:
        #     feature_examples[key] = features.iloc[
        #         len(feature_labels) * 0
        #     ].to_dict()

    total_rows += len(df)

    print(f"Processed {i}/{len(parquet_files)} files")

# Find conflicting groups
conflicts = {
    key: labels
    for key, labels in feature_labels.items()
    if len(labels) > 1
}

print("\nCONFLICT SUMMARY")
print("-" * 65)

print(f"Total records scanned: {total_rows:,}")
print(f"Unique feature groups: {len(feature_labels):,}")
print(f"Conflicting feature groups: {len(conflicts):,}")

# Save label combinations
rows = []

for key, labels in conflicts.items():
    rows.append({
        "Feature_Hash": key,
        "Number_of_Labels": len(labels),
        "Labels": " | ".join(sorted(labels))
    })

conflict_df = pd.DataFrame(rows)

conflict_df.to_csv(
    REPORT_PATH / "conflicting_label_groups.csv",
    index=False
)

print("\nExamples of conflicting label combinations:")

for _, row in conflict_df.head(20).iterrows():
    print(row["Labels"])

print("\nReport saved:")
print(REPORT_PATH / "conflicting_label_groups.csv")
