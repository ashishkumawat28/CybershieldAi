
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "datasets" / "processed" / "cicids2017_clean"
)

REPORT_PATH = PROJECT_ROOT / "reports"
REPORT_PATH.mkdir(exist_ok=True)

parquet_files = sorted(DATA_PATH.rglob("*.parquet"))

# Feature hash -> label -> occurrence count
feature_label_counts = defaultdict(Counter)

print("=" * 65)
print("   CYBERSHIELD AI - CONFLICT FREQUENCY ANALYSIS")
print("=" * 65)

for i, file in enumerate(parquet_files, start=1):

    df = pd.read_parquet(file)

    features = df.drop(columns=["Label"])

    hashes = pd.util.hash_pandas_object(
        features,
        index=False,
        categorize=True
    )

    temp = pd.DataFrame({
        "Feature_Hash": hashes,
        "Label": df["Label"].astype(str)
    })

    grouped = temp.groupby(
        ["Feature_Hash", "Label"]
    ).size()

    for (feature_hash, label), count in grouped.items():
        feature_label_counts[int(feature_hash)][label] += int(count)

    print(f"Processed {i}/{len(parquet_files)} files")


# --------------------------------------------------
# IDENTIFY CONFLICTING GROUPS
# --------------------------------------------------

conflict_groups = {
    feature_hash: counts
    for feature_hash, counts in feature_label_counts.items()
    if len(counts) > 1
}

report_rows = []

total_conflict_records = 0
tie_groups = 0

for feature_hash, counts in conflict_groups.items():

    group_total = sum(counts.values())
    total_conflict_records += group_total

    maximum_count = max(counts.values())

    majority_labels = [
        label for label, count in counts.items()
        if count == maximum_count
    ]

    if len(majority_labels) > 1:
        tie_groups += 1

    report_rows.append({
        "Feature_Hash": feature_hash,
        "Total_Records": group_total,
        "Number_of_Labels": len(counts),
        "Label_Counts": " | ".join(
            f"{label}: {count}"
            for label, count in counts.most_common()
        ),
        "Majority_Label": (
            majority_labels[0]
            if len(majority_labels) == 1
            else "TIE"
        )
    })


conflict_report = pd.DataFrame(report_rows)

if not conflict_report.empty:
    conflict_report = conflict_report.sort_values(
        "Total_Records",
        ascending=False
    )

conflict_report.to_csv(
    REPORT_PATH / "detailed_conflict_frequency.csv",
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("                 FINAL RESULTS")
print("=" * 65)

print(f"Unique feature groups: {len(feature_label_counts):,}")
print(f"Conflicting feature groups: {len(conflict_groups):,}")
print(f"Records belonging to conflicting groups: {total_conflict_records:,}")
print(f"Groups with tied majority labels: {tie_groups:,}")

print("\nTOP 20 CONFLICTING GROUPS")

if not conflict_report.empty:
    print(
        conflict_report.head(20).to_string(index=False)
    )

print("\nReport saved:")
print(REPORT_PATH / "detailed_conflict_frequency.csv")
