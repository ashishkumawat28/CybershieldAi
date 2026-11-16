
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "datasets" / "raw"

CHUNK_SIZE = 100_000

csv_files = sorted(DATASET_PATH.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found!")

total_rows = 0
global_labels = Counter()
missing_counts = defaultdict(int)
infinite_counts = defaultdict(int)
file_results = []

print("=" * 70)
print("       CYBERSHIELD AI - FULL DATASET PROFILING")
print("=" * 70)

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    file_rows = 0
    file_labels = Counter()

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(
                f"Label column missing in {file.name}"
            )

        # Normalize labels
        labels = chunk["Label"].astype("string").str.strip()

        if labels.isna().any():
            print("Warning: Missing labels found!")

        valid_labels = labels.dropna()

        global_labels.update(valid_labels.tolist())
        file_labels.update(valid_labels.tolist())

        file_rows += len(chunk)

        # Analyze numerical features
        features = chunk.drop(columns=["Label"])

        numeric = features.apply(
            pd.to_numeric,
            errors="coerce"
        )

        missing = numeric.isna().sum()

        for column, count in missing.items():
            missing_counts[column] += int(count)

        infinite = np.isinf(numeric).sum()

        for column, count in infinite.items():
            infinite_counts[column] += int(count)

    total_rows += file_rows

    file_results.append({
        "file": file.name,
        "rows": file_rows,
        "labels": dict(file_labels)
    })

    print(f"Rows: {file_rows:,}")
    print(f"Labels: {dict(file_labels)}")

print("\n" + "=" * 70)
print("                    FINAL PROFILE")
print("=" * 70)

print(f"\nTotal CSV files: {len(csv_files)}")
print(f"Total records: {total_rows:,}")

print("\nGLOBAL ATTACK DISTRIBUTION")
for label, count in global_labels.most_common():
    percentage = (count / total_rows) * 100

    print(
        f"{label:30s} "
        f"{count:12,} "
        f"({percentage:.2f}%)"
    )

print("\nTOP FEATURES WITH MISSING VALUES")
for column, count in sorted(
    missing_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    if count > 0:
        print(f"{column}: {count:,}")

print("\nFEATURES WITH INFINITE VALUES")
for column, count in sorted(
    infinite_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    if count > 0:
        print(f"{column}: {count:,}")

# Save profiling reports
report_dir = PROJECT_ROOT / "reports"
report_dir.mkdir(exist_ok=True)

pd.DataFrame(file_results).to_json(
    report_dir / "dataset_profile.json",
    orient="records",
    indent=4
)

pd.DataFrame(
    global_labels.items(),
    columns=["Label", "Count"]
).to_csv(
    report_dir / "attack_distribution.csv",
    index=False
)

print("\nReports saved successfully!")
print("reports/dataset_profile.json")
print("reports/attack_distribution.csv")
