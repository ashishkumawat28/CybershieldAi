
import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "datasets" / "processed" / "cicids2017_split" / "train"
TEST_DIR = ROOT / "datasets" / "processed" / "cicids2017_split" / "test"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

TRAIN_GLOB = (TRAIN_DIR / "**" / "*.parquet").as_posix()
TEST_GLOB = (TEST_DIR / "**" / "*.parquet").as_posix()

con = duckdb.connect()

def get_feature_columns(path):
    description = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}')"
    ).fetchdf()

    return [
        name for name in description["column_name"]
        if name != "Label"
    ]

train_features = get_feature_columns(TRAIN_GLOB)
test_features = get_feature_columns(TEST_GLOB)

print("=" * 65)
print(" CYBERSHIELD AI - FEATURE QUALITY AUDIT")
print("=" * 65)

# --------------------------------------------------
# STEP 1: Check feature consistency
# --------------------------------------------------

if set(train_features) != set(test_features):
    raise RuntimeError(
        "Train and test feature columns do not match."
    )

print(f"\nTraining features: {len(train_features)}")
print(f"Testing features:  {len(test_features)}")

# --------------------------------------------------
# STEP 2: Generate audit query
# --------------------------------------------------

def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'

def audit_dataset(path, split_name, features):

    expressions = []

    for col in features:
        q = quote_identifier(col)

        expressions.extend([
            f"COUNT({q}) AS {quote_identifier(col + '__non_null')}",
            f"COUNT(DISTINCT {q}) AS {quote_identifier(col + '__distinct')}",
            f"MIN({q}) AS {quote_identifier(col + '__min')}",
            f"MAX({q}) AS {quote_identifier(col + '__max')}"
        ])

    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            {", ".join(expressions)}
        FROM read_parquet('{path}', union_by_name = true)
    """

    result = con.execute(query).fetchdf().iloc[0]
    total_rows = int(result["total_rows"])

    rows = []

    for col in features:
        non_null = int(result[col + "__non_null"])
        distinct = int(result[col + "__distinct"])

        missing = total_rows - non_null

        minimum = result[col + "__min"]
        maximum = result[col + "__max"]

        rows.append({
            "Feature": col,
            "Total_Rows": total_rows,
            "Missing_Values": missing,
            "Missing_Percentage": round(
                missing / total_rows * 100, 6
            ),
            "Distinct_Values": distinct,
            "Minimum": minimum,
            "Maximum": maximum,
            "Is_Constant": distinct <= 1,
            "Split": split_name
        })

    return pd.DataFrame(rows)

# --------------------------------------------------
# STEP 3: Audit train and test
# --------------------------------------------------

print("\nAuditing training dataset...")
train_report = audit_dataset(
    TRAIN_GLOB, "Train", train_features
)

print("Auditing testing dataset...")
test_report = audit_dataset(
    TEST_GLOB, "Test", test_features
)

# --------------------------------------------------
# STEP 4: Save reports
# --------------------------------------------------

train_report.to_csv(
    REPORT_DIR / "train_feature_quality.csv",
    index=False
)

test_report.to_csv(
    REPORT_DIR / "test_feature_quality.csv",
    index=False
)

# --------------------------------------------------
# STEP 5: Display important findings
# --------------------------------------------------

print("\n" + "=" * 65)
print("                 TRAINING DATA RESULTS")
print("=" * 65)

print("Total rows:", f"{train_report['Total_Rows'].iloc[0]:,}")

print("\nFeatures with missing values:")
print(
    train_report[
        train_report["Missing_Values"] > 0
    ][[
        "Feature",
        "Missing_Values",
        "Missing_Percentage"
    ]].to_string(index=False)
)

print("\nConstant features:")
print(
    train_report[
        train_report["Is_Constant"]
    ]["Feature"].to_list()
)

print("\n" + "=" * 65)
print("                 TEST DATA RESULTS")
print("=" * 65)

print("Total rows:", f"{test_report['Total_Rows'].iloc[0]:,}")

print("\nFeatures with missing values:")
print(
    test_report[
        test_report["Missing_Values"] > 0
    ][[
        "Feature",
        "Missing_Values",
        "Missing_Percentage"
    ]].to_string(index=False)
)

print("\nConstant features:")
print(
    test_report[
        test_report["Is_Constant"]
    ]["Feature"].to_list()
)

print("\nReports saved:")
print(REPORT_DIR / "train_feature_quality.csv")
print(REPORT_DIR / "test_feature_quality.csv")

print("\nStep 12 completed successfully.")

con.close()
