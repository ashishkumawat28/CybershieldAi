
import json
import time
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight


warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "cicids2017_model_ready"
    / "train"
)

TEST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "cicids2017_model_ready"
    / "test"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT / "reports" / "model_feature_list.json"
)

MODEL_DIR = PROJECT_ROOT / "ml_models" / "saved_models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 100_000
RANDOM_STATE = 42
TEST_BATCH_SIZE = 100_000

MODEL_PATH = MODEL_DIR / "xgboost_baseline.joblib"
NATIVE_MODEL_PATH = MODEL_DIR / "xgboost_baseline.json"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_feature_list(path):
    """Load the exact feature list created during preprocessing."""

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        features = data
    elif isinstance(data, dict) and "features" in data:
        features = data["features"]
    else:
        raise ValueError(
            "Unexpected format in model_feature_list.json"
        )

    if not features:
        raise ValueError("Feature list is empty.")

    return features


def save_json(data, path):
    """Save a JSON report."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


# ============================================================
# 3. VERIFY INPUT FILES
# ============================================================

print("\n" + "=" * 65)
print("CYBERSHIELD AI - XGBOOST TRAINING")
print("=" * 65)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Testing directory not found: {TEST_DIR}")

if not FEATURE_LIST_PATH.exists():
    raise FileNotFoundError(
        f"Feature list not found: {FEATURE_LIST_PATH}"
    )

# Search recursively, including files inside subfolders.
train_files = list(TRAIN_DIR.rglob("*.parquet"))
test_files = list(TEST_DIR.rglob("*.parquet"))

print("\nDataset path verification:")
print("Training directory:", TRAIN_DIR.resolve())
print("Testing directory:", TEST_DIR.resolve())

print("Training files found:", len(train_files))
print("Testing files found:", len(test_files))

if train_files:
    print("First training file:", train_files[0])

if test_files:
    print("First testing file:", test_files[0])

if not train_files:
    raise FileNotFoundError("No training Parquet files found.")

if not test_files:
    raise FileNotFoundError("No testing Parquet files found.")

feature_columns = load_feature_list(FEATURE_LIST_PATH)

print(f"\nTraining Parquet files: {len(train_files)}")
print(f"Testing Parquet files:  {len(test_files)}")
print(f"Number of features:     {len(feature_columns)}")

if len(feature_columns) != 70:
    raise RuntimeError(
        f"Expected 70 features, found {len(feature_columns)}. "
        "Check model_feature_list.json."
    )


# ============================================================
# 4. CONNECT TO DUCKDB
# ============================================================

con = duckdb.connect()

# DuckDB random() uses a seed in the range [-1, 1].
con.execute("SELECT setseed(0.42)")

train_glob = (str(TRAIN_DIR / "**" / "*.parquet")).replace("\\", "/")
test_glob = (str(TEST_DIR / "**" / "*.parquet")).replace("\\", "/")

sql_feature_columns = ", ".join(
    f'"{column}"' for column in feature_columns
)

print("\nDuckDB connection established.")


# ============================================================
# 5. CREATE STRATIFIED TRAINING SAMPLE
# ============================================================

print("\nCreating training sample...")
print(f"Maximum records per class: {MAX_PER_CLASS:,}")

sampling_query = f"""
WITH randomized AS (
    SELECT
        {sql_feature_columns},
        Label_Encoded,
        Label,
        random() AS random_value
    FROM read_parquet('{train_glob}')
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY Label_Encoded
            ORDER BY random_value
        ) AS sample_rank
    FROM randomized
)
SELECT
    {sql_feature_columns},
    Label_Encoded,
    Label
FROM ranked
WHERE sample_rank <= {MAX_PER_CLASS}
"""

start_time = time.time()

train_df = con.execute(sampling_query).fetchdf()

print(f"Training sample loaded in {time.time() - start_time:.2f} seconds")
print("Training sample shape:", train_df.shape)

# Check feature columns.
missing_columns = [
    column
    for column in feature_columns
    if column not in train_df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing feature columns: {missing_columns}"
    )

# Verify labels.
if train_df["Label_Encoded"].isna().any():
    raise RuntimeError("Training labels contain missing values.")

train_df["Label_Encoded"] = train_df["Label_Encoded"].astype(np.int32)

class_mapping = (
    train_df[["Label_Encoded", "Label"]]
    .drop_duplicates()
    .sort_values("Label_Encoded")
)

class_ids = class_mapping["Label_Encoded"].astype(int).tolist()
class_names = class_mapping["Label"].astype(str).tolist()

expected_class_ids = list(range(len(class_ids)))

if class_ids != expected_class_ids:
    raise RuntimeError(
        "Encoded labels must be contiguous integers starting at zero. "
        f"Found: {class_ids}"
    )

if len(class_ids) != 15:
    raise RuntimeError(
        f"Expected 15 classes, found {len(class_ids)}."
    )

# Save the class distribution of the training sample.
sample_distribution = (
    train_df["Label"]
    .value_counts()
    .rename_axis("Attack_Type")
    .reset_index(name="Records")
)

sample_distribution.to_csv(
    REPORT_DIR / "xgboost_training_sample.csv",
    index=False
)

print("\nTraining sample class distribution:")
print(sample_distribution.to_string(index=False))


# ============================================================
# 6. PREPARE FEATURES AND LABELS
# ============================================================

print("\nPreparing training features...")

X_train = train_df[feature_columns].to_numpy(
    dtype=np.float32
)

y_train = train_df["Label_Encoded"].to_numpy(
    dtype=np.int32
)

if not np.isfinite(X_train).all():
    raise RuntimeError(
        "Training features contain NaN or infinite values."
    )

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

del train_df


# ============================================================
# 7. HANDLE CLASS IMBALANCE
# ============================================================

print("\nCalculating balanced sample weights...")

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("Sample weights calculated.")
print(
    "Minimum sample weight:",
    float(sample_weights.min())
)
print(
    "Maximum sample weight:",
    float(sample_weights.max())
)


# ============================================================
# 8. INITIALIZE XGBOOST CLASSIFIER
# ============================================================

print("\nInitializing XGBoost classifier...")

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_ids),
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1
)

print(model)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\n" + "=" * 65)
print("STARTING XGBOOST TRAINING")
print("=" * 65)

training_start = time.time()

# model.fit(
#     X_train,
#     y_train,
#     sample_weight=sample_weights,
#     verbose=True
# )

# Load previously trained XGBoost model.
print("\nLoading previously trained XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Saved XGBoost model loaded successfully.")



training_time = time.time() - training_start

print("\nXGBoost training completed.")
print(f"Training time: {training_time / 60:.2f} minutes")


# ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(model, MODEL_PATH)
model.save_model(str(NATIVE_MODEL_PATH))

print("Joblib model saved:", MODEL_PATH)
print("Native XGBoost model saved:", NATIVE_MODEL_PATH)




# ============================================================
# 11. EVALUATE ON THE COMPLETE TEST DATASET
# ============================================================

print("\n" + "=" * 65)
print("EVALUATING ON COMPLETE TEST DATASET")
print("=" * 65)


# Build test query using the same 70 model-ready features.
test_query = f"""
SELECT
    {sql_feature_columns},
    Label_Encoded
FROM read_parquet('{test_glob}')
"""


# Execute test query once.
con.execute(test_query)

all_true = []
all_predicted = []

total_test_records = 0
batch_number = 0
evaluation_start = time.time()

while True:

    # Fetch one chunk from DuckDB.
    test_df = con.fetch_df_chunk(
        vectors_per_chunk=100
    )

    if test_df.empty:
        break

    batch_number += 1

    X_batch = test_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_batch = test_df["Label_Encoded"].to_numpy(
        dtype=np.int32
    )

    if not np.isfinite(X_batch).all():
        raise RuntimeError(
            f"Invalid test features in batch {batch_number}."
        )

    predictions = model.predict(X_batch)

    all_true.append(y_batch)
    all_predicted.append(
        predictions.astype(np.int32)
    )

    total_test_records += len(y_batch)

    print(
        f"Batch {batch_number}: "
        f"{len(y_batch):,} records evaluated | "
        f"Total: {total_test_records:,}"
    )

if total_test_records == 0:
    raise RuntimeError("No test records were evaluated.")

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_predicted)

evaluation_time = time.time() - evaluation_start

print("\nTest evaluation completed.")
print(f"Total test records: {total_test_records:,}")
print(f"Evaluation time: {evaluation_time / 60:.2f} minutes")


# ============================================================
# 12. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=class_ids,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

conf_matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=class_ids
)

print("\n" + "=" * 65)
print("XGBOOST FINAL RESULTS")
print("=" * 65)

print(f"Accuracy:   {accuracy:.6f}")
print(f"Macro F1:   {macro_f1:.6f}")
print(f"Weighted F1:{weighted_f1:.6f}")

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 13. SAVE CLASSIFICATION REPORT
# ============================================================

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    REPORT_DIR / "xgboost_classification_report.csv",
    index=True
)

# Save confusion matrix with class names.
confusion_df = pd.DataFrame(
    conf_matrix,
    index=class_names,
    columns=class_names
)

confusion_df.index.name = "Actual"
confusion_df.to_csv(
    REPORT_DIR / "xgboost_confusion_matrix.csv"
)


# ============================================================
# 14. SAVE FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

booster = model.get_booster()

gain_scores = booster.get_score(
    importance_type="gain"
)

importance_values = [
    float(gain_scores.get(f"f{index}", 0.0))
    for index in range(len(feature_columns))
]

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Gain_Importance": importance_values
})

importance_total = importance_df["Gain_Importance"].sum()

if importance_total > 0:
    importance_df["Normalized_Importance"] = (
        importance_df["Gain_Importance"] / importance_total
    )
else:
    importance_df["Normalized_Importance"] = 0.0

importance_df = importance_df.sort_values(
    "Normalized_Importance",
    ascending=False
)

importance_df.to_csv(
    REPORT_DIR / "xgboost_feature_importance.csv",
    index=False
)

print("\nTop 15 important features:")
print(importance_df.head(15).to_string(index=False))


# ============================================================
# 15. SAVE FINAL METRICS
# ============================================================

metrics = {
    "model": "XGBoost",
    "dataset": "CICIDS2017",
    "number_of_features": len(feature_columns),
    "number_of_classes": len(class_ids),
    "training_sample_records": int(len(y_train)),
    "test_records": int(total_test_records),
    "training_time_minutes": round(training_time / 60, 2),
    "evaluation_time_minutes": round(evaluation_time / 60, 2),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "class_names": class_names,
    "class_ids": class_ids,
    "configuration": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "tree_method": "hist",
        "sample_weight": "balanced"
    }
}

save_json(
    metrics,
    REPORT_DIR / "xgboost_metrics.json"
)


# ============================================================
# 16. CLEANUP AND COMPLETION
# ============================================================

con.close()

print("\n" + "=" * 65)
print("STEP 15 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSaved model:")
print(MODEL_PATH)

print("\nSaved reports:")
print(REPORT_DIR / "xgboost_metrics.json")
print(REPORT_DIR / "xgboost_classification_report.csv")
print(REPORT_DIR / "xgboost_confusion_matrix.csv")
print(REPORT_DIR / "xgboost_feature_importance.csv")
print(REPORT_DIR / "xgboost_training_sample.csv")

print("\nFinal metrics:")
print(f"Accuracy:    {accuracy:.4%}")
print(f"Macro F1:    {macro_f1:.4%}")
print(f"Weighted F1: {weighted_f1:.4%}")

print("\nCyberShield AI - XGBoost training finished.")
