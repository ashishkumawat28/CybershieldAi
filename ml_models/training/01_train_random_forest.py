
import json
import joblib
import duckdb
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "datasets" / "processed" / "cicids2017_model_ready"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = ROOT / "ml_models" / "saved_models"
REPORT_DIR = ROOT / "reports"

FEATURE_FILE = REPORT_DIR / "model_feature_list.json"
LABEL_FILE = REPORT_DIR / "label_mapping.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "random_forest_baseline.joblib"

if MODEL_FILE.exists():
    raise FileExistsError(
        f"Model already exists: {MODEL_FILE}. "
        "Move the previous model before retraining."
    )

print("=" * 70)
print(" CYBERSHIELD AI - RANDOM FOREST BASELINE TRAINING")
print("=" * 70)

# --------------------------------------------------
# STEP 1: Load feature names and label mapping
# --------------------------------------------------

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)
    
sql_feature_columns = ", ".join(
    f'"{col}"' for col in feature_columns
)

label_df = pd.read_csv(LABEL_FILE)
label_df = label_df.sort_values("Label_Encoded")

class_ids = label_df["Label_Encoded"].astype(int).tolist()
class_names = label_df["Label"].astype(str).tolist()

print(f"\nFeatures: {len(feature_columns)}")
print(f"Classes: {len(class_names)}")

# --------------------------------------------------
# STEP 2: Create a reproducible stratified sample
# --------------------------------------------------

print("\nPreparing training sample...")
print("Maximum records per class: 100,000")

train_glob = (TRAIN_DIR / "**" / "*.parquet").as_posix()
train_glob_sql = train_glob.replace("'", "''")

def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'

selected_columns = [
    quote_identifier(col) for col in feature_columns
]

selected_columns.extend([
    quote_identifier("Label_Encoded"),
    quote_identifier("Label")
])

con = duckdb.connect()

# Set a fixed random seed for reproducibility.
con.execute("SELECT setseed(0.42)")

MAX_PER_CLASS = 100000

sample_query = f"""
    WITH randomized AS (
        SELECT
            {sql_feature_columns},
            Label_Encoded,
            Label,
            random() AS random_value
        FROM read_parquet('{train_glob}', union_by_name=true)
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

train_df = con.execute(sample_query).fetchdf()
con.close()

print("\nVerifying sampled dataset...")

missing_columns = [
    col for col in feature_columns
    if col not in train_df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing feature columns: {missing_columns}"
    )

print("All 70 feature columns found successfully.")
print("Training dataframe shape:", train_df.shape)

# train_df = train_df.drop(columns=["sample_rank"])

sample_distribution = (
    train_df.groupby(["Label", "Label_Encoded"])
    .size()
    .reset_index(name="Sample_Count")
    .sort_values("Label_Encoded")
)

sample_distribution.to_csv(
    REPORT_DIR / "random_forest_training_sample.csv",
    index=False
)

print(f"\nTraining sample records: {len(train_df):,}")
print("\nTraining sample distribution:")
print(sample_distribution.to_string(index=False))

# --------------------------------------------------
# STEP 3: Prepare training arrays
# --------------------------------------------------

X_train = train_df[feature_columns].to_numpy(
    dtype=np.float32
)

y_train = train_df["Label_Encoded"].to_numpy(
    dtype=np.int64
)

del train_df

if not np.isfinite(X_train).all():
    raise RuntimeError("Training data contains NaN or infinity.")

# --------------------------------------------------
# STEP 4: Create Random Forest model
# --------------------------------------------------

print("\nInitializing Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# --------------------------------------------------
# STEP 5: Train model
# --------------------------------------------------

print("\nStarting model training...")
model.fit(X_train, y_train)

print("\nModel training completed.")

# --------------------------------------------------
# STEP 6: Save trained model
# --------------------------------------------------

joblib.dump(model, MODEL_FILE, compress=3)

print("\nModel saved:")
print(MODEL_FILE)

del X_train, y_train

# --------------------------------------------------
# STEP 7: Evaluate on the COMPLETE test dataset
# --------------------------------------------------

print("\nEvaluating on complete test dataset...")

test_files = sorted(TEST_DIR.rglob("*.parquet"))

all_y_true = []
all_y_pred = []

for i, file in enumerate(test_files, start=1):

    df = pd.read_parquet(file)

    X_batch = df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y_batch = df["Label_Encoded"].to_numpy(
        dtype=np.int64
    )

    if not np.isfinite(X_batch).all():
        raise RuntimeError(
            f"Non-finite test features found in {file}"
        )

    predictions = model.predict(X_batch)

    all_y_true.append(y_batch)
    all_y_pred.append(predictions)

    print(f"Evaluated {i}/{len(test_files)} test files")

y_true = np.concatenate(all_y_true)
y_pred = np.concatenate(all_y_pred)

if len(y_true) != 578418:
    raise RuntimeError(
        f"Expected 578,418 test records, found {len(y_true):,}"
    )

# --------------------------------------------------
# STEP 8: Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=class_ids,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=class_ids,
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

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    REPORT_DIR / "random_forest_classification_report.csv"
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=class_ids
)

confusion_df = pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names
)

confusion_df.to_csv(
    REPORT_DIR / "random_forest_confusion_matrix.csv"
)

# --------------------------------------------------
# STEP 9: Save feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    REPORT_DIR / "random_forest_feature_importance.csv",
    index=False
)

# --------------------------------------------------
# STEP 10: Save overall metrics
# --------------------------------------------------

metrics = {
    "Model": "Random Forest",
    "Training_Sample_Size": int(len(sample_distribution) and sum(sample_distribution["Sample_Count"])),
    "Maximum_Samples_Per_Class": 100000,
    "Test_Size": int(len(y_true)),
    "Number_of_Features": len(feature_columns),
    "Number_of_Classes": len(class_names),
    "Accuracy": float(accuracy),
    "Macro_F1": float(macro_f1),
    "Weighted_F1": float(weighted_f1),
    "Random_State": 42,
    "Estimators": 100,
    "Maximum_Depth": 20
}

with open(
    REPORT_DIR / "random_forest_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

# --------------------------------------------------
# STEP 11: Display final results
# --------------------------------------------------

print("\n" + "=" * 70)
print("                 RANDOM FOREST RESULTS")
print("=" * 70)

print(f"Test records: {len(y_true):,}")
print(f"Accuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(report_df.to_string())

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved reports:")
print(REPORT_DIR / "random_forest_metrics.json")
print(REPORT_DIR / "random_forest_classification_report.csv")
print(REPORT_DIR / "random_forest_confusion_matrix.csv")
print(REPORT_DIR / "random_forest_feature_importance.csv")
print(REPORT_DIR / "random_forest_training_sample.csv")

print("\nStep 14 completed successfully.")
