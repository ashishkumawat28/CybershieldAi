
from pathlib import Path
import json
import time

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    ROOT / "datasets" / "processed" /
    "cicids2017_validation_split" / "train"
)

MODEL_DIR = ROOT / "ml_models" / "saved_models"
REPORT_DIR = ROOT / "reports"

FEATURE_PATH = REPORT_DIR / "model_feature_list.json"

LABEL_COLUMN = "Label_Encoded"
MAX_PER_CLASS = 100000
RANDOM_STATE = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD FEATURE LIST
# ============================================================

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

train_files = list(TRAIN_DIR.rglob("*.parquet"))

if not train_files:
    raise FileNotFoundError("Training Parquet files not found.")

train_glob = (
    str(TRAIN_DIR / "**" / "*.parquet")
    .replace("\\", "/")
)

print("=" * 65)
print("CYBERSHIELD AI - VALIDATED MODEL RETRAINING")
print("=" * 65)

print(f"Training files: {len(train_files)}")
print(f"Features: {len(FEATURES)}")


# ============================================================
# 3. SAMPLE TRAINING DATA
# ============================================================

print("\nPreparing balanced training sample...")

con = duckdb.connect()

con.execute("SELECT setseed(0.42)")

quoted_features = ", ".join(
    '"' + feature.replace('"', '""') + '"'
    for feature in FEATURES
)

columns = f'{quoted_features}, "{LABEL_COLUMN}"'


sample_query = f"""
WITH randomized AS MATERIALIZED (
    SELECT
        {columns},
        random() AS random_value
    FROM read_parquet('{train_glob}', union_by_name=true)
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY "{LABEL_COLUMN}"
            ORDER BY random_value
        ) AS row_num
    FROM randomized
)
SELECT {columns}
FROM ranked
WHERE row_num <= {MAX_PER_CLASS}
"""

start = time.time()
train_df = con.execute(sample_query).fetchdf()
con.close()

print(f"Sample shape: {train_df.shape}")
print(f"Sampling time: {(time.time() - start) / 60:.2f} minutes")

X_train = train_df[FEATURES].astype(np.float32)
y_train = train_df[LABEL_COLUMN].astype(np.int32)

class_distribution = (
    y_train.value_counts()
    .sort_index()
    .rename_axis("class_id")
    .reset_index(name="records")
)

class_distribution.to_csv(
    REPORT_DIR / "validated_training_sample_distribution.csv",
    index=False
)

print("\nTraining class distribution:")
print(class_distribution.to_string(index=False))


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST")
print("=" * 65)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

start = time.time()

rf_model.fit(X_train, y_train)

rf_time = time.time() - start

rf_path = MODEL_DIR / "random_forest_validated.joblib"

joblib.dump(rf_model, rf_path)

print(f"Random Forest training time: {rf_time / 60:.2f} minutes")
print(f"Saved: {rf_path}")


# ============================================================
# 5. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 65)
print("TRAINING XGBOOST")
print("=" * 65)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=15,
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

start = time.time()

xgb_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    verbose=True
)

xgb_time = time.time() - start

xgb_path = MODEL_DIR / "xgboost_validated.joblib"

joblib.dump(xgb_model, xgb_path)

print(f"XGBoost training time: {xgb_time / 60:.2f} minutes")
print(f"Saved: {xgb_path}")


# ============================================================
# 6. SAVE TRAINING SUMMARY
# ============================================================

summary = {
    "training_records": int(len(train_df)),
    "feature_count": len(FEATURES),
    "class_count": int(y_train.nunique()),
    "max_per_class": MAX_PER_CLASS,
    "random_state": RANDOM_STATE,
    "random_forest_training_minutes": round(rf_time / 60, 3),
    "xgboost_training_minutes": round(xgb_time / 60, 3),
    "random_forest_model": str(rf_path.relative_to(ROOT)),
    "xgboost_model": str(xgb_path.relative_to(ROOT))
}

with open(
    REPORT_DIR / "validated_training_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)
print("VALIDATED MODEL TRAINING COMPLETED")
print("=" * 65)

print(json.dumps(summary, indent=4))
