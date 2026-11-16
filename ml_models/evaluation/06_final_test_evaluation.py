
from pathlib import Path
import json
import time

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = (
    ROOT / "datasets" / "processed" /
    "cicids2017_model_ready" / "test"
)

MODEL_DIR = ROOT / "ml_models" / "saved_models"
REPORT_DIR = ROOT / "reports"

FEATURE_FILE = REPORT_DIR / "model_feature_list.json"
THRESHOLD_FILE = REPORT_DIR / "calibrated_alert_thresholds.json"

RF_MODEL_PATH = MODEL_DIR / "random_forest_validated.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_validated.joblib"

LABEL_COLUMN = "Label_Encoded"
BENIGN_CLASS_ID = 0

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
    thresholds = json.load(f)

models = {
    "Random Forest": {
        "model": joblib.load(RF_MODEL_PATH),
        "threshold": thresholds["random_forest"]["threshold"]
    },
    "XGBoost": {
        "model": joblib.load(XGB_MODEL_PATH),
        "threshold": thresholds["xgboost"]["threshold"]
    }
}

test_files = list(TEST_DIR.rglob("*.parquet"))

if not test_files:
    raise FileNotFoundError(
        f"No test Parquet files found: {TEST_DIR}"
    )

test_glob = (
    str(TEST_DIR / "**" / "*.parquet")
    .replace("\\", "/")
)

print("=" * 70)
print("CYBERSHIELD AI - FINAL TEST EVALUATION")
print("=" * 70)

print(f"Test files: {len(test_files)}")
print(f"Features: {len(FEATURES)}")

for name, item in models.items():
    model = item["model"]

    if model.n_features_in_ != len(FEATURES):
        raise ValueError(f"{name}: feature count mismatch.")

    print(
        f"{name} calibrated threshold: "
        f"{item['threshold']:.6f}"
    )


# ============================================================
# 3. PREPARE DUCKDB QUERY
# ============================================================

quoted_features = ", ".join(
    '"' + feature.replace('"', '""') + '"'
    for feature in FEATURES
)

columns = f'{quoted_features}, "{LABEL_COLUMN}"'

query = f"""
SELECT {columns}
FROM read_parquet('{test_glob}', union_by_name=true)
"""

con = duckdb.connect()

print("\nPreparing untouched test dataset...")

# First, count the test records
total_records = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_parquet('{test_glob}', union_by_name=true)
    """
).fetchone()[0]

print(f"Total test records: {total_records:,}")

# Execute the actual test query AFTER the count query
con.execute(query)

print("Test data query executed successfully.")

# ============================================================
# 4. INITIALIZE RESULT STORAGE
# ============================================================

storage = {}

for name in models:

    storage[name] = {
        "y_true": [],
        "y_binary": [],
        "attack_scores": [],
        "multiclass_predictions": [],
        "binary_predictions": [],
        "inference_seconds": 0.0
    }


# ============================================================
# 5. PROCESS TEST DATA IN CHUNKS
# ============================================================

print("\nEvaluating both models...")

processed = 0
start_total = time.time()

while True:

    chunk = con.fetch_df_chunk(vectors_per_chunk=100)

    if chunk.empty:
        break

    X = chunk[FEATURES].astype(np.float32)
    y = chunk[LABEL_COLUMN].to_numpy(dtype=np.int32)

    y_binary = (
        y != BENIGN_CLASS_ID
    ).astype(np.int32)

    for name, item in models.items():

        model = item["model"]
        threshold = item["threshold"]

        start = time.time()

        probabilities = model.predict_proba(X)

        elapsed = time.time() - start

        storage[name]["inference_seconds"] += elapsed

        benign_index = list(model.classes_).index(
            BENIGN_CLASS_ID
        )

        # Multiclass prediction
        predicted_classes = model.classes_[
            np.argmax(probabilities, axis=1)
        ]

        # Binary attack score
        attack_scores = (
            1.0 - probabilities[:, benign_index]
        )

        # Binary prediction using validation threshold
        binary_predictions = (
            attack_scores >= threshold
        ).astype(np.int32)

        storage[name]["y_true"].append(y.copy())

        storage[name]["y_binary"].append(
            y_binary.copy()
        )

        storage[name]["attack_scores"].append(
            attack_scores.copy()
        )

        storage[name]["multiclass_predictions"].append(
            predicted_classes.copy()
        )

        storage[name]["binary_predictions"].append(
            binary_predictions.copy()
        )

    processed += len(chunk)

    print(
        f"Processed {processed:,} / "
        f"{total_records:,} records"
    )

con.close()

total_time = time.time() - start_total

print(f"\nEvaluation processing time: {total_time:.2f} seconds")


# ============================================================
# 6. CALCULATE FINAL METRICS
# ============================================================

all_results = []

for name, item in models.items():

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {name}")
    print("=" * 70)

    data = storage[name]

    y_true = np.concatenate(data["y_true"])
    y_binary = np.concatenate(data["y_binary"])

    attack_scores = np.concatenate(
        data["attack_scores"]
    )

    multiclass_predictions = np.concatenate(
        data["multiclass_predictions"]
    )

    binary_predictions = np.concatenate(
        data["binary_predictions"]
    )

    threshold = item["threshold"]

    # --------------------------------------------------------
    # Multiclass metrics
    # --------------------------------------------------------

    multiclass_accuracy = accuracy_score(
        y_true,
        multiclass_predictions
    )

    multiclass_macro_f1 = f1_score(
        y_true,
        multiclass_predictions,
        average="macro",
        zero_division=0
    )

    multiclass_weighted_f1 = f1_score(
        y_true,
        multiclass_predictions,
        average="weighted",
        zero_division=0
    )

    class_ids = item["model"].classes_

    class_report = classification_report(
        y_true,
        multiclass_predictions,
        labels=class_ids,
        output_dict=True,
        zero_division=0
    )

    class_report_df = pd.DataFrame(class_report).transpose()

    safe_name = name.lower().replace(" ", "_")

    class_report_df.to_csv(
        REPORT_DIR / f"{safe_name}_final_test_classification_report.csv"
    )

    # --------------------------------------------------------
    # Binary metrics
    # --------------------------------------------------------

    binary_accuracy = accuracy_score(
        y_binary,
        binary_predictions
    )

    attack_precision = precision_score(
        y_binary,
        binary_predictions,
        zero_division=0
    )

    attack_recall = recall_score(
        y_binary,
        binary_predictions,
        zero_division=0
    )

    attack_f1 = f1_score(
        y_binary,
        binary_predictions,
        zero_division=0
    )

    binary_roc_auc = roc_auc_score(
        y_binary,
        attack_scores
    )

    binary_ap = average_precision_score(
        y_binary,
        attack_scores
    )

    tn, fp, fn, tp = confusion_matrix(
        y_binary,
        binary_predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    binary_cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual BENIGN", "Actual ATTACK"],
        columns=["Predicted BENIGN", "Predicted ATTACK"]
    )

    binary_cm.to_csv(
        REPORT_DIR / f"{safe_name}_final_test_binary_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"Calibrated threshold: {threshold:.6f}")

    print("\nMulticlass performance:")
    print(f"Accuracy:       {multiclass_accuracy:.6f}")
    print(f"Macro F1:       {multiclass_macro_f1:.6f}")
    print(f"Weighted F1:    {multiclass_weighted_f1:.6f}")

    print("\nBinary attack detection:")
    print(f"Accuracy:       {binary_accuracy:.6f}")
    print(f"Precision:      {attack_precision:.6f}")
    print(f"Recall:         {attack_recall:.6f}")
    print(f"F1 Score:       {attack_f1:.6f}")
    print(f"ROC-AUC:        {binary_roc_auc:.6f}")
    print(f"Average Precision: {binary_ap:.6f}")

    print("\nError rates:")
    print(f"False Positive Rate: {fpr:.6f}")
    print(f"False Negative Rate: {fnr:.6f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn:,} | FP: {fp:,}")
    print(f"FN: {fn:,} | TP: {tp:,}")

    print(
        f"\nInference time: "
        f"{data['inference_seconds']:.2f} seconds"
    )

    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    result = {
        "model": name,
        "test_records": int(len(y_true)),
        "calibrated_threshold": float(threshold),

        "multiclass_accuracy": float(multiclass_accuracy),
        "multiclass_macro_f1": float(multiclass_macro_f1),
        "multiclass_weighted_f1": float(multiclass_weighted_f1),

        "binary_accuracy": float(binary_accuracy),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),

        "binary_roc_auc": float(binary_roc_auc),
        "binary_average_precision": float(binary_ap),

        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "inference_time_seconds": float(
            data["inference_seconds"]
        )
    }

    all_results.append(result)


# ============================================================
# 7. SAVE FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame(all_results)

comparison.to_csv(
    REPORT_DIR / "final_test_model_comparison.csv",
    index=False
)

with open(
    REPORT_DIR / "final_test_evaluation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "dataset": "CICIDS2017 untouched test dataset",
            "test_records": int(total_records),
            "threshold_source": "Separate validation dataset",
            "results": all_results
        },
        f,
        indent=4
    )


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 70)

print(
    comparison[
        [
            "model",
            "calibrated_threshold",
            "multiclass_accuracy",
            "multiclass_macro_f1",
            "binary_accuracy",
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "false_positive_rate",
            "false_negative_rate"
        ]
    ].to_string(index=False)
)

print("\nSaved reports:")
print("1. final_test_model_comparison.csv")
print("2. final_test_evaluation_summary.json")
print("3. random_forest_final_test_classification_report.csv")
print("4. xgboost_final_test_classification_report.csv")
print("5. Individual binary confusion matrices")

print("\nThresholds were selected using validation data.")
print("Test data was used only for final evaluation.")

print("\nSTEP 19.4 COMPLETED SUCCESSFULLY")
print("=" * 70)
