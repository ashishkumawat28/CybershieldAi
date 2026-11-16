
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_FILE = (
    ROOT / "datasets" / "processed" /
    "cicids2017_validation_split" /
    "validation" / "validation.parquet"
)

MODEL_DIR = ROOT / "ml_models" / "saved_models"
REPORT_DIR = ROOT / "reports"

FEATURE_FILE = REPORT_DIR / "model_feature_list.json"

RF_MODEL_PATH = MODEL_DIR / "random_forest_validated.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_validated.joblib"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLUMN = "Label_Encoded"
BENIGN_CLASS_ID = 0


# ============================================================
# 2. LOAD FEATURES
# ============================================================

with open(FEATURE_FILE, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)


# ============================================================
# 3. LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("CYBERSHIELD AI - ALERT THRESHOLD CALIBRATION")
print("=" * 70)

print("\nLoading validation dataset...")

validation_df = pd.read_parquet(VALIDATION_FILE)

missing_features = [
    feature for feature in FEATURES
    if feature not in validation_df.columns
]

if missing_features:
    raise ValueError(
        f"Missing validation features: {missing_features}"
    )

if LABEL_COLUMN not in validation_df.columns:
    raise ValueError(f"{LABEL_COLUMN} not found.")

X_validation = validation_df[FEATURES].astype(np.float32)
y_validation = validation_df[LABEL_COLUMN].astype(np.int32)

print(f"Validation records: {len(validation_df):,}")
print(f"Features: {len(FEATURES)}")
print(f"Classes: {y_validation.nunique()}")

print("\nValidation class distribution:")
print(y_validation.value_counts().sort_index().to_string())


# ============================================================
# 4. LOAD VALIDATED MODELS
# ============================================================

print("\nLoading validated models...")

models = {
    "Random Forest": joblib.load(RF_MODEL_PATH),
    "XGBoost": joblib.load(XGB_MODEL_PATH)
}

for model_name, model in models.items():

    if model.n_features_in_ != len(FEATURES):
        raise ValueError(
            f"{model_name} feature count mismatch."
        )

    if BENIGN_CLASS_ID not in model.classes_:
        raise ValueError(
            f"{model_name} does not contain BENIGN class ID 0."
        )

print("Both models loaded successfully.")


# ============================================================
# 5. CALIBRATION FUNCTION
# ============================================================

def calibrate_model(model_name, model):

    print("\n" + "=" * 70)
    print(f"CALIBRATING: {model_name}")
    print("=" * 70)

    start_time = time.time()

    probabilities = model.predict_proba(X_validation)

    inference_time = time.time() - start_time

    # Multiclass predictions
    predicted_classes = model.classes_[
        np.argmax(probabilities, axis=1)
    ]

    multiclass_report = classification_report(
        y_validation,
        predicted_classes,
        output_dict=True,
        zero_division=0
    )

    multiclass_accuracy = accuracy_score(
        y_validation,
        predicted_classes
    )

    multiclass_macro_f1 = f1_score(
        y_validation,
        predicted_classes,
        average="macro",
        zero_division=0
    )

    # Find BENIGN probability column
    benign_index = list(model.classes_).index(
        BENIGN_CLASS_ID
    )

    # Binary labels:
    # 0 = BENIGN
    # 1 = ATTACK

    y_binary = (
        y_validation.to_numpy() != BENIGN_CLASS_ID
    ).astype(np.int32)

    # Total probability of all attack classes
    attack_scores = 1.0 - probabilities[:, benign_index]

    # Binary ROC-AUC and Average Precision
    binary_roc_auc = roc_auc_score(
        y_binary,
        attack_scores
    )

    binary_average_precision = average_precision_score(
        y_binary,
        attack_scores
    )

    # Precision-recall threshold search
    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_binary,
            attack_scores
        )
    )

    if len(thresholds) == 0:
        raise ValueError("No thresholds generated.")

    f1_values = (
        2 * precision_values[:-1] * recall_values[:-1]
        /
        (
            precision_values[:-1] +
            recall_values[:-1] + 1e-12
        )
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    # Binary predictions using selected threshold
    binary_predictions = (
        attack_scores >= best_threshold
    ).astype(np.int32)

    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]

    tn, fp, fn, tp = confusion_matrix(
        y_binary,
        binary_predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
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

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    result = {
        "model": model_name,
        "validation_records": int(len(y_validation)),
        "validation_attack_records": int(y_binary.sum()),
        "validation_benign_records": int((y_binary == 0).sum()),

        "multiclass_accuracy": float(multiclass_accuracy),
        "multiclass_macro_f1": float(multiclass_macro_f1),

        "binary_roc_auc": float(binary_roc_auc),
        "binary_average_precision": float(binary_average_precision),

        "selected_threshold": best_threshold,

        "threshold_selection_metric": "Maximum validation attack F1",

        "binary_accuracy": float(accuracy),
        "attack_precision": float(attack_precision),
        "attack_recall": float(attack_recall),
        "attack_f1": float(attack_f1),

        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        "inference_time_seconds": float(inference_time),

        "multiclass_classification_report": multiclass_report
    }

    # Save precision-recall threshold analysis
    threshold_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision_values[:-1],
        "recall": recall_values[:-1],
        "f1_score": f1_values
    })

    safe_name = model_name.lower().replace(" ", "_")

    threshold_df.to_csv(
        REPORT_DIR / f"{safe_name}_validation_threshold_analysis.csv",
        index=False
    )

    print(f"\nSelected threshold: {best_threshold:.6f}")
    print(f"Binary ROC-AUC: {binary_roc_auc:.6f}")
    print(f"Binary Average Precision: {binary_average_precision:.6f}")
    print(f"Attack Precision: {attack_precision:.6f}")
    print(f"Attack Recall: {attack_recall:.6f}")
    print(f"Attack F1: {attack_f1:.6f}")
    print(f"False Positive Rate: {fpr:.6f}")
    print(f"False Negative Rate: {fnr:.6f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn:,} | FP: {fp:,}")
    print(f"FN: {fn:,} | TP: {tp:,}")

    print(f"\nInference time: {inference_time:.2f} seconds")

    return result


# ============================================================
# 6. CALIBRATE BOTH MODELS
# ============================================================

results = []

for model_name, model in models.items():

    result = calibrate_model(
        model_name,
        model
    )

    results.append(result)


# ============================================================
# 7. SAVE CALIBRATION RESULTS
# ============================================================

threshold_config = {}

for result in results:

    key = result["model"].lower().replace(" ", "_")

    threshold_config[key] = {
        "threshold": result["selected_threshold"],
        "selection_metric": result["threshold_selection_metric"]
    }

with open(
    REPORT_DIR / "calibrated_alert_thresholds.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        threshold_config,
        f,
        indent=4
    )


# ============================================================
# 8. SAVE COMPLETE METRICS
# ============================================================

summary = {
    "dataset": "CICIDS2017 validation dataset",
    "purpose": "Alert threshold calibration",
    "threshold_selection": "Maximum validation attack F1",
    "results": results
}

with open(
    REPORT_DIR / "validation_calibration_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 9. SAVE COMPARISON CSV
# ============================================================

comparison = pd.DataFrame([
    {
        "model": r["model"],
        "threshold": r["selected_threshold"],
        "accuracy": r["binary_accuracy"],
        "precision": r["attack_precision"],
        "recall": r["attack_recall"],
        "f1": r["attack_f1"],
        "roc_auc": r["binary_roc_auc"],
        "average_precision": r["binary_average_precision"],
        "fpr": r["false_positive_rate"],
        "fnr": r["false_negative_rate"]
    }
    for r in results
])

comparison.to_csv(
    REPORT_DIR / "validation_calibration_comparison.csv",
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION CALIBRATION COMPLETED")
print("=" * 70)

print(comparison.to_string(index=False))

print("\nSaved files:")
print("1. calibrated_alert_thresholds.json")
print("2. validation_calibration_summary.json")
print("3. validation_calibration_comparison.csv")
print("4. Individual threshold analysis CSV files")

print("\nOriginal test dataset was NOT used.")
print("STEP 19.3 COMPLETED SUCCESSFULLY")

