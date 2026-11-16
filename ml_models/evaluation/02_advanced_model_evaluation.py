
import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml_models" / "saved_models"
TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "cicids2017_model_ready" / "test"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH = MODEL_DIR / "random_forest_baseline.joblib"
XGB_PATH = MODEL_DIR / "xgboost_baseline.joblib"

TEST_GLOB = str(TEST_DIR / "**" / "*.parquet").replace("\\", "/")


# ============================================================
# 2. LOAD MODELS
# ============================================================

print("\n" + "=" * 65)
print("CYBERSHIELD AI - ADVANCED MODEL EVALUATION")
print("=" * 65)

print("\nLoading trained models...")

rf_model = joblib.load(RF_PATH)
xgb_model = joblib.load(XGB_PATH)

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

print("Models loaded successfully.")



# ============================================================
# 3. LOAD FEATURE NAMES FROM PREPROCESSING REPORT
# ============================================================

FEATURE_LIST_PATH = REPORT_DIR / "model_feature_list.json"

with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as file:
    FEATURES = json.load(file)

if len(FEATURES) != 70:
    raise ValueError(
        f"Expected 70 features, found {len(FEATURES)}"
    )

if len(set(FEATURES)) != len(FEATURES):
    raise ValueError("Duplicate feature names found.")

print(f"Loaded {len(FEATURES)} features successfully.")

print("Checking feature compatibility...")

# Check model input dimensions
if rf_model.n_features_in_ != len(FEATURES):
    raise ValueError("Random Forest feature count mismatch.")

if xgb_model.n_features_in_ != len(FEATURES):
    raise ValueError("XGBoost feature count mismatch.")

print("Both models have compatible input dimensions.")


# ============================================================
# 4. IDENTIFY TARGET COLUMN
# ============================================================

con = duckdb.connect()

schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{TEST_GLOB}', union_by_name=true)"
).fetchdf()

all_columns = schema["column_name"].tolist()

extra_columns = [c for c in all_columns if c not in FEATURES]

if not extra_columns:
    raise ValueError("Could not find the target column in test data.")

# Prefer a numeric target column, such as Label_Encoded.
numeric_types = (
    "INT", "INTEGER", "BIGINT", "SMALLINT",
    "TINYINT", "UBIGINT", "UINTEGER", "DOUBLE",
    "FLOAT", "REAL", "DECIMAL"
)

numeric_candidates = []

for _, row in schema.iterrows():
    if row["column_name"] in extra_columns:
        dtype = str(row["column_type"]).upper()
        if any(dtype.startswith(t) for t in numeric_types):
            numeric_candidates.append(row["column_name"])

if numeric_candidates:
    preferred = [
        c for c in numeric_candidates
        if any(word in c.lower() for word in
               ["encoded", "label", "target", "class"])
    ]
    TARGET_COLUMN = preferred[0] if preferred else numeric_candidates[0]
else:
    preferred = [
        c for c in extra_columns
        if any(word in c.lower() for word in
               ["label", "target", "class"])
    ]
    TARGET_COLUMN = preferred[0] if preferred else extra_columns[0]

print(f"Detected target column: {TARGET_COLUMN}")


# ============================================================
# 5. VERIFY MODEL CLASSES
# ============================================================

classes = np.asarray(rf_model.classes_)

if not np.array_equal(classes, np.asarray(xgb_model.classes_)):
    raise ValueError("The two models have different class IDs.")

class_ids = classes.tolist()
number_of_classes = len(class_ids)

print(f"Number of classes: {number_of_classes}")


# ============================================================
# 6. PREDICT TEST DATA IN CHUNKS
# ============================================================

quoted_features = ", ".join(
    '"' + feature.replace('"', '""') + '"'
    for feature in FEATURES
)

target_sql = '"' + TARGET_COLUMN.replace('"', '""') + '"'

query = f"""
SELECT {quoted_features}, {target_sql}
FROM read_parquet('{TEST_GLOB}', union_by_name=true)
"""

con.execute(query)

y_true_parts = []
probability_parts = {
    name: [] for name in models
}

inference_times = {
    name: 0.0 for name in models
}

total_records = 0

print("\nStarting chunk-based evaluation...")

while True:

    chunk = con.fetch_df_chunk(vectors_per_chunk=100)

    if chunk.empty:
        break

    X_chunk = chunk[FEATURES]
    y_chunk = chunk[TARGET_COLUMN].to_numpy()

    # If labels are strings, encode them using sorted order.
    if not np.issubdtype(y_chunk.dtype, np.number):
        unique_labels = sorted(pd.Series(y_chunk).astype(str).unique())
        label_to_id = {
            label: index for index, label in enumerate(unique_labels)
        }
        y_chunk = np.array([
            label_to_id[str(label)] for label in y_chunk
        ])
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is not numeric. "
            "The script needs the encoded target column, such as Label_Encoded."
        )

    y_true_parts.append(y_chunk)

    for name, model in models.items():

        start_time = time.perf_counter()

        probabilities = model.predict_proba(X_chunk)

        inference_times[name] += time.perf_counter() - start_time

        probability_parts[name].append(probabilities)

    total_records += len(chunk)

    if total_records % 100000 < len(chunk):
        print(f"Processed {total_records:,} records...")


con.close()

y_true = np.concatenate(y_true_parts)

probabilities_by_model = {
    name: np.vstack(parts)
    for name, parts in probability_parts.items()
}

print(f"\nTotal test records evaluated: {total_records:,}")


# ============================================================
# 7. CALCULATE ADVANCED METRICS
# ============================================================

y_binary = np.column_stack([
    (y_true == class_id).astype(int)
    for class_id in class_ids
])

overall_results = []
class_results = []
curve_data = {}

for name, probabilities in probabilities_by_model.items():

    print(f"\nEvaluating {name}...")

    predictions = classes[np.argmax(probabilities, axis=1)]

    macro_auc = roc_auc_score(
        y_true,
        probabilities,
        labels=classes,
        multi_class="ovr",
        average="macro"
    )

    weighted_auc = roc_auc_score(
        y_true,
        probabilities,
        labels=classes,
        multi_class="ovr",
        average="weighted"
    )

    macro_ap = average_precision_score(
        y_binary,
        probabilities,
        average="macro"
    )

    weighted_ap = average_precision_score(
        y_binary,
        probabilities,
        average="weighted"
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=classes
    )

    total = cm.sum()

    fp_total = 0
    tn_total = 0
    fn_total = 0
    tp_total = 0

    for i, class_id in enumerate(class_ids):

        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        fpr = fp / (fp + tn) if (fp + tn) else 0
        fnr = fn / (fn + tp) if (fn + tp) else 0

        fp_total += fp
        tn_total += tn
        fn_total += fn
        tp_total += tp

        class_results.append({
            "Model": name,
            "Class_ID": class_id,
            "Support": int(tp + fn),
            "FPR": fpr,
            "FNR": fnr,
            "Precision": tp / (tp + fp) if (tp + fp) else 0,
            "Recall": tp / (tp + fn) if (tp + fn) else 0,
            "Average_Precision": average_precision_score(
                y_binary[:, i], probabilities[:, i]
            ),
            "ROC_AUC": roc_auc_score(
                y_binary[:, i], probabilities[:, i]
            )
        })

    # Overall micro-average false positive and negative rates.
    overall_fpr = fp_total / (fp_total + tn_total)
    overall_fnr = fn_total / (fn_total + tp_total)

    inference_seconds = inference_times[name]
    throughput = (
        total_records / inference_seconds
        if inference_seconds > 0 else 0
    )

    model_path = RF_PATH if name == "Random Forest" else XGB_PATH
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    confidence = np.max(probabilities, axis=1)

    overall_results.append({
        "Model": name,
        "ROC_AUC_Macro": macro_auc,
        "ROC_AUC_Weighted": weighted_auc,
        "Average_Precision_Macro": macro_ap,
        "Average_Precision_Weighted": weighted_ap,
        "Overall_FPR": overall_fpr,
        "Overall_FNR": overall_fnr,
        "Inference_Time_Seconds": inference_seconds,
        "Throughput_Records_Per_Second": throughput,
        "Model_Size_MB": model_size_mb,
        "Mean_Prediction_Confidence": float(np.mean(confidence))
    })

    # Micro-average ROC and PR curves.
    fpr_curve, tpr_curve, _ = roc_curve(
        y_binary.ravel(),
        probabilities.ravel()
    )

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_binary.ravel(),
        probabilities.ravel()
    )

    curve_data[name] = {
        "fpr": fpr_curve,
        "tpr": tpr_curve,
        "precision": precision_curve,
        "recall": recall_curve
    }


# ============================================================
# 8. SAVE REPORTS
# ============================================================

overall_df = pd.DataFrame(overall_results)
class_df = pd.DataFrame(class_results)

overall_df.to_csv(
    REPORT_DIR / "advanced_model_metrics.csv",
    index=False
)

class_df.to_csv(
    REPORT_DIR / "advanced_per_class_metrics.csv",
    index=False
)

summary = {
    "dataset": "CICIDS2017",
    "test_records": int(total_records),
    "number_of_features": len(FEATURES),
    "number_of_classes": number_of_classes,
    "models": overall_results
}

with open(
    REPORT_DIR / "advanced_evaluation_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)


# ============================================================
# 9. PLOT MICRO-AVERAGE ROC CURVE
# ============================================================

plt.figure(figsize=(9, 7))

for name, data in curve_data.items():
    plt.plot(
        data["fpr"],
        data["tpr"],
        label=name
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Micro-Average ROC Curve - CyberShield AI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "advanced_roc_curve.png",
    dpi=300
)
plt.close()


# ============================================================
# 10. PLOT MICRO-AVERAGE PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(9, 7))

for name, data in curve_data.items():
    plt.plot(
        data["recall"],
        data["precision"],
        label=name
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Micro-Average Precision-Recall Curve - CyberShield AI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "advanced_precision_recall_curve.png",
    dpi=300
)
plt.close()


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("ADVANCED MODEL EVALUATION RESULTS")
print("=" * 65)

print(
    overall_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nGenerated reports:")
print(REPORT_DIR / "advanced_model_metrics.csv")
print(REPORT_DIR / "advanced_per_class_metrics.csv")
print(REPORT_DIR / "advanced_evaluation_summary.json")

print("\nGenerated figures:")
print(FIGURE_DIR / "advanced_roc_curve.png")
print(FIGURE_DIR / "advanced_precision_recall_curve.png")

print("\nSTEP 17 COMPLETED SUCCESSFULLY")

