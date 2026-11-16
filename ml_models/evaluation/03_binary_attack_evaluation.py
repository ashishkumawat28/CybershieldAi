
import json
import time
import numpy as np
import pandas as pd
import duckdb
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve
)

# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "ml_models" / "saved_models"
TEST_DIR = ROOT / "datasets" / "processed" / "cicids2017_model_ready" / "test"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RF_PATH = MODEL_DIR / "random_forest_baseline.joblib"
XGB_PATH = MODEL_DIR / "xgboost_baseline.joblib"

METRICS_PATH = REPORT_DIR / "xgboost_metrics.json"
FEATURE_PATH = REPORT_DIR / "model_feature_list.json"

TEST_GLOB = str(TEST_DIR / "**" / "*.parquet").replace("\\", "/")

# Thresholds to investigate
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

# ============================================================
# 2. LOAD MODELS AND METADATA
# ============================================================

print("\n" + "=" * 65)
print("CYBERSHIELD AI - BINARY ATTACK EVALUATION")
print("=" * 65)

rf_model = joblib.load(RF_PATH)
xgb_model = joblib.load(XGB_PATH)

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    xgb_metadata = json.load(f)

# Obtain BENIGN encoded ID from the saved class mapping.
class_names = xgb_metadata["class_names"]
class_ids = xgb_metadata["class_ids"]

if len(class_names) != len(class_ids):
    raise ValueError("Class names and class IDs do not match.")

if "BENIGN" not in class_names:
    raise ValueError("BENIGN class not found in model metadata.")

BENIGN_ID = class_ids[class_names.index("BENIGN")]

for name, model in models.items():
    if model.n_features_in_ != len(FEATURES):
        raise ValueError(f"{name}: feature count mismatch.")

    if not np.array_equal(model.classes_, class_ids):
        raise ValueError(f"{name}: class IDs do not match metadata.")

print(f"Features loaded: {len(FEATURES)}")
print(f"BENIGN encoded ID: {BENIGN_ID}")
print(f"Attack classes: {len(class_ids) - 1}")

# ============================================================
# 3. IDENTIFY TARGET COLUMN
# ============================================================

con = duckdb.connect()

schema = con.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{TEST_GLOB}', union_by_name=true)"
).fetchdf()

columns = schema["column_name"].tolist()

if "Label_Encoded" not in columns:
    raise ValueError(
        "Label_Encoded column not found in the test dataset."
    )

TARGET = "Label_Encoded"

# ============================================================
# 4. GENERATE BINARY ATTACK SCORES
# ============================================================

feature_sql = ", ".join(
    '"' + feature.replace('"', '""') + '"'
    for feature in FEATURES
)

query = f"""
SELECT {feature_sql}, "{TARGET}"
FROM read_parquet('{TEST_GLOB}', union_by_name=true)
"""

con.execute(query)

y_binary_parts = []
score_parts = {name: [] for name in models}

inference_times = {name: 0.0 for name in models}

total_records = 0

print("\nGenerating binary attack probabilities...")

while True:

    chunk = con.fetch_df_chunk(vectors_per_chunk=100)

    if chunk.empty:
        break

    X = chunk[FEATURES]
    y = chunk[TARGET].to_numpy()

    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("Encoded target must be numeric.")

    if not np.isin(y, class_ids).all():
        raise ValueError("Unknown class ID found in test data.")

    # 0 = BENIGN, 1 = ATTACK
    y_binary = (y != BENIGN_ID).astype(int)
    y_binary_parts.append(y_binary)

    for name, model in models.items():

        start = time.perf_counter()

        probabilities = model.predict_proba(X)

        inference_times[name] += time.perf_counter() - start

        # Find the BENIGN probability column.
        benign_index = list(model.classes_).index(BENIGN_ID)

        # Sum probability of all attack classes.
        attack_score = 1.0 - probabilities[:, benign_index]

        score_parts[name].append(attack_score)

    total_records += len(chunk)

    if total_records % 100000 < len(chunk):
        print(f"Processed {total_records:,} records...")

con.close()

y_true = np.concatenate(y_binary_parts)

scores = {
    name: np.concatenate(parts)
    for name, parts in score_parts.items()
}

print(f"\nTotal records: {total_records:,}")
print(f"Actual BENIGN records: {(y_true == 0).sum():,}")
print(f"Actual ATTACK records: {(y_true == 1).sum():,}")

# ============================================================
# 5. METRIC CALCULATION FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Attack_Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Attack_Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "Attack_F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "FPR": fpr,
        "FNR": fnr,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

# ============================================================
# 6. EVALUATE DEFAULT AND OPTIMAL THRESHOLDS
# ============================================================

overall_results = []
threshold_results = []
roc_data = {}

for name, attack_scores in scores.items():

    print(f"\nEvaluating {name}...")

    roc_auc = roc_auc_score(y_true, attack_scores)

    avg_precision = average_precision_score(
        y_true, attack_scores
    )

    # Default threshold
    default_predictions = (attack_scores >= 0.50).astype(int)

    default_metrics = calculate_metrics(
        y_true, default_predictions
    )

    # Evaluate every threshold
    model_threshold_rows = []

    for threshold in THRESHOLDS:

        predictions = (attack_scores >= threshold).astype(int)

        metrics = calculate_metrics(y_true, predictions)

        row = {
            "Model": name,
            "Threshold": float(threshold),
            **metrics
        }

        threshold_results.append(row)
        model_threshold_rows.append(row)

    threshold_df = pd.DataFrame(model_threshold_rows)

    # Choose the threshold with maximum attack F1.
    # This is exploratory, not a deployment threshold.
    best_index = threshold_df["Attack_F1"].idxmax()
    best_row = threshold_df.loc[best_index]

    # Inference speed
    seconds = inference_times[name]
    throughput = total_records / seconds if seconds > 0 else 0

    overall_results.append({
        "Model": name,
        "Binary_ROC_AUC": roc_auc,
        "Binary_Average_Precision": avg_precision,
        "Default_Threshold": 0.50,
        **default_metrics,
        "Best_F1_Threshold": best_row["Threshold"],
        "Best_Threshold_F1": best_row["Attack_F1"],
        "Inference_Time_Seconds": seconds,
        "Throughput_Records_Per_Second": throughput
    })

    fpr_curve, tpr_curve, _ = roc_curve(
        y_true, attack_scores
    )

    roc_data[name] = {
        "fpr": fpr_curve,
        "tpr": tpr_curve
    }

# ============================================================
# 7. SAVE CSV AND JSON REPORTS
# ============================================================

overall_df = pd.DataFrame(overall_results)
threshold_df = pd.DataFrame(threshold_results)

overall_df.to_csv(
    REPORT_DIR / "binary_attack_metrics.csv",
    index=False
)

threshold_df.to_csv(
    REPORT_DIR / "binary_threshold_analysis.csv",
    index=False
)

summary = {
    "dataset": "CICIDS2017",
    "test_records": int(total_records),
    "benign_class_id": int(BENIGN_ID),
    "binary_classes": {
        "0": "BENIGN",
        "1": "ATTACK"
    },
    "models": overall_results
}

with open(
    REPORT_DIR / "binary_attack_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

# ============================================================
# 8. PLOT BINARY ROC CURVE
# ============================================================

plt.figure(figsize=(9, 7))

for name, data in roc_data.items():
    auc_value = overall_df.loc[
        overall_df["Model"] == name,
        "Binary_ROC_AUC"
    ].iloc[0]

    plt.plot(
        data["fpr"],
        data["tpr"],
        label=f"{name} (AUC={auc_value:.4f})"
    )

plt.plot([0, 1], [0, 1], "--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Binary Attack Detection - ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "binary_attack_roc_curve.png",
    dpi=250
)
plt.close()

# ============================================================
# 9. PLOT THRESHOLD VS F1
# ============================================================

plt.figure(figsize=(9, 7))

for name in models:
    subset = threshold_df[
        threshold_df["Model"] == name
    ]

    plt.plot(
        subset["Threshold"],
        subset["Attack_F1"],
        marker="o",
        label=name
    )

plt.xlabel("Attack Probability Threshold")
plt.ylabel("Attack F1 Score")
plt.title("Attack F1 vs Alert Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "binary_threshold_f1.png",
    dpi=250
)
plt.close()

# ============================================================
# 10. FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("BINARY ATTACK DETECTION RESULTS")
print("=" * 65)

print(
    overall_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nTHRESHOLD ANALYSIS")
print("-" * 65)

for name in models:

    subset = threshold_df[
        threshold_df["Model"] == name
    ]

    best = subset.loc[subset["Attack_F1"].idxmax()]

    print(f"\n{name}")
    print(f"Best exploratory threshold: {best['Threshold']:.2f}")
    print(f"Attack F1: {best['Attack_F1']:.6f}")
    print(f"FPR: {best['FPR']:.6f}")
    print(f"FNR: {best['FNR']:.6f}")

print("\nGenerated reports:")
print(REPORT_DIR / "binary_attack_metrics.csv")
print(REPORT_DIR / "binary_threshold_analysis.csv")
print(REPORT_DIR / "binary_attack_summary.json")

print("\nGenerated figures:")
print(FIGURE_DIR / "binary_attack_roc_curve.png")
print(FIGURE_DIR / "binary_threshold_f1.png")

print("\nSTEP 18 COMPLETED SUCCESSFULLY")
print("=" * 65)
