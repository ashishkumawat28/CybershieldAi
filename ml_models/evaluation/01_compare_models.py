
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_metric(data, key):
    """
    Read metrics regardless of whether JSON keys
    use uppercase or lowercase naming.
    """
    key_map = {
        "accuracy": ["accuracy", "Accuracy"],
        "macro_f1": ["macro_f1", "Macro_F1"],
        "weighted_f1": ["weighted_f1", "Weighted_F1"],
    }

    for possible_key in key_map[key]:
        if possible_key in data:
            return data[possible_key]

    raise KeyError(f"Metric '{key}' not found. Available keys: {list(data.keys())}")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RF_METRICS_PATH = REPORT_DIR / "random_forest_metrics.json"
XGB_METRICS_PATH = REPORT_DIR / "xgboost_metrics.json"

RF_REPORT_PATH = REPORT_DIR / "random_forest_classification_report.csv"
XGB_REPORT_PATH = REPORT_DIR / "xgboost_classification_report.csv"

RF_CM_PATH = REPORT_DIR / "random_forest_confusion_matrix.csv"
XGB_CM_PATH = REPORT_DIR / "xgboost_confusion_matrix.csv"


# ============================================================
# 2. VERIFY REPORT FILES
# ============================================================

required_files = [
    RF_METRICS_PATH,
    XGB_METRICS_PATH,
    RF_REPORT_PATH,
    XGB_REPORT_PATH,
    RF_CM_PATH,
    XGB_CM_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required report not found: {file_path}"
        )

print("\n" + "=" * 70)
print("CYBERSHIELD AI - MODEL COMPARISON")
print("=" * 70)


# ============================================================
# 3. LOAD OVERALL METRICS
# ============================================================

with open(RF_METRICS_PATH, "r", encoding="utf-8") as file:
    rf_metrics = json.load(file)

with open(XGB_METRICS_PATH, "r", encoding="utf-8") as file:
    xgb_metrics = json.load(file)


# Extract metrics from both models
rf_accuracy = get_metric(rf_metrics, "accuracy")
rf_macro_f1 = get_metric(rf_metrics, "macro_f1")
rf_weighted_f1 = get_metric(rf_metrics, "weighted_f1")

xgb_accuracy = get_metric(xgb_metrics, "accuracy")
xgb_macro_f1 = get_metric(xgb_metrics, "macro_f1")
xgb_weighted_f1 = get_metric(xgb_metrics, "weighted_f1")


metric_names = [
    "accuracy",
    "macro_f1",
    "weighted_f1"
]

comparison_data = {
    "Metric": ["Accuracy", "Macro F1", "Weighted F1"],
    "Random Forest": [
        rf_accuracy,
        rf_macro_f1,
        rf_weighted_f1
    ],
    "XGBoost": [
        xgb_accuracy,
        xgb_macro_f1,
        xgb_weighted_f1
    ]
}

metrics_df = pd.DataFrame(comparison_data)

metrics_df["Improvement (percentage points)"] = (
    metrics_df["XGBoost"] - metrics_df["Random Forest"]
) * 100

metrics_df.to_csv(
    REPORT_DIR / "model_metrics_comparison.csv",
    index=False
)

print("\nOVERALL MODEL COMPARISON")
print("-" * 70)

print(
    metrics_df.to_string(
        index=False,
        formatters={
            "Random Forest": "{:.6f}".format,
            "XGBoost": "{:.6f}".format,
            "Improvement (percentage points)": "{:+.4f}".format
        }
    )
)


# ============================================================
# 4. LOAD CLASSIFICATION REPORTS
# ============================================================

rf_report = pd.read_csv(
    RF_REPORT_PATH,
    index_col=0
)

xgb_report = pd.read_csv(
    XGB_REPORT_PATH,
    index_col=0
)

# Remove aggregate rows; retain actual attack classes.
aggregate_rows = [
    "accuracy",
    "macro avg",
    "weighted avg",
    "micro avg",
    "samples avg"
]

rf_classes = rf_report.drop(
    index=aggregate_rows,
    errors="ignore"
)

xgb_classes = xgb_report.drop(
    index=aggregate_rows,
    errors="ignore"
)

# Keep only the classification metrics.
metric_columns = [
    "precision",
    "recall",
    "f1-score",
    "support"
]

rf_classes = rf_classes[
    [col for col in metric_columns if col in rf_classes.columns]
]

xgb_classes = xgb_classes[
    [col for col in metric_columns if col in xgb_classes.columns]
]

# Align class names across both models.
common_classes = sorted(
    set(rf_classes.index) & set(xgb_classes.index)
)

if len(common_classes) != 15:
    raise RuntimeError(
        "Expected 15 common attack classes. "
        f"Found {len(common_classes)}: {common_classes}"
    )

rf_classes = rf_classes.loc[common_classes]
xgb_classes = xgb_classes.loc[common_classes]


# ============================================================
# 5. CREATE PER-CLASS COMPARISON
# ============================================================

class_comparison = pd.DataFrame(index=common_classes)

for metric in ["precision", "recall", "f1-score"]:

    class_comparison[f"RF_{metric}"] = (
        rf_classes[metric]
    )

    class_comparison[f"XGB_{metric}"] = (
        xgb_classes[metric]
    )

    class_comparison[f"Improvement_{metric}"] = (
        xgb_classes[metric] - rf_classes[metric]
    )

class_comparison["Support"] = (
    xgb_classes["support"].astype(int)
)

class_comparison.index.name = "Attack_Type"

class_comparison.to_csv(
    REPORT_DIR / "per_class_model_comparison.csv"
)

print("\nPER-CLASS F1 COMPARISON")
print("-" * 70)

print(
    class_comparison[
        [
            "RF_f1-score",
            "XGB_f1-score",
            "Improvement_f1-score",
            "Support"
        ]
    ].to_string(
        formatters={
            "RF_f1-score": "{:.4f}".format,
            "XGB_f1-score": "{:.4f}".format,
            "Improvement_f1-score": "{:+.4f}".format
        }
    )
)


# ============================================================
# 6. MODEL METRICS BAR CHART
# ============================================================

x = np.arange(len(metrics_df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    metrics_df["Random Forest"],
    width,
    label="Random Forest"
)

ax.bar(
    x + width / 2,
    metrics_df["XGBoost"],
    width,
    label="XGBoost"
)

ax.set_title("Random Forest vs XGBoost - Overall Metrics")
ax.set_ylabel("Score")
ax.set_xlabel("Evaluation Metric")
ax.set_xticks(x)
ax.set_xticklabels(metrics_df["Metric"])
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "overall_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 7. PER-CLASS F1 COMPARISON
# ============================================================

x = np.arange(len(common_classes))

fig, ax = plt.subplots(figsize=(16, 7))

ax.bar(
    x - width / 2,
    class_comparison["RF_f1-score"],
    width,
    label="Random Forest"
)

ax.bar(
    x + width / 2,
    class_comparison["XGB_f1-score"],
    width,
    label="XGBoost"
)

ax.set_title("Attack-wise F1 Score Comparison")
ax.set_xlabel("Attack Class")
ax.set_ylabel("F1 Score")
ax.set_xticks(x)
ax.set_xticklabels(
    common_classes,
    rotation=55,
    ha="right"
)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "per_class_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 8. LOAD AND VALIDATE CONFUSION MATRICES
# ============================================================

rf_cm = pd.read_csv(
    RF_CM_PATH,
    index_col=0
)

xgb_cm = pd.read_csv(
    XGB_CM_PATH,
    index_col=0
)

rf_cm = rf_cm.loc[common_classes, common_classes]
xgb_cm = xgb_cm.loc[common_classes, common_classes]


# ============================================================
# 9. PLOT CONFUSION MATRICES
# ============================================================

def plot_confusion_matrix(matrix, title, output_path):

    fig, ax = plt.subplots(figsize=(14, 12))

    image = ax.imshow(
        matrix.to_numpy(),
        interpolation="nearest",
        aspect="auto"
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))

    ax.set_xticklabels(
        matrix.columns,
        rotation=55,
        ha="right"
    )

    ax.set_yticklabels(matrix.index)

    fig.colorbar(image, ax=ax, label="Number of Records")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


plot_confusion_matrix(
    rf_cm,
    "Random Forest - Confusion Matrix",
    FIGURE_DIR / "random_forest_confusion_matrix.png"
)

plot_confusion_matrix(
    xgb_cm,
    "XGBoost - Confusion Matrix",
    FIGURE_DIR / "xgboost_confusion_matrix.png"
)


# ============================================================
# 10. IDENTIFY THE CURRENTLY STRONGER MODEL
# ============================================================

# rf_macro = rf_metrics["macro_f1"]
# xgb_macro = xgb_metrics["macro_f1"]

# rf_accuracy = rf_metrics["accuracy"]
# xgb_accuracy = xgb_metrics["accuracy"]

# Extract metrics from both models
rf_accuracy = get_metric(rf_metrics, "accuracy")
rf_macro_f1 = get_metric(rf_metrics, "macro_f1")
rf_weighted_f1 = get_metric(rf_metrics, "weighted_f1")

xgb_accuracy = get_metric(xgb_metrics, "accuracy")
xgb_macro_f1 = get_metric(xgb_metrics, "macro_f1")
xgb_weighted_f1 = get_metric(xgb_metrics, "weighted_f1")

if xgb_macro_f1 > rf_macro_f1:
    selected_model = "XGBoost"
elif rf_macro_f1 > xgb_macro_f1:
    selected_model = "Random Forest"
else:
    selected_model = "Tie"

summary = {
    "selected_model_by_macro_f1": selected_model,
    "random_forest_accuracy": rf_accuracy,
    "xgboost_accuracy": xgb_accuracy,
    "random_forest_macro_f1": rf_macro_f1,
    "xgboost_macro_f1": xgb_macro_f1,
    "random_forest_weighted_f1": rf_weighted_f1,
    "xgboost_weighted_f1": xgb_weighted_f1,
    "test_records": xgb_metrics.get("test_records"),
    "number_of_classes": len(common_classes),
    "selection_basis": "Higher macro F1 on the same CICIDS2017 test split"
}

with open(
    REPORT_DIR / "model_comparison_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

print("Selected model by macro F1:", selected_model)

print("\nGenerated reports:")
print(REPORT_DIR / "model_metrics_comparison.csv")
print(REPORT_DIR / "per_class_model_comparison.csv")
print(REPORT_DIR / "model_comparison_summary.json")

print("\nGenerated figures:")
print(FIGURE_DIR / "overall_model_comparison.png")
print(FIGURE_DIR / "per_class_f1_comparison.png")
print(FIGURE_DIR / "random_forest_confusion_matrix.png")
print(FIGURE_DIR / "xgboost_confusion_matrix.png")

print("\nCyberShield AI - Step 16 completed successfully.")
