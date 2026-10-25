"""
10_correct_crop_bias.py

CROP RECOMMENDATION BIAS CORRECTION

This script:
1. Loads the existing training/test data.
2. Audits crop-level suitability imbalance.
3. Calculates bounded crop-aware sample weights.
4. Trains a bias-corrected Decision Tree.
5. Compares original vs corrected model.
6. Performs manual 5-fold CV compatible with modern scikit-learn.
7. Calculates corrected feature/permutation importance.
8. Saves the corrected model to the EXACT location expected by
   script 11:

models/corrected/crop_suitability_model_corrected.joblib
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_DATA_PATH = (
    DATA_DIR / "north_central_training_dataset.csv"
)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

# Your actual original model path
ORIGINAL_MODEL_PATH = (
    MODELS_DIR / "final_crop_suitability_model.joblib"
)

# ============================================================
# CRITICAL: SCRIPT 11 MUST FIND THE MODEL HERE
# ============================================================

CORRECTED_MODEL_DIR = (
    MODELS_DIR / "corrected"
)

CORRECTED_MODEL_PATH = (
    CORRECTED_MODEL_DIR
    / "crop_suitability_model_corrected.joblib"
)

# Output directory
OUTPUT_DIR = (
    DATA_DIR
    / "model_results"
    / "bias_correction"
)

TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

for directory in [
    CORRECTED_MODEL_DIR,
    TABLE_DIR,
    PLOT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(number, title):
    print()
    print("=" * 70)
    print(f"[{number}] {title}")
    print("=" * 70)


def check_file(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"Found {name}:")
    print(path)


def evaluate_model(model, X, y, model_name):

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = (
            model.predict_proba(X)[:, 1]
        )
    else:
        probabilities = predictions

    return {
        "model": model_name,
        "accuracy": accuracy_score(
            y,
            predictions
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities
        ),
    }


def print_metrics(metrics):

    print(
        f"\n{metrics['model']}"
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CORRECTING CROP RECOMMENDATION BIAS")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print_section(
    1,
    "CHECKING INPUT FILES"
)

check_file(
    TRAINING_DATA_PATH,
    "Training dataset"
)

check_file(
    X_TRAIN_PATH,
    "X_train"
)

check_file(
    X_TEST_PATH,
    "X_test"
)

check_file(
    Y_TRAIN_PATH,
    "y_train"
)

check_file(
    Y_TEST_PATH,
    "y_test"
)

check_file(
    ORIGINAL_MODEL_PATH,
    "Original final model"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print_section(
    2,
    "LOADING DATASETS AND MODEL"
)

training_df = pd.read_csv(
    TRAINING_DATA_PATH
)

X_train = pd.read_csv(
    X_TRAIN_PATH
)

X_test = pd.read_csv(
    X_TEST_PATH
)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze("columns")

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze("columns")

original_model = joblib.load(
    ORIGINAL_MODEL_PATH
)

print(
    f"Training dataset shape: "
    f"{training_df.shape}"
)

print(
    f"X_train shape: "
    f"{X_train.shape}"
)

print(
    f"X_test shape:  "
    f"{X_test.shape}"
)

print(
    f"y_train shape: "
    f"{y_train.shape}"
)

print(
    f"y_test shape:  "
    f"{y_test.shape}"
)

print(
    f"Original model: "
    f"{type(original_model).__name__}"
)


# ============================================================
# 3. VALIDATE DATA
# ============================================================

print_section(
    3,
    "VALIDATING DATASET STRUCTURE"
)

required_columns = [
    "crop_encoded"
]

for column in required_columns:

    if column not in X_train.columns:
        raise ValueError(
            f"{column} missing from X_train."
        )

    if column not in X_test.columns:
        raise ValueError(
            f"{column} missing from X_test."
        )

if len(X_train) != len(y_train):
    raise ValueError(
        "X_train and y_train lengths do not match."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test lengths do not match."
    )

if "suitable" not in training_df.columns:
    raise ValueError(
        "suitable column missing from training dataset."
    )

print(
    "SUCCESS: Dataset structure is valid."
)


# ============================================================
# 4. AUDIT CROP-LEVEL SUITABILITY
# ============================================================

print_section(
    4,
    "AUDITING CROP-LEVEL SUITABILITY"
)

crop_audit = (
    training_df
    .groupby("crop")
    .agg(
        total_records=(
            "suitable",
            "size"
        ),
        suitable_records=(
            "suitable",
            "sum"
        ),
        mean_suitability=(
            "suitable",
            "mean"
        )
    )
    .reset_index()
)

crop_audit[
    "suitability_percentage"
] = (
    crop_audit[
        "mean_suitability"
    ] * 100
)

crop_audit = crop_audit.sort_values(
    "mean_suitability",
    ascending=False
)

print(
    crop_audit.to_string(
        index=False
    )
)

crop_audit.to_csv(
    TABLE_DIR
    / "crop_suitability_before_correction.csv",
    index=False
)

zero_crops = crop_audit.loc[
    crop_audit["suitable_records"] == 0,
    "crop"
].tolist()

low_crops = crop_audit.loc[
    crop_audit[
        "suitability_percentage"
    ] < 10,
    "crop"
].tolist()

high_crops = crop_audit.loc[
    crop_audit[
        "suitability_percentage"
    ] > 80,
    "crop"
].tolist()

print(
    f"\nCrops with zero suitable records: "
    f"{len(zero_crops)}"
)

print(zero_crops)

print(
    f"\nCrops below 10% suitability: "
    f"{len(low_crops)}"
)

print(low_crops)

print(
    f"\nCrops above 80% suitability: "
    f"{len(high_crops)}"
)

print(high_crops)


# ============================================================
# 5. BEFORE-CORRECTION PLOT
# ============================================================

print_section(
    5,
    "GENERATING CROP SUITABILITY PLOT"
)

plt.figure(
    figsize=(12, 7)
)

plt.bar(
    crop_audit["crop"],
    crop_audit[
        "suitability_percentage"
    ]
)

plt.axhline(
    50,
    linestyle="--"
)

plt.xlabel(
    "Crop"
)

plt.ylabel(
    "Suitability Percentage (%)"
)

plt.title(
    "Crop-Level Suitability Before Bias Correction"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

before_plot = (
    PLOT_DIR
    / "crop_suitability_before_correction.png"
)

plt.savefig(
    before_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved plot:\n{before_plot}"
)


# ============================================================
# 6. PREPARE CROP INFORMATION
# ============================================================

print_section(
    6,
    "PREPARING CROP-AWARE TRAINING INFORMATION"
)

if "crop_encoded" not in X_train.columns:
    raise ValueError(
        "crop_encoded is required."
    )

crop_codes = (
    X_train["crop_encoded"]
    .to_numpy()
)

targets = (
    y_train
    .to_numpy()
)

if len(crop_codes) != len(targets):
    raise ValueError(
        "crop_encoded and y_train cannot be aligned."
    )

crop_training_df = pd.DataFrame(
    {
        "crop_encoded": crop_codes,
        "suitable": targets
    }
)

crop_training_stats = (
    crop_training_df
    .groupby("crop_encoded")
    .agg(
        training_records=(
            "suitable",
            "size"
        ),
        suitable_records=(
            "suitable",
            "sum"
        ),
        training_suitability_rate=(
            "suitable",
            "mean"
        )
    )
    .reset_index()
)

print(
    crop_training_stats.to_string(
        index=False
    )
)


# ============================================================
# 7. CALCULATE BOUNDED CROP WEIGHTS
# ============================================================

print_section(
    7,
    "CALCULATING BOUNDED CROP-AWARE WEIGHTS"
)

"""
Important:

We deliberately avoid:

    weight = 1 / suitability_rate

because:

- Sorghum has zero positive records.
- Yam has extremely few positive records.
- Millet has extremely few positive records.

Those formulas could generate extremely large or infinite
weights and make the model unstable.

Instead, we calculate a bounded weight from the distance
between each crop's suitability rate and the overall 50%
reference.

The correction is intentionally moderate.
"""

overall_rate = np.mean(targets)

print(
    f"Overall training suitability rate: "
    f"{overall_rate:.4f}"
)

crop_training_stats[
    "distance_from_overall"
] = (
    crop_training_stats[
        "training_suitability_rate"
    ]
    - overall_rate
).abs()

max_distance = max(
    crop_training_stats[
        "distance_from_overall"
    ].max(),
    1e-9
)

crop_training_stats[
    "normalized_distance"
] = (
    crop_training_stats[
        "distance_from_overall"
    ]
    / max_distance
)

# Moderate bounded adjustment:
#
# closest to overall rate -> approximately 0.90
# most extreme crop       -> approximately 1.10

crop_training_stats[
    "crop_weight"
] = (
    0.90
    + 0.20
    * crop_training_stats[
        "normalized_distance"
    ]
)

# Normalize around 1.0
crop_training_stats[
    "crop_weight"
] = (
    crop_training_stats[
        "crop_weight"
    ]
    / crop_training_stats[
        "crop_weight"
    ].mean()
)

print(
    "\nCalculated crop-aware weights:"
)

print(
    crop_training_stats[
        [
            "crop_encoded",
            "training_suitability_rate",
            "training_records",
            "crop_weight"
        ]
    ].to_string(
        index=False
    )
)

crop_training_stats.to_csv(
    TABLE_DIR
    / "crop_aware_training_weights.csv",
    index=False
)


# ============================================================
# 8. GENERATE SAMPLE WEIGHTS
# ============================================================

print_section(
    8,
    "GENERATING SAMPLE WEIGHTS"
)

# ------------------------------------------------------------
# IMPORTANT FIX:
#
# .to_numpy(copy=True) ensures that the array is writable.
# This prevents:
#
# ValueError:
# assignment destination is read-only
# ------------------------------------------------------------

sample_weights = np.ones(
    len(y_train),
    dtype=float
).copy()

# Class weights
class_counts = (
    pd.Series(targets)
    .value_counts()
    .to_dict()
)

n_samples = len(targets)
n_classes = len(class_counts)

class_weights = {}

for cls, count in class_counts.items():

    class_weights[cls] = (
        n_samples
        / (n_classes * count)
    )

print(
    "\nClass weights:"
)

print(
    class_weights
)

crop_weight_map = dict(
    zip(
        crop_training_stats[
            "crop_encoded"
        ],
        crop_training_stats[
            "crop_weight"
        ]
    )
)

for i in range(
    len(sample_weights)
):

    crop_code = crop_codes[i]
    target = targets[i]

    crop_weight = crop_weight_map.get(
        crop_code,
        1.0
    )

    class_weight = class_weights.get(
        target,
        1.0
    )

    sample_weights[i] = (
        crop_weight
        * class_weight
    )

# Normalize weights
sample_weights = (
    sample_weights
    / np.mean(sample_weights)
)

# Guarantee writable array
sample_weights = np.array(
    sample_weights,
    dtype=float,
    copy=True
)

print(
    "\nSample weight statistics:"
)

print(
    pd.Series(
        sample_weights
    ).describe()
)

np.save(
    TABLE_DIR
    / "sample_weights.npy",
    sample_weights
)


# ============================================================
# 9. TRAIN CORRECTED MODEL
# ============================================================

print_section(
    9,
    "TRAINING BIAS-CORRECTED DECISION TREE"
)

# Start from the original model's parameters where possible.

original_params = (
    original_model.get_params()
)

corrected_params = {}

allowed_params = [
    "criterion",
    "splitter",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "random_state",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "class_weight",
    "ccp_alpha",
    "monotonic_cst",
]

for parameter in allowed_params:

    if parameter in original_params:

        corrected_params[
            parameter
        ] = original_params[
            parameter
        ]

# Explicit random state
corrected_params[
    "random_state"
] = 42

corrected_model = (
    DecisionTreeClassifier(
        **corrected_params
    )
)

corrected_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print(
    "SUCCESS: Bias-corrected Decision Tree trained."
)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

print_section(
    10,
    "COMPARING ORIGINAL AND CORRECTED MODELS"
)

original_metrics = evaluate_model(
    original_model,
    X_test,
    y_test,
    "Original Decision Tree"
)

corrected_metrics = evaluate_model(
    corrected_model,
    X_test,
    y_test,
    "Bias-Corrected Decision Tree"
)

print_metrics(
    original_metrics
)

print_metrics(
    corrected_metrics
)

comparison = pd.DataFrame(
    [
        original_metrics,
        corrected_metrics
    ]
)

print(
    "\nModel comparison:"
)

print(
    comparison.to_string(
        index=False
    )
)

comparison.to_csv(
    TABLE_DIR
    / "model_comparison.csv",
    index=False
)


# Confusion matrix
corrected_predictions = (
    corrected_model.predict(X_test)
)

corrected_probabilities = (
    corrected_model.predict_proba(
        X_test
    )[:, 1]
)

cm = confusion_matrix(
    y_test,
    corrected_predictions
)

print(
    "\nCorrected model confusion matrix:"
)

print(cm)

pd.DataFrame(
    cm,
    index=[
        "Actual_Not_Suitable",
        "Actual_Suitable"
    ],
    columns=[
        "Predicted_Not_Suitable",
        "Predicted_Suitable"
    ]
).to_csv(
    TABLE_DIR
    / "corrected_confusion_matrix.csv"
)

print(
    "\nCorrected classification report:"
)

classification_text = (
    classification_report(
        y_test,
        corrected_predictions,
        target_names=[
            "Not Suitable",
            "Suitable"
        ],
        zero_division=0
    )
)

print(
    classification_text
)

with open(
    TABLE_DIR
    / "corrected_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        classification_text
    )


# ============================================================
# 11. MANUAL 5-FOLD CROSS VALIDATION
# ============================================================

print_section(
    11,
    "5-FOLD CROSS-VALIDATION"
)

"""
We intentionally do NOT use:

cross_validate(..., fit_params=...)

because your installed scikit-learn version rejects
fit_params.

Instead, each fold is fitted manually.
"""

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_rows = []

for fold_number, (
    train_indices,
    validation_indices
) in enumerate(
    skf.split(
        X_train,
        y_train
    ),
    start=1
):

    X_fold_train = (
        X_train.iloc[
            train_indices
        ]
    )

    X_fold_validation = (
        X_train.iloc[
            validation_indices
        ]
    )

    y_fold_train = (
        y_train.iloc[
            train_indices
        ]
    )

    y_fold_validation = (
        y_train.iloc[
            validation_indices
        ]
    )

    fold_weights = (
        sample_weights[
            train_indices
        ]
        .copy()
    )

    fold_model = clone(
        corrected_model
    )

    fold_model.fit(
        X_fold_train,
        y_fold_train,
        sample_weight=fold_weights
    )

    fold_predictions = (
        fold_model.predict(
            X_fold_validation
        )
    )

    fold_probabilities = (
        fold_model.predict_proba(
            X_fold_validation
        )[:, 1]
    )

    cv_rows.append(
        {
            "fold": fold_number,
            "accuracy": accuracy_score(
                y_fold_validation,
                fold_predictions
            ),
            "precision": precision_score(
                y_fold_validation,
                fold_predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_fold_validation,
                fold_predictions,
                zero_division=0
            ),
            "f1_score": f1_score(
                y_fold_validation,
                fold_predictions,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_fold_validation,
                fold_probabilities
            )
        }
    )

cv_results = pd.DataFrame(
    cv_rows
)

print(
    "\nCross-validation results:"
)

print(
    cv_results.to_string(
        index=False
    )
)

cv_summary = pd.DataFrame(
    {
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ],
        "mean": [
            cv_results[
                "accuracy"
            ].mean(),

            cv_results[
                "precision"
            ].mean(),

            cv_results[
                "recall"
            ].mean(),

            cv_results[
                "f1_score"
            ].mean(),

            cv_results[
                "roc_auc"
            ].mean()
        ],
        "std": [
            cv_results[
                "accuracy"
            ].std(),

            cv_results[
                "precision"
            ].std(),

            cv_results[
                "recall"
            ].std(),

            cv_results[
                "f1_score"
            ].std(),

            cv_results[
                "roc_auc"
            ].std()
        ]
    }
)

print(
    "\nCross-validation summary:"
)

print(
    cv_summary.to_string(
        index=False
    )
)

cv_results.to_csv(
    TABLE_DIR
    / "corrected_model_cv_results.csv",
    index=False
)

cv_summary.to_csv(
    TABLE_DIR
    / "corrected_model_cv_summary.csv",
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("CORRECTED MODEL FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(
    corrected_model,
    "feature_importances_"
):

    feature_importance = pd.DataFrame(
        {
            "feature":
                X_train.columns,

            "importance":
                corrected_model
                .feature_importances_
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    feature_importance.to_csv(
        TABLE_DIR
        / "corrected_model_feature_importance.csv",
        index=False
    )

    crop_importance = float(
        feature_importance.loc[
            feature_importance[
                "feature"
            ] == "crop_encoded",
            "importance"
        ].iloc[0]
    )

    environmental_importance = (
        1.0 - crop_importance
    )

    print(
        f"\nCorrected crop_encoded importance: "
        f"{crop_importance:.6f}"
    )

    print(
        f"Corrected environmental importance: "
        f"{environmental_importance:.6f}"
    )

else:

    crop_importance = None
    environmental_importance = None


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("CORRECTED PERMUTATION IMPORTANCE")
print("=" * 70)

permutation = permutation_importance(
    corrected_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame(
    {
        "feature":
            X_test.columns,

        "importance_mean":
            permutation.importances_mean,

        "importance_std":
            permutation.importances_std
    }
).sort_values(
    "importance_mean",
    ascending=False
)

print(
    permutation_df.head(
        15
    ).to_string(
        index=False
    )
)

permutation_df.to_csv(
    TABLE_DIR
    / "corrected_permutation_importance.csv",
    index=False
)


# ============================================================
# SAVE CORRECTED MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING CORRECTED MODEL")
print("=" * 70)

CORRECTED_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# THIS IS THE EXACT FILE SCRIPT 11 SHOULD FIND.

joblib.dump(
    corrected_model,
    CORRECTED_MODEL_PATH
)

if not CORRECTED_MODEL_PATH.exists():

    raise FileNotFoundError(
        "Corrected model was NOT created:\n"
        f"{CORRECTED_MODEL_PATH}"
    )

print(
    "\nSUCCESS: Corrected model saved to:"
)

print(
    CORRECTED_MODEL_PATH
)

# Verify by loading it again.

test_loaded_model = joblib.load(
    CORRECTED_MODEL_PATH
)

print(
    "\nSUCCESS: Corrected model reloaded successfully."
)

print(
    "Model type:",
    type(test_loaded_model).__name__
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "training_records":
        int(len(X_train)),

    "test_records":
        int(len(X_test)),

    "feature_count":
        int(X_train.shape[1]),

    "original_model":
        type(original_model).__name__,

    "corrected_model":
        type(corrected_model).__name__,

    "original_accuracy":
        float(
            original_metrics[
                "accuracy"
            ]
        ),

    "corrected_accuracy":
        float(
            corrected_metrics[
                "accuracy"
            ]
        ),

    "original_precision":
        float(
            original_metrics[
                "precision"
            ]
        ),

    "corrected_precision":
        float(
            corrected_metrics[
                "precision"
            ]
        ),

    "original_recall":
        float(
            original_metrics[
                "recall"
            ]
        ),

    "corrected_recall":
        float(
            corrected_metrics[
                "recall"
            ]
        ),

    "original_f1":
        float(
            original_metrics[
                "f1_score"
            ]
        ),

    "corrected_f1":
        float(
            corrected_metrics[
                "f1_score"
            ]
        ),

    "original_roc_auc":
        float(
            original_metrics[
                "roc_auc"
            ]
        ),

    "corrected_roc_auc":
        float(
            corrected_metrics[
                "roc_auc"
            ]
        ),

    "corrected_cv_f1_mean":
        float(
            cv_summary.loc[
                cv_summary[
                    "metric"
                ] == "f1_score",
                "mean"
            ].iloc[0]
        ),

    "corrected_cv_f1_std":
        float(
            cv_summary.loc[
                cv_summary[
                    "metric"
                ] == "f1_score",
                "std"
            ].iloc[0]
        ),

    "corrected_cv_roc_auc_mean":
        float(
            cv_summary.loc[
                cv_summary[
                    "metric"
                ] == "roc_auc",
                "mean"
            ].iloc[0]
        ),

    "corrected_cv_roc_auc_std":
        float(
            cv_summary.loc[
                cv_summary[
                    "metric"
                ] == "roc_auc",
                "std"
            ].iloc[0]
        ),

    "corrected_model_path":
        str(
            CORRECTED_MODEL_PATH
        )
}

with open(
    OUTPUT_DIR
    / "bias_correction_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINAL REPORT
# ============================================================

report = []

report.append(
    "CROP RECOMMENDATION BIAS CORRECTION REPORT"
)

report.append(
    "=" * 60
)

report.append("")

report.append(
    f"Training records: {len(X_train)}"
)

report.append(
    f"Test records: {len(X_test)}"
)

report.append(
    f"Features: {X_train.shape[1]}"
)

report.append("")

report.append(
    "ORIGINAL MODEL"
)

report.append(
    f"Accuracy : "
    f"{original_metrics['accuracy']:.4f}"
)

report.append(
    f"Precision: "
    f"{original_metrics['precision']:.4f}"
)

report.append(
    f"Recall   : "
    f"{original_metrics['recall']:.4f}"
)

report.append(
    f"F1-score : "
    f"{original_metrics['f1_score']:.4f}"
)

report.append(
    f"ROC-AUC  : "
    f"{original_metrics['roc_auc']:.4f}"
)

report.append("")

report.append(
    "BIAS-CORRECTED MODEL"
)

report.append(
    f"Accuracy : "
    f"{corrected_metrics['accuracy']:.4f}"
)

report.append(
    f"Precision: "
    f"{corrected_metrics['precision']:.4f}"
)

report.append(
    f"Recall   : "
    f"{corrected_metrics['recall']:.4f}"
)

report.append(
    f"F1-score : "
    f"{corrected_metrics['f1_score']:.4f}"
)

report.append(
    f"ROC-AUC  : "
    f"{corrected_metrics['roc_auc']:.4f}"
)

report.append("")

report.append(
    "5-FOLD CROSS-VALIDATION"
)

report.append(
    f"F1 mean: "
    f"{summary['corrected_cv_f1_mean']:.4f}"
)

report.append(
    f"F1 std: "
    f"{summary['corrected_cv_f1_std']:.4f}"
)

report.append(
    f"ROC-AUC mean: "
    f"{summary['corrected_cv_roc_auc_mean']:.4f}"
)

report.append(
    f"ROC-AUC std: "
    f"{summary['corrected_cv_roc_auc_std']:.4f}"
)

report.append("")

if crop_importance is not None:

    report.append(
        f"Corrected crop_encoded importance: "
        f"{crop_importance:.6f}"
    )

    report.append(
        f"Corrected environmental importance: "
        f"{environmental_importance:.6f}"
    )

report.append("")

report.append(
    "CORRECTED MODEL PATH:"
)

report.append(
    str(CORRECTED_MODEL_PATH)
)

report.append("")

report.append(
    "The corrected model must be evaluated by "
    "script 11 before it is used for final crop "
    "recommendations."
)

report_path = (
    OUTPUT_DIR
    / "bias_correction_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("BIAS CORRECTION COMPLETED")
print("=" * 70)

print(
    "\nCorrected model:"
)

print(
    CORRECTED_MODEL_PATH
)

print(
    "\nFile exists:",
    CORRECTED_MODEL_PATH.exists()
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print(
    "Run:"
)

print(
    r'python scripts\11_validate_corrected_model.py'
)

print("=" * 70)