# ================================================================
# 11_validate_corrected_model.py
# VALIDATING CORRECTED CROP SUITABILITY MODEL
# ================================================================

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings("ignore")


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = Path(
    r"C:\Users\DELL\Desktop\projects\Emmanuel Alia\CropRecommendation"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = (
    DATA_DIR
    / "model_results"
    / "corrected_model_validation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# INPUT FILES
# ================================================================

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"

Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

TRAINING_DATASET_PATH = (
    DATA_DIR / "north_central_training_dataset.csv"
)

RECOMMENDATION_DATASET_PATH = (
    DATA_DIR
    / "recommendations"
    / "crop_recommendations.csv"
)

ORIGINAL_MODEL_PATH = (
    MODEL_DIR / "final_crop_suitability_model.joblib"
)


# ================================================================
# FUNCTIONS
# ================================================================

def print_section(number, title):

    print("\n" + "=" * 70)
    print(f"[{number}] {title}")
    print("=" * 70)


def check_file(path, name):

    if path.exists():

        print(f"Found {name}:")
        print(path)

        return True

    else:

        print(f"ERROR: {name} not found:")
        print(path)

        return False


def find_corrected_model():

    """
    Automatically searches the entire models directory
    for likely corrected crop suitability models.
    """

    print("\nSearching for corrected model...")

    if not MODEL_DIR.exists():

        print(
            f"Models directory does not exist:\n"
            f"{MODEL_DIR}"
        )

        return None


    # ------------------------------------------------------------
    # First: exact likely filenames
    # ------------------------------------------------------------

    preferred_names = [

        "crop_suitability_model_corrected.joblib",

        "corrected_crop_suitability_model.joblib",

        "crop_suitability_corrected.joblib",

        "corrected_crop_model.joblib",

        "bias_corrected_crop_suitability_model.joblib",

        "crop_suitability_model_bias_corrected.joblib",

        "final_corrected_crop_suitability_model.joblib",

    ]


    for name in preferred_names:

        matches = list(MODEL_DIR.rglob(name))

        if matches:

            print("\nCorrected model found:")

            print(matches[0])

            return matches[0]


    # ------------------------------------------------------------
    # Second: search all JOBLIB files
    # ------------------------------------------------------------

    all_joblib_files = list(
        MODEL_DIR.rglob("*.joblib")
    )


    if not all_joblib_files:

        print(
            "\nNo .joblib files were found inside:"
        )

        print(MODEL_DIR)

        return None


    print("\nAll model files found:")

    for file in all_joblib_files:

        print(f" - {file}")


    # ------------------------------------------------------------
    # Search filenames containing correction-related keywords
    # ------------------------------------------------------------

    keywords = [

        "correct",
        "bias",
        "weighted",
        "balanced",
        "corrected"

    ]


    candidates = []

    for file in all_joblib_files:

        filename = file.name.lower()

        if any(
            keyword in filename
            for keyword in keywords
        ):

            candidates.append(file)


    if candidates:

        print("\nPossible corrected models:")

        for file in candidates:

            print(f" - {file}")


        # Prefer files containing "corrected"

        corrected_candidates = [

            file for file in candidates
            if "corrected" in file.name.lower()
        ]


        if corrected_candidates:

            selected = corrected_candidates[0]

        else:

            selected = candidates[0]


        print(
            "\nSelected corrected model:"
        )

        print(selected)

        return selected


    print(
        "\nWARNING: No filename clearly indicates "
        "a corrected model."
    )

    return None


def evaluate_model(
    model,
    X,
    y,
    model_name
):

    print(f"\n{model_name}")

    predictions = model.predict(X)

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(X)

            if probabilities.shape[1] == 2:

                probabilities = probabilities[:, 1]

        except Exception:

            probabilities = None


    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )


    if probabilities is not None:

        try:

            roc_auc = roc_auc_score(
                y,
                probabilities
            )

        except Exception:

            roc_auc = np.nan

    else:

        roc_auc = np.nan


    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    if not np.isnan(roc_auc):

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    else:

        print(
            "ROC-AUC  : N/A"
        )


    return {

        "model": model_name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc

    }


# ================================================================
# HEADER
# ================================================================

print("\n" + "=" * 70)

print(
    "VALIDATING CORRECTED CROP SUITABILITY MODEL"
)

print("=" * 70)


# ================================================================
# [1] CHECK INPUT FILES
# ================================================================

print_section(
    1,
    "CHECKING INPUT FILES"
)


required_files = [

    (X_TRAIN_PATH, "X_train"),

    (X_TEST_PATH, "X_test"),

    (Y_TRAIN_PATH, "y_train"),

    (Y_TEST_PATH, "y_test"),

    (
        TRAINING_DATASET_PATH,
        "Training dataset"
    ),

    (
        RECOMMENDATION_DATASET_PATH,
        "Recommendation dataset"
    ),

    (
        ORIGINAL_MODEL_PATH,
        "Original model"
    )

]


for path, name in required_files:

    check_file(
        path,
        name
    )


# ================================================================
# FIND CORRECTED MODEL
# ================================================================

CORRECTED_MODEL_PATH = find_corrected_model()


if CORRECTED_MODEL_PATH is None:

    print("\n" + "=" * 70)

    print(
        "ERROR: CORRECTED MODEL COULD NOT BE FOUND"
    )

    print("=" * 70)

    print(
        "\nPlease run:"
    )

    print(
        "10_correct_crop_bias.py"
    )

    print(
        "\nThen check the models directory:"
    )

    print(MODEL_DIR)

    print(
        "\nIf script 10 completed successfully, "
        "the corrected .joblib file should be somewhere "
        "inside the models directory."
    )

    sys.exit(1)


# ================================================================
# [2] LOAD DATASETS
# ================================================================

print_section(
    2,
    "LOADING DATASETS"
)


X_train = pd.read_csv(
    X_TRAIN_PATH
)

X_test = pd.read_csv(
    X_TEST_PATH
)

y_train = pd.read_csv(
    Y_TRAIN_PATH
).squeeze()

y_test = pd.read_csv(
    Y_TEST_PATH
).squeeze()


training_df = pd.read_csv(
    TRAINING_DATASET_PATH
)

recommendation_df = pd.read_csv(
    RECOMMENDATION_DATASET_PATH
)


print(
    f"X_train shape: {X_train.shape}"
)

print(
    f"X_test shape:  {X_test.shape}"
)

print(
    f"y_train shape: {y_train.shape}"
)

print(
    f"y_test shape:  {y_test.shape}"
)

print(
    f"Training dataset shape: "
    f"{training_df.shape}"
)

print(
    f"Recommendation dataset shape: "
    f"{recommendation_df.shape}"
)


# ================================================================
# [3] LOAD MODELS
# ================================================================

print_section(
    3,
    "LOADING MODELS"
)


original_model = joblib.load(
    ORIGINAL_MODEL_PATH
)

corrected_model = joblib.load(
    CORRECTED_MODEL_PATH
)


print(
    "Original model:"
)

print(
    type(original_model).__name__
)


print(
    "\nCorrected model:"
)

print(
    type(corrected_model).__name__
)


# ================================================================
# [4] MODEL FEATURE VALIDATION
# ================================================================

print_section(
    4,
    "VALIDATING MODEL FEATURE STRUCTURE"
)


print(
    f"X_train features: "
    f"{X_train.shape[1]}"
)


if hasattr(
    original_model,
    "n_features_in_"
):

    print(
        f"Original model expects: "
        f"{original_model.n_features_in_}"
    )


if hasattr(
    corrected_model,
    "n_features_in_"
):

    print(
        f"Corrected model expects: "
        f"{corrected_model.n_features_in_}"
    )


if hasattr(
    corrected_model,
    "n_features_in_"
):

    if (
        corrected_model.n_features_in_
        != X_test.shape[1]
    ):

        raise ValueError(
            "Corrected model feature count "
            "does not match X_test."
        )


print(
    "SUCCESS: Feature structure is valid."
)


# ================================================================
# [5] ORIGINAL MODEL EVALUATION
# ================================================================

print_section(
    5,
    "EVALUATING ORIGINAL MODEL"
)


original_results = evaluate_model(
    original_model,
    X_test,
    y_test,
    "Original Decision Tree"
)


# ================================================================
# [6] CORRECTED MODEL EVALUATION
# ================================================================

print_section(
    6,
    "EVALUATING CORRECTED MODEL"
)


corrected_results = evaluate_model(
    corrected_model,
    X_test,
    y_test,
    "Bias-Corrected Decision Tree"
)


# ================================================================
# [7] MODEL COMPARISON
# ================================================================

print_section(
    7,
    "MODEL COMPARISON"
)


comparison_df = pd.DataFrame(
    [
        original_results,
        corrected_results
    ]
)


print(
    comparison_df.to_string(
        index=False
    )
)


comparison_path = (
    RESULTS_DIR
    / "model_comparison.csv"
)


comparison_df.to_csv(
    comparison_path,
    index=False
)


print(
    "\nSaved:"
)

print(
    comparison_path
)


# ================================================================
# [8] CONFUSION MATRIX
# ================================================================

print_section(
    8,
    "CORRECTED MODEL CONFUSION MATRIX"
)


corrected_predictions = (
    corrected_model.predict(X_test)
)


cm = confusion_matrix(
    y_test,
    corrected_predictions
)


print(cm)


cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)


cm_path = (
    RESULTS_DIR
    / "corrected_model_confusion_matrix.csv"
)


cm_df.to_csv(
    cm_path
)


print(
    "\nSaved:"
)

print(cm_path)


# ================================================================
# [9] CLASSIFICATION REPORT
# ================================================================

print_section(
    9,
    "CORRECTED MODEL CLASSIFICATION REPORT"
)


report = classification_report(
    y_test,
    corrected_predictions,
    target_names=[
        "Not Suitable",
        "Suitable"
    ],
    zero_division=0
)


print(report)


report_path = (
    RESULTS_DIR
    / "corrected_model_classification_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ================================================================
# [10] CROSS-VALIDATION
# ================================================================

print_section(
    10,
    "CROSS-VALIDATION"
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {

    "accuracy": "accuracy",

    "precision": "precision",

    "recall": "recall",

    "f1": "f1",

    "roc_auc": "roc_auc"

}


print(
    "Running 5-fold cross-validation..."
)


cv_results = cross_validate(

    corrected_model,

    X_train,

    y_train,

    cv=cv,

    scoring=scoring,

    n_jobs=-1

)


cv_summary = {

    "accuracy_mean":
        cv_results[
            "test_accuracy"
        ].mean(),

    "accuracy_std":
        cv_results[
            "test_accuracy"
        ].std(),

    "precision_mean":
        cv_results[
            "test_precision"
        ].mean(),

    "precision_std":
        cv_results[
            "test_precision"
        ].std(),

    "recall_mean":
        cv_results[
            "test_recall"
        ].mean(),

    "recall_std":
        cv_results[
            "test_recall"
        ].std(),

    "f1_mean":
        cv_results[
            "test_f1"
        ].mean(),

    "f1_std":
        cv_results[
            "test_f1"
        ].std(),

    "roc_auc_mean":
        cv_results[
            "test_roc_auc"
        ].mean(),

    "roc_auc_std":
        cv_results[
            "test_roc_auc"
        ].std()

}


cv_df = pd.DataFrame(
    [cv_summary]
)


print(
    cv_df.to_string(
        index=False
    )
)


cv_path = (
    RESULTS_DIR
    / "corrected_model_cross_validation.csv"
)


cv_df.to_csv(
    cv_path,
    index=False
)


print(
    "\nSaved:"
)

print(cv_path)


# ================================================================
# [11] FEATURE IMPORTANCE
# ================================================================

print_section(
    11,
    "ANALYZING FEATURE IMPORTANCE"
)


if hasattr(
    corrected_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "feature":
            X_train.columns,

        "importance":
            corrected_model.feature_importances_

    })


    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    print(
        importance_df.to_string(
            index=False
        )
    )


    importance_path = (
        RESULTS_DIR
        / "corrected_model_feature_importance.csv"
    )


    importance_df.to_csv(
        importance_path,
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        importance_path
    )


# ================================================================
# [12] RECOMMENDATION DATASET VALIDATION
# ================================================================

print_section(
    12,
    "VALIDATING RECOMMENDATION DATASET"
)


if "crop" in recommendation_df.columns:

    crop_counts = (
        recommendation_df[
            "crop"
        ]
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nRecommendation records by crop:"
    )

    print(crop_counts)


if "recommendation_rank" in recommendation_df.columns:

    top1 = recommendation_df[
        recommendation_df[
            "recommendation_rank"
        ] == 1
    ]


    if len(top1) > 0:

        top1_distribution = (
            top1["crop"]
            .value_counts()
        )


        print(
            "\nTop-1 recommendation distribution:"
        )

        print(
            top1_distribution
        )


        top1_path = (
            RESULTS_DIR
            / "corrected_top1_distribution.csv"
        )


        top1_distribution.to_csv(
            top1_path,
            header=[
                "count"
            ]
        )


# ================================================================
# [13] FINAL VALIDATION SUMMARY
# ================================================================

print_section(
    13,
    "FINAL VALIDATION SUMMARY"
)


accuracy_change = (
    corrected_results["accuracy"]
    -
    original_results["accuracy"]
)


precision_change = (
    corrected_results["precision"]
    -
    original_results["precision"]
)


recall_change = (
    corrected_results["recall"]
    -
    original_results["recall"]
)


f1_change = (
    corrected_results["f1_score"]
    -
    original_results["f1_score"]
)


roc_change = (
    corrected_results["roc_auc"]
    -
    original_results["roc_auc"]
)


print(
    f"Accuracy change : "
    f"{accuracy_change:+.6f}"
)

print(
    f"Precision change: "
    f"{precision_change:+.6f}"
)

print(
    f"Recall change   : "
    f"{recall_change:+.6f}"
)

print(
    f"F1 change       : "
    f"{f1_change:+.6f}"
)

print(
    f"ROC-AUC change  : "
    f"{roc_change:+.6f}"
)


# ================================================================
# [14] SAVE FINAL REPORT
# ================================================================

print_section(
    14,
    "SAVING FINAL VALIDATION REPORT"
)


summary = {

    "original_model":
        str(
            ORIGINAL_MODEL_PATH
        ),

    "corrected_model":
        str(
            CORRECTED_MODEL_PATH
        ),

    "test_records":
        len(y_test),

    "original_accuracy":
        original_results[
            "accuracy"
        ],

    "corrected_accuracy":
        corrected_results[
            "accuracy"
        ],

    "original_precision":
        original_results[
            "precision"
        ],

    "corrected_precision":
        corrected_results[
            "precision"
        ],

    "original_recall":
        original_results[
            "recall"
        ],

    "corrected_recall":
        corrected_results[
            "recall"
        ],

    "original_f1":
        original_results[
            "f1_score"
        ],

    "corrected_f1":
        corrected_results[
            "f1_score"
        ],

    "original_roc_auc":
        original_results[
            "roc_auc"
        ],

    "corrected_roc_auc":
        corrected_results[
            "roc_auc"
        ],

    "accuracy_change":
        accuracy_change,

    "precision_change":
        precision_change,

    "recall_change":
        recall_change,

    "f1_change":
        f1_change,

    "roc_auc_change":
        roc_change

}


summary_path = (
    RESULTS_DIR
    / "corrected_model_validation_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


print(
    "Validation summary saved:"
)

print(
    summary_path
)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 70)

print(
    "CORRECTED MODEL VALIDATION COMPLETED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nOutput directory:"
)

print(
    RESULTS_DIR
)

print(
    "\nThe corrected model has now been evaluated "
    "against the original model."
)

print("=" * 70)