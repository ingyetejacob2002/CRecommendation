"""
10_correct_crop_bias.py

INVESTIGATING AND CORRECTING CROP RECOMMENDATION BIAS

This script:
1. Loads the training and test datasets.
2. Audits crop-level suitability imbalance.
3. Calculates crop-aware sample weights.
4. Trains a bias-corrected Decision Tree.
5. Compares the original and corrected models.
6. Performs cross-validation WITHOUT deprecated fit_params.
7. Analyzes feature importance.
8. Generates corrected crop predictions.
9. Rebuilds crop recommendations.
10. Analyzes whether recommendation bias has improved.
11. Saves the corrected model.
12. Generates a detailed correction report.
"""

# ================================================================
# IMPORTS
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ================================================================
# PROJECT PATH
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# ================================================================
# INPUT PATHS
# ================================================================

TRAINING_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "north_central_training_dataset.csv"
)

# IMPORTANT:
# These are the ACTUAL paths shown in your terminal.
X_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_train.csv"
)

X_TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "X_test.csv"
)

Y_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "y_train.csv"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "y_test.csv"
)

ORIGINAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_crop_suitability_model.joblib"
)

RECOMMENDATION_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recommendations",
    "crop_recommendations.csv"
)


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "model_results",
    "bias_correction"
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "plots"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "corrected"
)

RECOMMENDATION_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recommendations",
    "corrected"
)


for directory in [
    OUTPUT_DIR,
    TABLE_DIR,
    PLOT_DIR,
    MODEL_DIR,
    RECOMMENDATION_OUTPUT_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def section(number, title):
    print("\n" + "=" * 70)
    print(f"[{number}] {title}")
    print("=" * 70)


def check_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"Found {name}:")
    print(path)


def calculate_metrics(model, X, y):
    """
    Calculate classification metrics.
    """

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = predictions

    metrics = {
        "accuracy": accuracy_score(y, predictions),
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
        )
    }

    return metrics


def train_crop_weights(training_data):
    """
    Calculate crop-aware weights.

    The goal is to prevent crops with extreme suitability
    distributions from dominating the model.
    """

    crop_stats = (
        training_data
        .groupby("crop")
        .agg(
            total_records=("suitable", "count"),
            suitable_records=("suitable", "sum")
        )
        .reset_index()
    )

    crop_stats["suitability_rate"] = (
        crop_stats["suitable_records"]
        /
        crop_stats["total_records"]
    )

    overall_rate = training_data["suitable"].mean()

    # Calculate a moderated weight.
    #
    # We do NOT use extreme inverse weights because
    # Sorghum has zero suitable observations.
    #
    # Instead, use a bounded correction around the
    # overall suitability rate.

    epsilon = 0.05

    crop_stats["adjusted_rate"] = (
        crop_stats["suitability_rate"]
        .clip(
            lower=epsilon,
            upper=1 - epsilon
        )
    )

    crop_stats["crop_weight"] = (
        overall_rate
        /
        crop_stats["adjusted_rate"]
    )

    # Prevent excessively large weights.
    crop_stats["crop_weight"] = (
        crop_stats["crop_weight"]
        .clip(
            lower=0.50,
            upper=2.00
        )
    )

    # Normalize around 1.
    crop_stats["crop_weight"] = (
        crop_stats["crop_weight"]
        /
        crop_stats["crop_weight"].mean()
    )

    return crop_stats


# ================================================================
# 1. CHECK INPUT FILES
# ================================================================

section(1, "CHECKING INPUT FILES")

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

# Recommendation dataset is optional for model correction.
# If it exists, we will analyze it later.

if os.path.exists(RECOMMENDATION_PATH):
    print("Found Recommendation dataset:")
    print(RECOMMENDATION_PATH)
else:
    print(
        "\nRecommendation dataset not found."
        "\nThe model correction will continue."
    )


# ================================================================
# 2. LOAD DATASETS
# ================================================================

section(2, "LOADING DATASETS")

training_data = pd.read_csv(
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
)

y_test = pd.read_csv(
    Y_TEST_PATH
)

original_model = joblib.load(
    ORIGINAL_MODEL_PATH
)


# Convert y to one-dimensional arrays.

if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]

if isinstance(y_test, pd.DataFrame):
    y_test = y_test.iloc[:, 0]


y_train = np.asarray(
    y_train
).ravel()

y_test = np.asarray(
    y_test
).ravel()


print(
    f"Training dataset shape: {training_data.shape}"
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
    f"Original model: "
    f"{type(original_model).__name__}"
)


# ================================================================
# 3. VALIDATE DATASET STRUCTURE
# ================================================================

section(3, "VALIDATING DATASET STRUCTURE")

required_training_columns = [
    "crop",
    "suitable"
]

missing_columns = [
    column
    for column in required_training_columns
    if column not in training_data.columns
]

if missing_columns:
    raise ValueError(
        "Missing required training columns:\n"
        + "\n".join(missing_columns)
    )

if len(X_train) != len(y_train):
    raise ValueError(
        "X_train and y_train have different numbers of records."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test have different numbers of records."
    )

if X_train.shape[1] != X_test.shape[1]:
    raise ValueError(
        "X_train and X_test have different numbers of features."
    )

print(
    "SUCCESS: Dataset structure is valid."
)


# ================================================================
# 4. AUDIT TARGET DISTRIBUTION
# ================================================================

section(4, "AUDITING CROP-LEVEL SUITABILITY IMBALANCE")

target_distribution = (
    training_data["suitable"]
    .value_counts()
    .sort_index()
)

target_percentage = (
    training_data["suitable"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print("\nOverall target distribution:")
print(target_distribution)

print("\nOverall percentages:")
print(
    target_percentage.round(2)
)


# Crop-level suitability

crop_stats = (
    training_data
    .groupby("crop")
    .agg(
        total_records=("suitable", "count"),
        suitable_records=("suitable", "sum")
    )
    .reset_index()
)

crop_stats["mean_suitability"] = (
    crop_stats["suitable_records"]
    /
    crop_stats["total_records"]
)

crop_stats["suitability_percentage"] = (
    crop_stats["mean_suitability"]
    * 100
)

crop_stats = crop_stats.sort_values(
    "suitability_percentage",
    ascending=False
)

print("\nCrop-level suitability:")
print(crop_stats.to_string(index=False))


# Save

crop_stats.to_csv(
    os.path.join(
        TABLE_DIR,
        "crop_target_distribution_before.csv"
    ),
    index=False
)


# ================================================================
# 5. IDENTIFY EXTREME IMBALANCE
# ================================================================

section(5, "IDENTIFYING EXTREME CROP IMBALANCE")

zero_suitable = crop_stats[
    crop_stats["suitable_records"] == 0
]["crop"].tolist()

low_suitability = crop_stats[
    crop_stats["suitability_percentage"] < 10
]["crop"].tolist()

high_suitability = crop_stats[
    crop_stats["suitability_percentage"] > 80
]["crop"].tolist()

print(
    f"\nCrops with zero suitable records: "
    f"{len(zero_suitable)}"
)

print(zero_suitable)

print(
    f"\nCrops below 10% suitability: "
    f"{len(low_suitability)}"
)

print(low_suitability)

print(
    f"\nCrops above 80% suitability: "
    f"{len(high_suitability)}"
)

print(high_suitability)


# ================================================================
# 6. PLOT BEFORE CORRECTION
# ================================================================

section(6, "CREATING CROP SUITABILITY PLOT")

before_plot_dir = os.path.join(
    PLOT_DIR,
    "crop_suitability"
)

os.makedirs(
    before_plot_dir,
    exist_ok=True
)

plt.figure(
    figsize=(12, 7)
)

plt.bar(
    crop_stats["crop"],
    crop_stats["suitability_percentage"]
)

plt.axhline(
    50,
    linestyle="--"
)

plt.xlabel(
    "Crop"
)

plt.ylabel(
    "Suitability Percentage"
)

plt.title(
    "Crop Suitability Distribution Before Bias Correction"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

before_plot_path = os.path.join(
    before_plot_dir,
    "crop_suitability_before_correction.png"
)

plt.savefig(
    before_plot_path,
    dpi=300
)

plt.close()

print(
    f"Saved plot:\n{before_plot_path}"
)


# ================================================================
# 7. CALCULATE CROP-AWARE TRAINING WEIGHTS
# ================================================================

section(7, "CALCULATING CROP-AWARE TRAINING WEIGHTS")

# We need crop information corresponding to X_train.
#
# The training dataset has 1210 rows while X_train has
# 968 rows. Therefore, we must identify the training rows
# correctly.
#
# The safest approach is to use the same index relationship
# if the preprocessing preserved row indices.

if "crop_encoded" not in X_train.columns:
    raise ValueError(
        "X_train does not contain crop_encoded."
    )

# Build a mapping from crop_encoded to crop name.

crop_mapping = (
    training_data[
        ["crop", "crop_encoded"]
    ]
    .drop_duplicates()
)

# Check for ambiguous mappings.

mapping_counts = (
    crop_mapping
    .groupby("crop_encoded")["crop"]
    .nunique()
)

if (mapping_counts > 1).any():
    raise ValueError(
        "crop_encoded maps to multiple crop names."
    )


# Attach crop name to training data.

crop_code_to_name = dict(
    zip(
        crop_mapping["crop_encoded"],
        crop_mapping["crop"]
    )
)


# ================================================================
# Recover crop information for X_train
# ================================================================

# Because X_train contains crop_encoded, we can directly
# map each training record to its crop.

X_train_crop_codes = (
    X_train["crop_encoded"]
    .values
)

X_test_crop_codes = (
    X_test["crop_encoded"]
    .values
)


# Calculate crop statistics using the ACTUAL training split,
# not the complete dataset.

training_split = pd.DataFrame(
    {
        "crop_encoded": X_train_crop_codes,
        "suitable": y_train
    }
)

crop_weight_stats = (
    training_split
    .groupby("crop_encoded")
    .agg(
        training_records=("suitable", "count"),
        suitable_records=("suitable", "sum")
    )
    .reset_index()
)

crop_weight_stats[
    "training_suitability_rate"
] = (
    crop_weight_stats["suitable_records"]
    /
    crop_weight_stats["training_records"]
)


overall_training_rate = (
    training_split["suitable"].mean()
)


# Calculate moderated weights.

epsilon = 0.05

crop_weight_stats[
    "adjusted_rate"
] = (
    crop_weight_stats[
        "training_suitability_rate"
    ]
    .clip(
        lower=epsilon,
        upper=1 - epsilon
    )
)

crop_weight_stats[
    "crop_weight"
] = (
    overall_training_rate
    /
    crop_weight_stats[
        "adjusted_rate"
    ]
)

# Bound the correction.

crop_weight_stats[
    "crop_weight"
] = (
    crop_weight_stats[
        "crop_weight"
    ]
    .clip(
        lower=0.50,
        upper=2.00
    )
)

# Normalize.

crop_weight_stats[
    "crop_weight"
] = (
    crop_weight_stats["crop_weight"]
    /
    crop_weight_stats["crop_weight"].mean()
)


# Add crop names.

crop_weight_stats[
    "crop"
] = crop_weight_stats[
    "crop_encoded"
].map(
    crop_code_to_name
)

crop_weight_stats = crop_weight_stats[
    [
        "crop",
        "crop_encoded",
        "training_suitability_rate",
        "training_records",
        "suitable_records",
        "crop_weight"
    ]
].sort_values(
    "crop_weight",
    ascending=False
)

print(
    "\nCalculated crop-aware weights:"
)

print(
    crop_weight_stats.to_string(
        index=False
    )
)

crop_weight_stats.to_csv(
    os.path.join(
        TABLE_DIR,
        "crop_aware_weights.csv"
    ),
    index=False
)


# ================================================================
# 8. GENERATE SAMPLE WEIGHTS
# ================================================================

section(8, "GENERATING SAMPLE WEIGHTS")

# IMPORTANT:
# np.asarray() can return a read-only array in some situations.
#
# Therefore we explicitly create a writable COPY.

sample_weights = np.ones(
    len(y_train),
    dtype=float
)


# Class weights.

class_counts = pd.Series(
    y_train
).value_counts()

n_samples = len(y_train)
n_classes = len(class_counts)

class_weights = {
    cls: n_samples /
    (
        n_classes *
        count
    )
    for cls, count
    in class_counts.items()
}

print(
    "\nClass weights:"
)

print(
    class_weights
)


# Crop weights dictionary.

crop_weight_dictionary = dict(
    zip(
        crop_weight_stats[
            "crop_encoded"
        ],
        crop_weight_stats[
            "crop_weight"
        ]
    )
)


# Apply crop-aware weights.

for i in range(
    len(sample_weights)
):

    crop_code = X_train.iloc[
        i
    ]["crop_encoded"]

    crop_weight = crop_weight_dictionary.get(
        crop_code,
        1.0
    )

    class_weight = class_weights.get(
        y_train[i],
        1.0
    )

    sample_weights[i] = (
        crop_weight
        *
        class_weight
    )


# Normalize final sample weights.

sample_weights = (
    sample_weights
    /
    np.mean(sample_weights)
)


# Make absolutely sure it is writable.

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


# ================================================================
# 9. TRAIN CORRECTED MODEL
# ================================================================

section(9, "TRAINING BIAS-CORRECTED DECISION TREE")

corrected_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)


corrected_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print(
    "\nSUCCESS: Bias-corrected Decision Tree trained."
)


# ================================================================
# 10. MODEL EVALUATION
# ================================================================

section(10, "EVALUATING ORIGINAL AND CORRECTED MODELS")


original_metrics = calculate_metrics(
    original_model,
    X_test,
    y_test
)

corrected_metrics = calculate_metrics(
    corrected_model,
    X_test,
    y_test
)


print(
    "\nOriginal Decision Tree"
)

for key, value in original_metrics.items():

    print(
        f"{key.replace('_', ' ').title():10s}: "
        f"{value:.4f}"
    )


print(
    "\nBias-Corrected Decision Tree"
)

for key, value in corrected_metrics.items():

    print(
        f"{key.replace('_', ' ').title():10s}: "
        f"{value:.4f}"
    )


comparison = pd.DataFrame(
    [
        {
            "model": "Original Decision Tree",
            **original_metrics
        },
        {
            "model": "Bias-Corrected Decision Tree",
            **corrected_metrics
        }
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
    os.path.join(
        TABLE_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

corrected_predictions = (
    corrected_model.predict(
        X_test
    )
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


print(
    "\nCorrected model classification report:"
)

print(
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


# Save confusion matrix.

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Not Suitable",
        "Actual Suitable"
    ],
    columns=[
        "Predicted Not Suitable",
        "Predicted Suitable"
    ]
)

cm_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "corrected_confusion_matrix.csv"
    )
)


# ================================================================
# 11. CROSS-VALIDATION
# ================================================================

section(
    11,
    "CROSS-VALIDATING BIAS-CORRECTED MODEL"
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# IMPORTANT:
# Do NOT use fit_params here.
#
# Your version of scikit-learn rejects:
#
# cross_validate(..., fit_params=...)
#
# Therefore we perform manual CV so sample weights
# can be generated separately inside every fold.

cv_results_list = []


for fold_number, (
    train_indices,
    validation_indices
) in enumerate(
    cv.split(
        X_train,
        y_train
    ),
    start=1
):

    X_cv_train = (
        X_train.iloc[
            train_indices
        ]
    )

    X_cv_valid = (
        X_train.iloc[
            validation_indices
        ]
    )

    y_cv_train = (
        y_train[
            train_indices
        ]
    )

    y_cv_valid = (
        y_train[
            validation_indices
        ]
    )


    # ------------------------------------------------------------
    # Calculate class weights for this fold
    # ------------------------------------------------------------

    fold_class_counts = (
        pd.Series(
            y_cv_train
        ).value_counts()
    )

    fold_n_samples = len(
        y_cv_train
    )

    fold_n_classes = len(
        fold_class_counts
    )

    fold_class_weights = {
        cls:
        fold_n_samples /
        (
            fold_n_classes *
            count
        )
        for cls, count
        in fold_class_counts.items()
    }


    # ------------------------------------------------------------
    # Crop weights for this fold
    # ------------------------------------------------------------

    fold_crop_data = pd.DataFrame(
        {
            "crop_encoded":
                X_cv_train[
                    "crop_encoded"
                ].values,

            "suitable":
                y_cv_train
        }
    )


    fold_crop_stats = (
        fold_crop_data
        .groupby(
            "crop_encoded"
        )
        .agg(
            total=("suitable", "count"),
            suitable=("suitable", "sum")
        )
        .reset_index()
    )


    fold_crop_stats[
        "rate"
    ] = (
        fold_crop_stats[
            "suitable"
        ]
        /
        fold_crop_stats[
            "total"
        ]
    )


    fold_overall_rate = (
        fold_crop_data[
            "suitable"
        ].mean()
    )


    fold_crop_stats[
        "adjusted_rate"
    ] = (
        fold_crop_stats[
            "rate"
        ]
        .clip(
            lower=0.05,
            upper=0.95
        )
    )


    fold_crop_stats[
        "weight"
    ] = (
        fold_overall_rate
        /
        fold_crop_stats[
            "adjusted_rate"
        ]
    )


    fold_crop_stats[
        "weight"
    ] = (
        fold_crop_stats[
            "weight"
        ]
        .clip(
            lower=0.50,
            upper=2.00
        )
    )


    fold_crop_stats[
        "weight"
    ] = (
        fold_crop_stats[
            "weight"
        ]
        /
        fold_crop_stats[
            "weight"
        ].mean()
    )


    fold_crop_weights = dict(
        zip(
            fold_crop_stats[
                "crop_encoded"
            ],
            fold_crop_stats[
                "weight"
            ]
        )
    )


    # ------------------------------------------------------------
    # Generate fold sample weights
    # ------------------------------------------------------------

    fold_sample_weights = np.ones(
        len(y_cv_train),
        dtype=float
    )


    for j in range(
        len(y_cv_train)
    ):

        crop_code = (
            X_cv_train.iloc[
                j
            ]["crop_encoded"]
        )

        crop_weight = (
            fold_crop_weights.get(
                crop_code,
                1.0
            )
        )

        class_weight = (
            fold_class_weights.get(
                y_cv_train[j],
                1.0
            )
        )

        fold_sample_weights[j] = (
            crop_weight
            *
            class_weight
        )


    fold_sample_weights /= (
        np.mean(
            fold_sample_weights
        )
    )


    # ------------------------------------------------------------
    # Train fold model
    # ------------------------------------------------------------

    fold_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )


    fold_model.fit(
        X_cv_train,
        y_cv_train,
        sample_weight=fold_sample_weights
    )


    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    fold_predictions = (
        fold_model.predict(
            X_cv_valid
        )
    )

    fold_probabilities = (
        fold_model.predict_proba(
            X_cv_valid
        )[:, 1]
    )


    fold_metrics = {
        "fold": fold_number,

        "accuracy":
            accuracy_score(
                y_cv_valid,
                fold_predictions
            ),

        "precision":
            precision_score(
                y_cv_valid,
                fold_predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_cv_valid,
                fold_predictions,
                zero_division=0
            ),

        "f1_score":
            f1_score(
                y_cv_valid,
                fold_predictions,
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_cv_valid,
                fold_probabilities
            )
    }


    cv_results_list.append(
        fold_metrics
    )


cv_results_df = pd.DataFrame(
    cv_results_list
)

print(
    "\nCross-validation results:"
)

print(
    cv_results_df.to_string(
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
            cv_results_df[
                "accuracy"
            ].mean(),

            cv_results_df[
                "precision"
            ].mean(),

            cv_results_df[
                "recall"
            ].mean(),

            cv_results_df[
                "f1_score"
            ].mean(),

            cv_results_df[
                "roc_auc"
            ].mean()
        ],

        "std": [
            cv_results_df[
                "accuracy"
            ].std(),

            cv_results_df[
                "precision"
            ].std(),

            cv_results_df[
                "recall"
            ].std(),

            cv_results_df[
                "f1_score"
            ].std(),

            cv_results_df[
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

cv_results_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "corrected_model_cv_results.csv"
    ),
    index=False
)

cv_summary.to_csv(
    os.path.join(
        TABLE_DIR,
        "corrected_model_cv_summary.csv"
    ),
    index=False
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

section(
    12,
    "ANALYZING CORRECTED MODEL FEATURE IMPORTANCE"
)


feature_importance = pd.DataFrame(
    {
        "feature":
            X_train.columns,

        "importance":
            corrected_model.feature_importances_
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print(
    "\nCorrected model feature importance:"
)

print(
    feature_importance.to_string(
        index=False
    )
)

feature_importance.to_csv(
    os.path.join(
        TABLE_DIR,
        "corrected_model_feature_importance.csv"
    ),
    index=False
)


# Crop importance.

crop_importance = feature_importance[
    feature_importance[
        "feature"
    ] == "crop_encoded"
]["importance"].sum()


environmental_importance = (
    feature_importance[
        feature_importance[
            "feature"
        ] != "crop_encoded"
    ]["importance"].sum()
)


print(
    f"\nCrop encoded importance: "
    f"{crop_importance:.6f}"
)

print(
    f"Environmental importance: "
    f"{environmental_importance:.6f}"
)


# ================================================================
# PERMUTATION IMPORTANCE
# ================================================================

print(
    "\nCalculating permutation importance..."
)

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
)

permutation_df = (
    permutation_df
    .sort_values(
        "importance_mean",
        ascending=False
    )
)

print(
    "\nPermutation importance:"
)

print(
    permutation_df.head(
        15
    ).to_string(
        index=False
    )
)

permutation_df.to_csv(
    os.path.join(
        TABLE_DIR,
        "corrected_permutation_importance.csv"
    ),
    index=False
)


# ================================================================
# SAVE CORRECTED MODEL
# ================================================================

corrected_model_path = os.path.join(
    MODEL_DIR,
    "bias_corrected_crop_suitability_model.joblib"
)

joblib.dump(
    corrected_model,
    corrected_model_path
)

print(
    f"\nCorrected model saved to:"
)

print(
    corrected_model_path
)


# ================================================================
# GENERATE TEST PREDICTION DATA
# ================================================================

test_prediction_data = X_test.copy()

test_prediction_data[
    "actual_suitable"
] = y_test

test_prediction_data[
    "predicted_suitable"
] = corrected_predictions

test_prediction_data[
    "suitability_probability"
] = corrected_probabilities

test_prediction_data.to_csv(
    os.path.join(
        TABLE_DIR,
        "corrected_test_predictions.csv"
    ),
    index=False
)


# ================================================================
# RECOMMENDATION BIAS ANALYSIS
# ================================================================

section(
    13,
    "ANALYZING CORRECTED RECOMMENDATION BIAS"
)


if os.path.exists(
    RECOMMENDATION_PATH
):

    recommendation_data = pd.read_csv(
        RECOMMENDATION_PATH
    )

    required_recommendation_columns = [
        "location_id",
        "state",
        "lga",
        "crop"
    ]

    missing_recommendation_columns = [
        col
        for col
        in required_recommendation_columns
        if col not in recommendation_data.columns
    ]

    if missing_recommendation_columns:

        print(
            "\nRecommendation dataset exists but "
            "does not contain all expected columns."
        )

    else:

        # --------------------------------------------------------
        # Create environmental/crop feature combinations
        # --------------------------------------------------------

        if "crop_encoded" in X_test.columns:

            # We use the original recommendation dataset
            # structure to prepare corrected predictions.
            #
            # If the recommendation dataset already contains
            # the model-ready features, use them.

            feature_columns = list(
                X_train.columns
            )

            missing_features = [
                col
                for col in feature_columns
                if col not in recommendation_data.columns
            ]

            if not missing_features:

                recommendation_features = (
                    recommendation_data[
                        feature_columns
                    ]
                )

                corrected_recommendation_predictions = (
                    corrected_model.predict(
                        recommendation_features
                    )
                )

                corrected_recommendation_probabilities = (
                    corrected_model.predict_proba(
                        recommendation_features
                    )[:, 1]
                )

                corrected_recommendations = (
                    recommendation_data[
                        [
                            "location_id",
                            "state",
                            "lga",
                            "crop"
                        ]
                    ].copy()
                )

                corrected_recommendations[
                    "suitable"
                ] = (
                    corrected_recommendation_predictions
                )

                corrected_recommendations[
                    "suitability_probability"
                ] = (
                    corrected_recommendation_probabilities
                )


                # ------------------------------------------------
                # Rank crops within each location
                # ------------------------------------------------

                corrected_recommendations[
                    "recommendation_rank"
                ] = (
                    corrected_recommendations
                    .groupby(
                        "location_id"
                    )[
                        "suitability_probability"
                    ]
                    .rank(
                        method="first",
                        ascending=False
                    )
                    .astype(int)
                )


                # Recommendation category

                def recommendation_category(
                    probability
                ):

                    if probability >= 0.80:
                        return "Highly Recommended"

                    elif probability >= 0.60:
                        return "Recommended"

                    elif probability >= 0.40:
                        return "Moderately Suitable"

                    else:
                        return "Not Recommended"


                corrected_recommendations[
                    "recommendation_category"
                ] = (
                    corrected_recommendations[
                        "suitability_probability"
                    ]
                    .apply(
                        recommendation_category
                    )
                )


                # Save full corrected recommendations.

                corrected_recommendations_path = os.path.join(
                    RECOMMENDATION_OUTPUT_DIR,
                    "corrected_crop_recommendations.csv"
                )

                corrected_recommendations.to_csv(
                    corrected_recommendations_path,
                    index=False
                )


                # ------------------------------------------------
                # Top 3
                # ------------------------------------------------

                top3_corrected = (
                    corrected_recommendations[
                        corrected_recommendations[
                            "recommendation_rank"
                        ] <= 3
                    ]
                    .sort_values(
                        [
                            "location_id",
                            "recommendation_rank"
                        ]
                    )
                )

                top3_path = os.path.join(
                    RECOMMENDATION_OUTPUT_DIR,
                    "corrected_top_crop_recommendations.csv"
                )

                top3_corrected.to_csv(
                    top3_path,
                    index=False
                )


                # ------------------------------------------------
                # Top-1 distribution
                # ------------------------------------------------

                top1 = (
                    corrected_recommendations[
                        corrected_recommendations[
                            "recommendation_rank"
                        ] == 1
                    ]
                )

                top1_distribution = (
                    top1[
                        "crop"
                    ]
                    .value_counts()
                    .reset_index()
                )

                top1_distribution.columns = [
                    "crop",
                    "top1_count"
                ]

                top1_distribution[
                    "top1_percentage"
                ] = (
                    top1_distribution[
                        "top1_count"
                    ]
                    /
                    len(top1)
                    *
                    100
                )


                print(
                    "\nCorrected top-1 distribution:"
                )

                print(
                    top1_distribution.to_string(
                        index=False
                    )
                )


                top1_distribution.to_csv(
                    os.path.join(
                        TABLE_DIR,
                        "corrected_top1_distribution.csv"
                    ),
                    index=False
                )


                # ------------------------------------------------
                # Top-3 diversity
                # ------------------------------------------------

                top3_combinations = (
                    top3_corrected
                    .sort_values(
                        [
                            "location_id",
                            "recommendation_rank"
                        ]
                    )
                    .groupby(
                        "location_id"
                    )["crop"]
                    .apply(
                        lambda x:
                        " | ".join(x)
                    )
                    .value_counts()
                    .reset_index()
                )

                top3_combinations.columns = [
                    "top_3_crops",
                    "locations"
                ]


                unique_top3 = (
                    top3_combinations.shape[0]
                )

                total_locations = (
                    corrected_recommendations[
                        "location_id"
                    ].nunique()
                )

                diversity_percentage = (
                    unique_top3
                    /
                    total_locations
                    *
                    100
                )


                print(
                    f"\nUnique corrected top-3 "
                    f"combinations: {unique_top3}"
                )

                print(
                    f"Corrected top-3 diversity: "
                    f"{diversity_percentage:.2f}%"
                )


                top3_combinations.to_csv(
                    os.path.join(
                        TABLE_DIR,
                        "corrected_top3_combinations.csv"
                    ),
                    index=False
                )


                # ------------------------------------------------
                # Recommendation category distribution
                # ------------------------------------------------

                category_distribution = (
                    corrected_recommendations[
                        "recommendation_category"
                    ]
                    .value_counts()
                    .reset_index()
                )

                category_distribution.columns = [
                    "recommendation_category",
                    "count"
                ]

                category_distribution[
                    "percentage"
                ] = (
                    category_distribution[
                        "count"
                    ]
                    /
                    len(
                        corrected_recommendations
                    )
                    *
                    100
                )


                category_distribution.to_csv(
                    os.path.join(
                        TABLE_DIR,
                        "corrected_category_distribution.csv"
                    ),
                    index=False
                )


                # ------------------------------------------------
                # Corrected recommendation summary
                # ------------------------------------------------

                dominant_top1_crop = (
                    top1_distribution.iloc[
                        0
                    ]["crop"]
                )

                dominant_top1_count = int(
                    top1_distribution.iloc[
                        0
                    ]["top1_count"]
                )

                dominant_top1_percentage = float(
                    top1_distribution.iloc[
                        0
                    ]["top1_percentage"]
                )


                corrected_summary = pd.DataFrame(
                    [
                        {
                            "total_locations":
                                total_locations,

                            "total_crops":
                                corrected_recommendations[
                                    "crop"
                                ].nunique(),

                            "total_predictions":
                                len(
                                    corrected_recommendations
                                ),

                            "recommended_predictions":
                                (
                                    corrected_recommendations[
                                        "suitable"
                                    ] == 1
                                ).sum(),

                            "recommended_percentage":
                                (
                                    corrected_recommendations[
                                        "suitable"
                                    ].mean()
                                    *
                                    100
                                ),

                            "dominant_top1_crop":
                                dominant_top1_crop,

                            "top1_dominance_percentage":
                                dominant_top1_percentage,

                            "unique_top3_combinations":
                                unique_top3,

                            "top3_diversity_percentage":
                                diversity_percentage
                        }
                    ]
                )


                corrected_summary.to_csv(
                    os.path.join(
                        TABLE_DIR,
                        "corrected_recommendation_summary.csv"
                    ),
                    index=False
                )


                print(
                    "\nCorrected recommendation analysis "
                    "completed successfully."
                )

                print(
                    f"Dominant corrected top-1 crop: "
                    f"{dominant_top1_crop}"
                )

                print(
                    f"Corrected top-1 dominance: "
                    f"{dominant_top1_percentage:.2f}%"
                )

            else:

                print(
                    "\nRecommendation dataset does not "
                    "contain all model features."
                )

                print(
                    "Missing features:"
                )

                print(
                    missing_features
                )

else:

    print(
        "\nRecommendation dataset not available."
    )


# ================================================================
# 14. GENERATE FINAL REPORT
# ================================================================

section(
    14,
    "GENERATING BIAS CORRECTION REPORT"
)


report_path = os.path.join(
    OUTPUT_DIR,
    "bias_correction_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CROP RECOMMENDATION BIAS CORRECTION REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )


    report.write(
        "DATASET INFORMATION\n"
    )

    report.write(
        f"Training records: "
        f"{len(training_data)}\n"
    )

    report.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    report.write(
        f"Test samples: "
        f"{len(X_test)}\n\n"
    )


    report.write(
        "ORIGINAL MODEL PERFORMANCE\n"
    )

    for key, value in original_metrics.items():

        report.write(
            f"{key}: {value:.6f}\n"
        )

    report.write(
        "\n"
    )


    report.write(
        "BIAS-CORRECTED MODEL PERFORMANCE\n"
    )

    for key, value in corrected_metrics.items():

        report.write(
            f"{key}: {value:.6f}\n"
        )

    report.write(
        "\n"
    )


    report.write(
        "CROSS-VALIDATION\n"
    )

    for _, row in cv_summary.iterrows():

        report.write(
            f"{row['metric']}: "
            f"{row['mean']:.6f} "
            f"+/- "
            f"{row['std']:.6f}\n"
        )

    report.write(
        "\n"
    )


    report.write(
        "FEATURE IMPORTANCE\n"
    )

    report.write(
        f"Crop encoded importance: "
        f"{crop_importance:.6f}\n"
    )

    report.write(
        f"Environmental importance: "
        f"{environmental_importance:.6f}\n"
    )

    report.write(
        "\n"
    )


    report.write(
        "CROP IMBALANCE\n"
    )

    report.write(
        "Zero suitable crops:\n"
    )

    for crop in zero_suitable:

        report.write(
            f"- {crop}\n"
        )

    report.write(
        "\nCrops below 10% suitability:\n"
    )

    for crop in low_suitability:

        report.write(
            f"- {crop}\n"
        )

    report.write(
        "\nCrops above 80% suitability:\n"
    )

    for crop in high_suitability:

        report.write(
            f"- {crop}\n"
        )

    report.write(
        "\n"
    )


    report.write(
        "INTERPRETATION\n"
    )

    if crop_importance > 0.50:

        report.write(
            "- Crop identity remains a major predictor "
            "of suitability.\n"
        )

    else:

        report.write(
            "- Crop identity is no longer the dominant "
            "predictor.\n"
        )


    if os.path.exists(
        RECOMMENDATION_PATH
    ):

        report.write(
            "- Corrected recommendation outputs were "
            "generated where compatible model features "
            "were available.\n"
        )


    report.write(
        "\n"
    )

    report.write(
        "MODEL PATH\n"
    )

    report.write(
        corrected_model_path
        +
        "\n"
    )


print(
    f"\nBias correction report saved to:"
)

print(
    report_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print(
    "\n"
    + "=" * 70
)

print(
    "BIAS CORRECTION COMPLETED"
)

print(
    "=" * 70
)

print(
    f"Original F1-score: "
    f"{original_metrics['f1_score']:.4f}"
)

print(
    f"Corrected F1-score: "
    f"{corrected_metrics['f1_score']:.4f}"
)

print(
    f"Original ROC-AUC: "
    f"{original_metrics['roc_auc']:.4f}"
)

print(
    f"Corrected ROC-AUC: "
    f"{corrected_metrics['roc_auc']:.4f}"
)

print(
    f"\nCorrected model:"
)

print(
    corrected_model_path
)

print(
    f"\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nNEXT STEP:"
)

print(
    "Run the recommendation engine again using the "
    "bias-corrected model and then evaluate the new "
    "recommendations."
)

print(
    "=" * 70
)