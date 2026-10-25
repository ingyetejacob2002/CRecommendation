from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

X_TRAIN_FILE = PROCESSED_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DIR / "X_test.csv"

Y_TRAIN_FILE = PROCESSED_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

BEST_MODEL_FILE = (
    MODEL_DIR /
    "best_crop_suitability_model.joblib"
)

FINAL_MODEL_FILE = (
    MODEL_DIR /
    "final_crop_suitability_model.joblib"
)

RESULTS_DIR = PROCESSED_DIR / "model_results"
VALIDATION_DIR = RESULTS_DIR / "validation"

COMPARISON_FILE = (
    VALIDATION_DIR /
    "validated_model_comparison.csv"
)

CV_RESULTS_FILE = (
    VALIDATION_DIR /
    "cross_validation_results.csv"
)

FEATURE_IMPORTANCE_FILE = (
    VALIDATION_DIR /
    "feature_importance.csv"
)

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VALIDATING AND TUNING CROP SUITABILITY MODELS")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT DATA
# ============================================================

print("\n[1/10] Checking input datasets...")

required_files = [
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE
]

for file in required_files:

    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )

    print(f"Found: {file}")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2/10] Loading datasets...")

X_train = pd.read_csv(X_TRAIN_FILE)
X_test = pd.read_csv(X_TEST_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_test = pd.read_csv(Y_TEST_FILE).squeeze()

print(f"Training features: {X_train.shape}")
print(f"Testing features:  {X_test.shape}")
print(f"Training target:   {y_train.shape}")
print(f"Testing target:    {y_test.shape}")


# ============================================================
# 3. VALIDATE DATA
# ============================================================

print("\n[3/10] Validating datasets...")

if len(X_train) != len(y_train):
    raise ValueError(
        "X_train and y_train contain different numbers of records."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test contain different numbers of records."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )

if X_train.isna().sum().sum() > 0:
    raise ValueError(
        "Missing values detected in X_train."
    )

if X_test.isna().sum().sum() > 0:
    raise ValueError(
        "Missing values detected in X_test."
    )

if np.isinf(X_train.select_dtypes(include=np.number)).sum().sum() > 0:
    raise ValueError(
        "Infinite values detected in X_train."
    )

if np.isinf(X_test.select_dtypes(include=np.number)).sum().sum() > 0:
    raise ValueError(
        "Infinite values detected in X_test."
    )

print("SUCCESS: Training and testing datasets are valid.")

print(f"Number of features: {X_train.shape[1]}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print(
    f"Target classes: "
    f"{sorted(y_train.unique().tolist())}"
)


# ============================================================
# 4. CROSS-VALIDATION SETUP
# ============================================================

print("\n[4/10] Setting up stratified cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Cross-validation: 5-fold StratifiedKFold")
print("Shuffle: True")
print("Random state: 42")


# ============================================================
# 5. BASELINE CROSS-VALIDATION
# ============================================================

print("\n[5/10] Performing baseline cross-validation...")

baseline_models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = []

for name, model in baseline_models.items():

    print(f"\nCross-validating: {name}")

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    row = {
        "model": name,

        "accuracy_mean":
            results["test_accuracy"].mean(),

        "accuracy_std":
            results["test_accuracy"].std(),

        "precision_mean":
            results["test_precision"].mean(),

        "precision_std":
            results["test_precision"].std(),

        "recall_mean":
            results["test_recall"].mean(),

        "recall_std":
            results["test_recall"].std(),

        "f1_mean":
            results["test_f1"].mean(),

        "f1_std":
            results["test_f1"].std(),

        "roc_auc_mean":
            results["test_roc_auc"].mean(),

        "roc_auc_std":
            results["test_roc_auc"].std()
    }

    cv_results.append(row)

    print(
        f"Accuracy : "
        f"{row['accuracy_mean']:.4f} "
        f"(± {row['accuracy_std']:.4f})"
    )

    print(
        f"Precision: "
        f"{row['precision_mean']:.4f} "
        f"(± {row['precision_std']:.4f})"
    )

    print(
        f"Recall   : "
        f"{row['recall_mean']:.4f} "
        f"(± {row['recall_std']:.4f})"
    )

    print(
        f"F1-score : "
        f"{row['f1_mean']:.4f} "
        f"(± {row['f1_std']:.4f})"
    )

    print(
        f"ROC-AUC  : "
        f"{row['roc_auc_mean']:.4f} "
        f"(± {row['roc_auc_std']:.4f})"
    )


# ============================================================
# 6. HYPERPARAMETER TUNING
# ============================================================

print("\n[6/10] Tuning Decision Tree and Gradient Boosting...")


# ------------------------------------------------------------
# DECISION TREE PARAMETER GRID
# ------------------------------------------------------------

dt_param_grid = {

    "criterion": [
        "gini",
        "entropy",
        "log_loss"
    ],

    "max_depth": [
        3,
        5,
        7,
        10,
        15,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10,
        20
    ],

    "min_samples_leaf": [
        1,
        2,
        5,
        10
    ],

    "class_weight": [
        None,
        "balanced"
    ]
}


print("\nTuning Decision Tree...")

dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(
        random_state=42
    ),
    param_grid=dt_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

dt_grid.fit(
    X_train,
    y_train
)

best_dt = dt_grid.best_estimator_

print("Decision Tree tuning completed.")

print(
    f"Best CV F1-score: "
    f"{dt_grid.best_score_:.4f}"
)

print("Best parameters:")

for parameter, value in dt_grid.best_params_.items():

    print(
        f"  {parameter}: {value}"
    )


# ------------------------------------------------------------
# GRADIENT BOOSTING PARAMETER GRID
# ------------------------------------------------------------

gb_param_grid = {

    "n_estimators": [
        50,
        100,
        150,
        200
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2
    ],

    "max_depth": [
        1,
        2,
        3,
        4,
        5
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        5
    ]
}


print("\nTuning Gradient Boosting...")

gb_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(
        random_state=42
    ),
    param_grid=gb_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=0
)

gb_grid.fit(
    X_train,
    y_train
)

best_gb = gb_grid.best_estimator_

print("Gradient Boosting tuning completed.")

print(
    f"Best CV F1-score: "
    f"{gb_grid.best_score_:.4f}"
)

print("Best parameters:")

for parameter, value in gb_grid.best_params_.items():

    print(
        f"  {parameter}: {value}"
    )


# ============================================================
# 7. EVALUATE TUNED MODELS ON TEST DATA
# ============================================================

print("\n[7/10] Evaluating tuned models on test data...")


models_to_evaluate = {

    "Tuned Decision Tree":
        best_dt,

    "Tuned Gradient Boosting":
        best_gb
}


evaluation_results = []


for name, model in models_to_evaluate.items():

    print("\n" + "-" * 60)
    print(f"MODEL: {name}")
    print("-" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for positive class
    if hasattr(model, "predict_proba"):

        y_probability = model.predict_proba(
            X_test
        )[:, 1]

    else:

        y_probability = None

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    if y_probability is not None:

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

    else:

        roc_auc = np.nan

    # Print metrics
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

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Suitable",
                "Suitable"
            ],
            zero_division=0
        )
    )

    evaluation_results.append({

        "model": name,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "roc_auc": roc_auc
    })

    # --------------------------------------------------------
    # Save confusion matrix figure
    # --------------------------------------------------------

    plt.figure()

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Suitable",
            "Suitable"
        ]
    ).plot()

    plt.title(
        f"{name} - Confusion Matrix"
    )

    safe_name = (
        name
        .lower()
        .replace(" ", "_")
    )

    confusion_file = (
        VALIDATION_DIR /
        f"{safe_name}_confusion_matrix.png"
    )

    plt.savefig(
        confusion_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n"
        f"{confusion_file}"
    )


# ============================================================
# 8. COMPARE MODELS
# ============================================================

print("\n[8/10] Comparing validated models...")

comparison_df = pd.DataFrame(
    evaluation_results
)

comparison_df = comparison_df.sort_values(
    by="f1_score",
    ascending=False
)

print("\nValidated model comparison:")
print(
    comparison_df.to_string(
        index=False
    )
)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)

print(
    f"\nComparison saved to:\n"
    f"{COMPARISON_FILE}"
)


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

print("\n[9/10] Analyzing feature importance...")


# Select best model based on F1
best_model_name = comparison_df.iloc[0]["model"]

if best_model_name == "Tuned Decision Tree":

    final_model = best_dt

else:

    final_model = best_gb


print(
    f"Best validated model: "
    f"{best_model_name}"
)


# Feature importance
if hasattr(
    final_model,
    "feature_importances_"
):

    importance_df = pd.DataFrame({

        "feature":
            X_train.columns,

        "importance":
            final_model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print("\nFeature importance:")

    print(
        importance_df.to_string(
            index=False
        )
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    # --------------------------------------------------------
    # Feature importance plot
    # --------------------------------------------------------

    top_features = (
        importance_df
        .head(15)
        .sort_values(
            by="importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"Top 15 Feature Importances - "
        f"{best_model_name}"
    )

    plt.tight_layout()

    feature_plot = (
        VALIDATION_DIR /
        "feature_importance.png"
    )

    plt.savefig(
        feature_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Feature importance plot saved to:\n"
        f"{feature_plot}"
    )


# ============================================================
# ROC CURVE
# ============================================================

print("\nGenerating ROC curve...")

plt.figure(
    figsize=(9, 7)
)

for name, model in models_to_evaluate.items():

    probability = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Tuned Models"
)

plt.legend()

plt.tight_layout()

roc_file = (
    VALIDATION_DIR /
    "roc_curve.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"ROC curve saved to:\n"
    f"{roc_file}"
)


# ============================================================
# SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_df = pd.DataFrame(
    cv_results
)

cv_df = cv_df.sort_values(
    by="f1_mean",
    ascending=False
)

cv_df.to_csv(
    CV_RESULTS_FILE,
    index=False
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final validated model...")

joblib.dump(
    final_model,
    FINAL_MODEL_FILE
)

print(
    f"Final model saved to:\n"
    f"{FINAL_MODEL_FILE}"
)


# Also overwrite/update the best model path
joblib.dump(
    final_model,
    BEST_MODEL_FILE
)

print(
    f"Best model path updated:\n"
    f"{BEST_MODEL_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

best_row = comparison_df.iloc[0]

print("\n" + "=" * 70)
print("VALIDATION AND TUNING COMPLETED")
print("=" * 70)

print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)

print(
    f"Features         : {X_train.shape[1]}"
)

print(
    f"Models validated : 2 tuned models"
)

print(
    f"Best model       : {best_model_name}"
)

print(
    f"Accuracy         : "
    f"{best_row['accuracy']:.4f}"
)

print(
    f"Precision        : "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall           : "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1-score         : "
    f"{best_row['f1_score']:.4f}"
)

print(
    f"ROC-AUC          : "
    f"{best_row['roc_auc']:.4f}"
)

print("\nOutput directory:")
print(VALIDATION_DIR)

print("\nFinal model:")
print(FINAL_MODEL_FILE)

print("\n" + "=" * 70)