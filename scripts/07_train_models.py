from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

X_TRAIN_FILE = PROCESSED_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DIR / "X_test.csv"

Y_TRAIN_FILE = PROCESSED_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = PROCESSED_DIR / "model_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TRAINING CROP SUITABILITY CLASSIFICATION MODELS")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1/10] Checking input files...")

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

y_train = pd.read_csv(Y_TRAIN_FILE)
y_test = pd.read_csv(Y_TEST_FILE)

# Convert target DataFrames to Series

if y_train.shape[1] == 1:
    y_train = y_train.iloc[:, 0]
else:
    y_train = y_train.squeeze()

if y_test.shape[1] == 1:
    y_test = y_test.iloc[:, 0]
else:
    y_test = y_test.squeeze()

print(f"Training features: {X_train.shape}")
print(f"Testing features:  {X_test.shape}")
print(f"Training target:   {y_train.shape}")
print(f"Testing target:    {y_test.shape}")


# ============================================================
# 3. VALIDATE DATA
# ============================================================

print("\n[3/10] Validating datasets...")

if X_train.isna().sum().sum() > 0:
    raise ValueError("X_train contains missing values.")

if X_test.isna().sum().sum() > 0:
    raise ValueError("X_test contains missing values.")

if not np.isfinite(X_train.values).all():
    raise ValueError("X_train contains infinite values.")

if not np.isfinite(X_test.values).all():
    raise ValueError("X_test contains infinite values.")

if len(X_train) != len(y_train):
    raise ValueError(
        "X_train and y_train have different numbers of records."
    )

if len(X_test) != len(y_test):
    raise ValueError(
        "X_test and y_test have different numbers of records."
    )

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "X_train and X_test columns do not match."
    )

print("SUCCESS: Training and testing datasets are valid.")

print(f"Number of features: {X_train.shape[1]}")

print(
    f"Target classes: "
    f"{sorted(y_train.unique().tolist())}"
)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n[4/10] Checking target distribution...")

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining percentages:")
print(
    (y_train.value_counts(normalize=True)
     .sort_index() * 100)
     .round(2)
)


# ============================================================
# 5. DEFINE MODELS
# ============================================================

print("\n[5/10] Creating classification models...")


models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ]),

    "KNN": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KNeighborsClassifier(
                n_neighbors=7
            )
        )
    ])
}


print(f"Models to train: {len(models)}")

for model_name in models:
    print(f"- {model_name}")


# ============================================================
# 6. TRAIN MODELS
# ============================================================

print("\n[6/10] Training models...")

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")
    print("-" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Probability / decision score
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        y_score = model.predict_proba(X_test)[:, 1]

    elif hasattr(model, "decision_function"):

        y_score = model.decision_function(X_test)

    else:

        y_score = y_pred

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    try:

        auc = roc_auc_score(
            y_test,
            y_score
        )

    except Exception:

        auc = np.nan

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

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

    # Save model in memory

    trained_models[name] = model

    # Save metrics

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })


# ============================================================
# 7. COMPARE MODELS
# ============================================================

print("\n[7/10] Comparing model performance...")

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_score",
    ascending=False
).reset_index(drop=True)

print("\nModel comparison:")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SELECT BEST MODEL
# ============================================================

print("\n[8/10] Selecting best model...")

best_model_name = results_df.iloc[0]["model"]

best_model = trained_models[
    best_model_name
]

print(
    f"Best model based on F1-score: "
    f"{best_model_name}"
)

best_row = results_df.iloc[0]

print(
    f"Accuracy : {best_row['accuracy']:.4f}"
)

print(
    f"Precision: {best_row['precision']:.4f}"
)

print(
    f"Recall   : {best_row['recall']:.4f}"
)

print(
    f"F1-score : {best_row['f1_score']:.4f}"
)

print(
    f"ROC-AUC  : {best_row['roc_auc']:.4f}"
)


# ============================================================
# 9. SAVE MODELS AND RESULTS
# ============================================================

print("\n[9/10] Saving models and evaluation results...")


# Save every trained model

for name, model in trained_models.items():

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    model_path = (
        MODEL_DIR /
        f"{safe_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Saved {name}:"
        f"\n{model_path}"
    )


# Save best model

best_model_path = (
    MODEL_DIR /
    "best_crop_suitability_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)

print(
    f"\nBest model saved to:"
    f"\n{best_model_path}"
)


# Save comparison table

results_file = (
    RESULTS_DIR /
    "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    f"\nModel comparison saved to:"
    f"\n{results_file}"
)


# Save metadata

metadata = {

    "best_model": best_model_name,

    "number_of_features": int(
        X_train.shape[1]
    ),

    "training_samples": int(
        len(X_train)
    ),

    "testing_samples": int(
        len(X_test)
    ),

    "target_classes": [
        int(x)
        for x in sorted(
            y_train.unique()
        )
    ],

    "metrics": {
        "accuracy": float(
            best_row["accuracy"]
        ),
        "precision": float(
            best_row["precision"]
        ),
        "recall": float(
            best_row["recall"]
        ),
        "f1_score": float(
            best_row["f1_score"]
        ),
        "roc_auc": float(
            best_row["roc_auc"]
        )
    },

    "features": list(
        X_train.columns
    )
}


metadata_file = (
    MODEL_DIR /
    "model_metadata.json"
)

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n[10/10] Final training summary")

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
    f"Models trained   : {len(models)}"
)

print(
    f"Best model       : {best_model_name}"
)

print(
    f"Best F1-score    : "
    f"{best_row['f1_score']:.4f}"
)

print(
    f"Best ROC-AUC     : "
    f"{best_row['roc_auc']:.4f}"
)

print(
    f"\nModel directory:"
    f"\n{MODEL_DIR}"
)

print(
    f"\nResults directory:"
    f"\n{RESULTS_DIR}"
)

print("\n" + "=" * 70)

print("MODEL TRAINING COMPLETED SUCCESSFULLY")

print("=" * 70)