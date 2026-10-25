"""
Train and compare models for the North-Central Nigeria 7-crop suitability project.

Run from the project root:
    python scripts/08_train_models_7_crops.py

Important:
The target labels in the current dataset are rule-derived from temperature,
rainfall and soil pH thresholds. Metrics therefore measure how well a model
reproduces those rules; they are not independent proof of real-world yield.
"""

from pathlib import Path
import json
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "north_central_training_dataset_7_crops.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "model_splits_7_crops"
RESULTS_DIR = ROOT / "data" / "processed" / "model_results"
MODELS_DIR = ROOT / "models"
MAPPING_FILE = ROOT / "data" / "processed" / "crop_label_mapping.csv"

EXPECTED_CROPS = [
    "Bambara_groundnut", "Cassava", "Cowpea", "Groundnut",
    "Rice", "Soybean", "Yam"
]

# These names must match the recommendation engine's environmental inputs.
ENVIRONMENT_FEATURES = [
    "latitude", "longitude",
    "mean_daily_precipitation_mm", "std_daily_precipitation_mm",
    "min_daily_precipitation_mm", "max_daily_precipitation_mm",
    "total_precipitation_mm", "mean_annual_precipitation_mm",
    "mean_temperature_c", "std_temperature_c", "min_temperature_c",
    "max_temperature_c", "mean_max_temperature_c", "absolute_max_temperature_c",
    "mean_min_temperature_c", "absolute_min_temperature_c",
    "mean_relative_humidity_percent", "minimum_relative_humidity_percent",
    "maximum_relative_humidity_percent",
    "mean_solar_radiation", "minimum_solar_radiation", "maximum_solar_radiation",
    "mean_wind_speed", "minimum_wind_speed", "maximum_wind_speed",
    "soil_ph", "soil_organic_carbon_g_kg", "soil_nitrogen_g_kg",
    "clay_percent", "sand_percent", "silt_percent",
    "cec_cmolc_kg", "bulk_density_g_cm3"
]
FEATURES = ENVIRONMENT_FEATURES + ["crop_encoded"]
TARGET = "suitable"


def norm_crop(value):
    return re.sub(r"[\s-]+", "_", str(value).strip()).lower()


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def main():
    if not DATA_FILE.exists():
        sys.exit(
            f"\nERROR: Dataset not found:\n{DATA_FILE}\n"
            "Run scripts/07_create_training_dataset_7_crops_corrected.py first."
        )

    df = pd.read_csv(DATA_FILE)
    print("=" * 72)
    print("SEVEN-CROP MODEL TRAINING")
    print("=" * 72)
    print(f"Dataset: {DATA_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    crop_col = find_column(df, ["crop", "crop_name"])
    if crop_col is None:
        sys.exit("ERROR: No 'crop' column found in the training dataset.")

    df["_crop_norm"] = df[crop_col].map(norm_crop)
    expected_norm = [norm_crop(c) for c in EXPECTED_CROPS]
    found_norm = set(df["_crop_norm"].dropna().unique())
    missing_crops = [c for c in EXPECTED_CROPS if norm_crop(c) not in found_norm]
    extra_crops = sorted(found_norm - set(expected_norm))
    if missing_crops or extra_crops:
        sys.exit(
            "ERROR: Dataset crop labels do not match the required seven crops.\n"
            f"Missing: {missing_crops}\nUnexpected: {extra_crops}\n"
            f"Found labels: {sorted(df[crop_col].astype(str).unique())}"
        )

    if TARGET not in df.columns:
        sys.exit(f"ERROR: Target column '{TARGET}' is missing.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Assign stable crop codes in a fixed order, independent of alphabetical sorting.
    crop_to_code = {norm_crop(crop): i for i, crop in enumerate(EXPECTED_CROPS)}
    df["crop_encoded"] = df["_crop_norm"].map(crop_to_code).astype(int)

    missing_features = [c for c in ENVIRONMENT_FEATURES if c not in df.columns]
    if missing_features:
        sys.exit(
            "ERROR: The dataset is missing expected feature columns:\n  - "
            + "\n  - ".join(missing_features)
            + "\nCheck the column names in north_central_training_dataset_7_crops.csv "
              "and align ENVIRONMENT_FEATURES in this script and the recommender."
        )

    # Exclude labels and rule-derived columns to prevent target leakage.
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET]
    if y.nunique() != 2:
        sys.exit(f"ERROR: Expected both target classes 0 and 1; found {sorted(y.unique())}.")
    if X.isna().all(axis=0).any():
        bad = X.columns[X.isna().all(axis=0)].tolist()
        sys.exit(f"ERROR: These model features are entirely empty: {bad}")

    # Group by location so the same location does not appear in both train and test.
    location_col = find_column(df, ["location_id", "location", "location_name", "site_id", "site"])
    if location_col is not None and df[location_col].nunique() >= 5:
        groups = df[location_col].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        split_method = f"GroupShuffleSplit by {location_col}"
    else:
        print("WARNING: No usable location identifier found; using stratified row split.")
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.20, random_state=42, stratify=y
        )
        split_method = "stratified row split (location grouping unavailable)"

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        sys.exit(
            "ERROR: The split does not contain both classes in train and test. "
            "Check the location labels or adjust the split."
        )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the exact feature matrices and labels used in this run.
    X_train.assign(suitable=y_train.values).to_csv(SPLIT_DIR / "train_7_crops.csv", index=False)
    X_test.assign(suitable=y_test.values).to_csv(SPLIT_DIR / "test_7_crops.csv", index=False)

    # Stable crop mapping used by the recommender.
    pd.DataFrame({
        "crop": EXPECTED_CROPS,
        "crop_encoded": list(range(len(EXPECTED_CROPS)))
    }).to_csv(MAPPING_FILE, index=False)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42))
        ]),
        "Support Vector Machine": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, class_weight="balanced", random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    rows = []
    fitted = {}
    print(f"\nSplit method: {split_method}")
    print(f"Training records: {len(X_train)} | Testing records: {len(X_test)}")
    print(f"Training class counts: {y_train.value_counts().sort_index().to_dict()}")
    print(f"Testing class counts: {y_test.value_counts().sort_index().to_dict()}")
    print("\nTraining and evaluating models...\n")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1_score": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob) if prob is not None and y_test.nunique() == 2 else np.nan,
            }
            rows.append(metrics)
            fitted[name] = model
            print(
                f"{name:24s} Accuracy={metrics['accuracy']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  F1={metrics['f1_score']:.4f}  "
                f"ROC-AUC={metrics['roc_auc']:.4f}"
            )
        except Exception as exc:
            print(f"FAILED: {name}: {exc}")

    if not rows:
        sys.exit("ERROR: All models failed to train. See errors above.")

    results = pd.DataFrame(rows).sort_values(
        ["f1_score", "accuracy"], ascending=False
    ).reset_index(drop=True)
    results.to_csv(RESULTS_DIR / "model_comparison_7_crops.csv", index=False)

    best_name = str(results.iloc[0]["model"])
    best_model = fitted[best_name]
    best_path = MODELS_DIR / "best_crop_suitability_model.joblib"
    joblib.dump(best_model, best_path)

    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    report = classification_report(
        y_test, best_pred, labels=[0, 1],
        target_names=["Not suitable", "Suitable"], zero_division=0
    )

    metadata = {
        "project": "North-Central Nigeria 7-crop suitability classification",
        "best_model": best_name,
        "model_file": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "dataset_file": str(DATA_FILE.relative_to(ROOT)).replace("\\", "/"),
        "target": TARGET,
        "features": FEATURES,
        "environment_features": ENVIRONMENT_FEATURES,
        "crop_order": EXPECTED_CROPS,
        "crop_mapping_file": str(MAPPING_FILE.relative_to(ROOT)).replace("\\", "/"),
        "split_method": split_method,
        "random_state": 42,
        "test_size": 0.20,
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "metrics": {
            key: (float(results.iloc[0][key]) if pd.notna(results.iloc[0][key]) else None)
            for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        },
        "confusion_matrix_labels_0_1": cm.tolist(),
        "label_caveat": (
            "Target labels are rule-derived from crop temperature, rainfall and pH "
            "requirements. Evaluation measures replication of those rules and does "
            "not independently validate real-world yield or agronomic performance."
        )
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(RESULTS_DIR / "best_model_test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Split method: {split_method}\n")
        f.write(f"Training records: {len(X_train)}\nTesting records: {len(X_test)}\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\nConfusion matrix (rows=true [0,1], columns=predicted [0,1]):\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write("\n\nIMPORTANT LIMITATION:\n")
        f.write(metadata["label_caveat"] + "\n")

    print("\n" + "=" * 72)
    print(f"BEST MODEL: {best_name}")
    print(f"Saved model: {best_path}")
    print(f"Saved mapping: {MAPPING_FILE}")
    print(f"Saved comparison: {RESULTS_DIR / 'model_comparison_7_crops.csv'}")
    print(f"Saved test report: {RESULTS_DIR / 'best_model_test_report.txt'}")
    print("\nConfusion matrix (true rows 0/1, predicted columns 0/1):")
    print(cm)
    print("\nClassification report:")
    print(report)
    print("\nIMPORTANT: Labels are rule-derived; do not describe these metrics as field-yield validation.")
    print("\nNEXT: Update the recommendation engine to load best_crop_suitability_model.joblib")
    print("and use the feature order recorded in models/model_metadata.json.")


if __name__ == "__main__":
    main()
