"""
09_analyze_recommendation_bias.py

Analyzes possible bias, dominance, and data leakage in the
crop recommendation engine.

Checks:
1. Crop-level recommendation distribution
2. Top-1 recommendation dominance
3. Top-3 recommendation diversity
4. Mean suitability probability by crop
5. Environmental suitability by crop
6. Model feature importance
7. Importance of crop_encoded
8. Correlation between crop identity and prediction
9. LGA-level recommendation variation
10. Alternative model without crop_encoded
11. Comparison between original and reduced-feature models
12. Saves diagnostic reports and plots
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RECOMMENDATION_FILE = os.path.join(
    DATA_DIR,
    "recommendations",
    "crop_recommendations.csv"
)

TRAINING_FILE = os.path.join(
    DATA_DIR,
    "north_central_training_dataset.csv"
)

X_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "X_train.csv"
)

X_TEST_FILE = os.path.join(
    DATA_DIR,
    "X_test.csv"
)

Y_TRAIN_FILE = os.path.join(
    DATA_DIR,
    "y_train.csv"
)

Y_TEST_FILE = os.path.join(
    DATA_DIR,
    "y_test.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "final_crop_suitability_model.joblib"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "model_results",
    "recommendation_bias"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def print_section(number, title):
    print()
    print("=" * 70)
    print(f"[{number}] {title}")
    print("=" * 70)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("ANALYZING CROP RECOMMENDATION BIAS")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print_section(1, "Checking input files...")

required_files = {
    "Recommendation dataset": RECOMMENDATION_FILE,
    "Training dataset": TRAINING_FILE,
    "X_train": X_TRAIN_FILE,
    "X_test": X_TEST_FILE,
    "y_train": Y_TRAIN_FILE,
    "y_test": Y_TEST_FILE,
    "Final model": MODEL_FILE,
}

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name} found:")
    print(path)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print_section(2, "Loading datasets...")

recommendations = pd.read_csv(RECOMMENDATION_FILE)
training = pd.read_csv(TRAINING_FILE)

X_train = pd.read_csv(X_TRAIN_FILE)
X_test = pd.read_csv(X_TEST_FILE)

y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()
y_test = pd.read_csv(Y_TEST_FILE).squeeze()

model = joblib.load(MODEL_FILE)

print(f"Recommendation records: {len(recommendations)}")
print(f"Training records:        {len(training)}")
print(f"X_train shape:           {X_train.shape}")
print(f"X_test shape:            {X_test.shape}")
print(f"Model:                   {type(model).__name__}")


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

print_section(3, "Validating recommendation dataset...")

required_recommendation_columns = [
    "location_id",
    "state",
    "lga",
    "crop",
    "suitability_probability",
    "recommendation_rank",
    "recommendation_category"
]

missing_columns = [
    col for col in required_recommendation_columns
    if col not in recommendations.columns
]

if missing_columns:
    raise ValueError(
        f"Missing recommendation columns:\n{missing_columns}"
    )

print("SUCCESS: Recommendation dataset is valid.")

print(f"Unique locations: {recommendations['location_id'].nunique()}")
print(f"Unique crops:     {recommendations['crop'].nunique()}")
print(f"Total records:    {len(recommendations)}")

expected_records = (
    recommendations["location_id"].nunique()
    * recommendations["crop"].nunique()
)

print(f"Expected records: {expected_records}")

if len(recommendations) == expected_records:
    print("SUCCESS: Every location has a record for every crop.")
else:
    print("WARNING: Expected and actual record counts differ.")


# ============================================================
# 4. CROP RECOMMENDATION DISTRIBUTION
# ============================================================

print_section(4, "Analyzing crop recommendation distribution...")

recommended = recommendations[
    recommendations["recommendation_rank"].notna()
].copy()

top1 = recommendations[
    recommendations["recommendation_rank"] == 1
].copy()

top2 = recommendations[
    recommendations["recommendation_rank"] == 2
].copy()

top3 = recommendations[
    recommendations["recommendation_rank"] == 3
].copy()


# ------------------------------------------------------------
# All recommended crops
# ------------------------------------------------------------

recommended_frequency = (
    recommendations[
        recommendations["recommendation_category"] != "Not Recommended"
    ]
    ["crop"]
    .value_counts()
    .rename("recommended_count")
    .reset_index()
)

recommended_frequency.columns = [
    "crop",
    "recommended_count"
]

recommended_frequency["percentage"] = (
    recommended_frequency["recommended_count"]
    / len(recommendations)
    * 100
)

print()
print("Recommended predictions by crop:")
print(recommended_frequency.to_string(index=False))

recommended_frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recommended_crop_frequency.csv"
    ),
    index=False
)


# ============================================================
# 5. TOP-1 DOMINANCE
# ============================================================

print_section(5, "Analyzing top-1 recommendation dominance...")

top1_frequency = (
    top1["crop"]
    .value_counts()
    .rename("top1_count")
    .reset_index()
)

top1_frequency.columns = [
    "crop",
    "top1_count"
]

top1_frequency["top1_percentage"] = (
    top1_frequency["top1_count"]
    / len(top1)
    * 100
)

print()
print("Top-1 recommendation distribution:")
print(top1_frequency.to_string(index=False))

dominant_crop = top1_frequency.iloc[0]["crop"]
dominant_count = top1_frequency.iloc[0]["top1_count"]
dominant_percentage = top1_frequency.iloc[0]["top1_percentage"]

print()
print(f"Dominant top-1 crop: {dominant_crop}")
print(f"Top-1 appearances:   {dominant_count}")
print(f"Top-1 dominance:     {dominant_percentage:.2f}%")

if dominant_percentage > 50:
    print()
    print("WARNING:")
    print(
        "A single crop accounts for more than 50% "
        "of all top-1 recommendations."
    )
else:
    print("No severe top-1 dominance detected.")


top1_frequency.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top1_crop_distribution.csv"
    ),
    index=False
)


# ============================================================
# 6. TOP-1 PLOT
# ============================================================

print_section(6, "Creating top-1 recommendation plot...")

plt.figure(figsize=(10, 6))

plt.bar(
    top1_frequency["crop"],
    top1_frequency["top1_count"]
)

plt.xlabel("Crop")
plt.ylabel("Number of Top-1 Recommendations")
plt.title("Top-1 Crop Recommendation Distribution")

plt.xticks(
    rotation=45,
    ha="right"
)

top1_plot = save_plot(
    "top1_crop_distribution.png"
)

print(f"Saved plot: {top1_plot}")


# ============================================================
# 7. SUITABILITY PROBABILITY BY CROP
# ============================================================

print_section(7, "Analyzing suitability probability by crop...")

probability_by_crop = (
    recommendations
    .groupby("crop")["suitability_probability"]
    .agg(
        [
            "count",
            "mean",
            "std",
            "min",
            "median",
            "max"
        ]
    )
    .reset_index()
)

print()
print(probability_by_crop.to_string(index=False))

probability_by_crop.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "probability_by_crop.csv"
    ),
    index=False
)


# ============================================================
# 8. PROBABILITY PLOT
# ============================================================

print_section(8, "Creating probability comparison plot...")

plot_data = probability_by_crop.sort_values(
    "mean",
    ascending=False
)

plt.figure(figsize=(11, 6))

plt.bar(
    plot_data["crop"],
    plot_data["mean"]
)

plt.xlabel("Crop")
plt.ylabel("Mean Suitability Probability")
plt.title(
    "Mean Model Suitability Probability by Crop"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.ylim(0, 1)

probability_plot = save_plot(
    "mean_probability_by_crop.png"
)

print(f"Saved plot: {probability_plot}")


# ============================================================
# 9. TRAINING DATA CROP-LEVEL ANALYSIS
# ============================================================

print_section(9, "Analyzing training data by crop...")

training_crop_stats = (
    training
    .groupby("crop")["suitable"]
    .agg(
        [
            "count",
            "sum",
            "mean"
        ]
    )
    .reset_index()
)

training_crop_stats.columns = [
    "crop",
    "total_records",
    "suitable_records",
    "suitability_rate"
]

training_crop_stats["suitability_percentage"] = (
    training_crop_stats["suitability_rate"] * 100
)

print()
print(
    training_crop_stats
    .sort_values(
        "suitability_rate",
        ascending=False
    )
    .to_string(index=False)
)

training_crop_stats.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_suitability_by_crop.csv"
    ),
    index=False
)


# ============================================================
# 10. ENVIRONMENTAL SUITABILITY ANALYSIS
# ============================================================

print_section(
    10,
    "Comparing environmental suitability with model predictions..."
)

environmental_columns = [
    "temperature_suitable",
    "rainfall_suitable",
    "ph_suitable",
    "environmental_match_score"
]

available_environmental_columns = [
    col
    for col in environmental_columns
    if col in training.columns
]

print()
print(
    "Environmental suitability columns available:"
)

for col in available_environmental_columns:
    print(f"- {col}")


environmental_by_crop = (
    training
    .groupby("crop")[available_environmental_columns]
    .mean()
    .reset_index()
)

print()
print(environmental_by_crop.to_string(index=False))

environmental_by_crop.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "environmental_suitability_by_crop.csv"
    ),
    index=False
)


# ============================================================
# 11. CROP ENCODING IMPORTANCE
# ============================================================

print_section(
    11,
    "Analyzing crop_encoded feature importance..."
)

if hasattr(model, "feature_importances_"):

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print(feature_importance.to_string(index=False))

    feature_importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_feature_importance.csv"
        ),
        index=False
    )

    crop_importance = feature_importance[
        feature_importance["feature"] == "crop_encoded"
    ]

    if not crop_importance.empty:

        crop_importance_value = (
            crop_importance.iloc[0]["importance"]
        )

        print()
        print(
            f"crop_encoded importance: "
            f"{crop_importance_value:.6f}"
        )

        if crop_importance_value > 0.50:

            print()
            print("WARNING:")
            print(
                "crop_encoded contributes more than "
                "50% of total Decision Tree importance."
            )

        elif crop_importance_value > 0.30:

            print()
            print("CAUTION:")
            print(
                "crop_encoded has relatively high "
                "feature importance."
            )

        else:

            print(
                "crop_encoded does not dominate "
                "feature importance."
            )

else:

    print(
        "Model does not expose feature_importances_."
    )


# ============================================================
# 12. CROP-ENCODED VS ENVIRONMENTAL MODEL
# ============================================================

print_section(
    12,
    "Testing model without crop_encoded..."
)

if "crop_encoded" not in X_train.columns:

    print(
        "crop_encoded is not present. "
        "Reduced-feature analysis skipped."
    )

else:

    environmental_features = [
        col
        for col in X_train.columns
        if col != "crop_encoded"
    ]

    X_train_environmental = X_train[
        environmental_features
    ]

    X_test_environmental = X_test[
        environmental_features
    ]

    reduced_model = DecisionTreeClassifier(
        class_weight="balanced",
        criterion="entropy",
        max_depth=7,
        min_samples_leaf=2,
        min_samples_split=2,
        random_state=42
    )

    reduced_model.fit(
        X_train_environmental,
        y_train
    )

    reduced_predictions = reduced_model.predict(
        X_test_environmental
    )

    reduced_probabilities = (
        reduced_model.predict_proba(
            X_test_environmental
        )[:, 1]
    )

    reduced_accuracy = accuracy_score(
        y_test,
        reduced_predictions
    )

    reduced_precision = precision_score(
        y_test,
        reduced_predictions,
        zero_division=0
    )

    reduced_recall = recall_score(
        y_test,
        reduced_predictions,
        zero_division=0
    )

    reduced_f1 = f1_score(
        y_test,
        reduced_predictions,
        zero_division=0
    )

    reduced_auc = roc_auc_score(
        y_test,
        reduced_probabilities
    )

    print()
    print("Environmental-only Decision Tree:")
    print(
        f"Accuracy : {reduced_accuracy:.4f}"
    )
    print(
        f"Precision: {reduced_precision:.4f}"
    )
    print(
        f"Recall   : {reduced_recall:.4f}"
    )
    print(
        f"F1-score : {reduced_f1:.4f}"
    )
    print(
        f"ROC-AUC  : {reduced_auc:.4f}"
    )

    original_predictions = model.predict(X_test)

    original_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    original_accuracy = accuracy_score(
        y_test,
        original_predictions
    )

    original_precision = precision_score(
        y_test,
        original_predictions,
        zero_division=0
    )

    original_recall = recall_score(
        y_test,
        original_predictions,
        zero_division=0
    )

    original_f1 = f1_score(
        y_test,
        original_predictions,
        zero_division=0
    )

    original_auc = roc_auc_score(
        y_test,
        original_probabilities
    )

    comparison = pd.DataFrame([
        {
            "model": "Original model",
            "features": X_train.shape[1],
            "accuracy": original_accuracy,
            "precision": original_precision,
            "recall": original_recall,
            "f1_score": original_f1,
            "roc_auc": original_auc
        },
        {
            "model": "Environmental-only model",
            "features": X_train_environmental.shape[1],
            "accuracy": reduced_accuracy,
            "precision": reduced_precision,
            "recall": reduced_recall,
            "f1_score": reduced_f1,
            "roc_auc": reduced_auc
        }
    ])

    print()
    print("Model comparison:")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "crop_encoded_model_comparison.csv"
        ),
        index=False
    )


# ============================================================
# 13. CROSS-VALIDATION ENVIRONMENTAL-ONLY MODEL
# ============================================================

print_section(
    13,
    "Cross-validating environmental-only model..."
)

if "crop_encoded" in X_train.columns:

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_model = DecisionTreeClassifier(
        class_weight="balanced",
        criterion="entropy",
        max_depth=7,
        min_samples_leaf=2,
        min_samples_split=2,
        random_state=42
    )

    cv_f1 = cross_val_score(
        cv_model,
        X_train_environmental,
        y_train,
        cv=cv,
        scoring="f1"
    )

    cv_auc = cross_val_score(
        cv_model,
        X_train_environmental,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    print()
    print(
        f"Environmental-only CV F1: "
        f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}"
    )

    print(
        f"Environmental-only CV ROC-AUC: "
        f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f}"
    )


# ============================================================
# 14. LGA RECOMMENDATION DIVERSITY
# ============================================================

print_section(
    14,
    "Analyzing LGA recommendation diversity..."
)

top3 = (
    recommendations[
        recommendations["recommendation_rank"] <= 3
    ]
    .sort_values(
        [
            "location_id",
            "recommendation_rank"
        ]
    )
)

top3_combinations = (
    top3
    .groupby(
        ["location_id", "state", "lga"]
    )["crop"]
    .apply(
        lambda x: " | ".join(x)
    )
    .reset_index(name="top_3_crops")
)

combination_counts = (
    top3_combinations["top_3_crops"]
    .value_counts()
    .rename("locations")
    .reset_index()
)

combination_counts.columns = [
    "top_3_crops",
    "locations"
]

print()
print("Top-3 crop combinations:")
print(
    combination_counts.to_string(
        index=False
    )
)

unique_combinations = (
    top3_combinations["top_3_crops"]
    .nunique()
)

total_locations = (
    top3_combinations["location_id"]
    .nunique()
)

print()
print(
    f"Unique top-3 combinations: "
    f"{unique_combinations}"
)

print(
    f"Total locations: "
    f"{total_locations}"
)

diversity_percentage = (
    unique_combinations
    / total_locations
    * 100
)

print(
    f"Combination diversity: "
    f"{diversity_percentage:.2f}%"
)

if unique_combinations <= 5:

    print()
    print("WARNING:")
    print(
        "Top-3 recommendations show low variation "
        "between locations."
    )

elif diversity_percentage < 20:

    print()
    print("CAUTION:")
    print(
        "Recommendation diversity is relatively low."
    )

else:

    print(
        "Recommendation diversity appears reasonable."
    )


top3_combinations.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lga_top3_combinations.csv"
    ),
    index=False
)

combination_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top3_combination_frequency.csv"
    ),
    index=False
)


# ============================================================
# 15. LOCATION-LEVEL RECOMMENDATION VARIATION
# ============================================================

print_section(
    15,
    "Analyzing location-level variation..."
)

location_statistics = (
    recommendations
    .groupby(
        ["location_id", "state", "lga"]
    )
    .agg(
        mean_probability=(
            "suitability_probability",
            "mean"
        ),
        max_probability=(
            "suitability_probability",
            "max"
        ),
        min_probability=(
            "suitability_probability",
            "min"
        ),
        recommended_crops=(
            "recommendation_category",
            lambda x: (
                x != "Not Recommended"
            ).sum()
        )
    )
    .reset_index()
)

print()
print(
    location_statistics.head(15)
    .to_string(index=False)
)

location_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "location_level_statistics.csv"
    ),
    index=False
)


# ============================================================
# 16. MODEL PROBABILITY VARIATION BY CROP
# ============================================================

print_section(
    16,
    "Checking probability variation..."
)

probability_variation = (
    recommendations
    .groupby("crop")[
        "suitability_probability"
    ]
    .agg(
        mean="mean",
        std="std",
        minimum="min",
        maximum="max"
    )
    .reset_index()
)

probability_variation["range"] = (
    probability_variation["maximum"]
    -
    probability_variation["minimum"]
)

print()
print(
    probability_variation
    .sort_values(
        "mean",
        ascending=False
    )
    .to_string(index=False)
)

probability_variation.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "probability_variation_by_crop.csv"
    ),
    index=False
)


# ============================================================
# 17. OVERALL BIAS DIAGNOSIS
# ============================================================

print_section(
    17,
    "Generating overall bias diagnosis..."
)

diagnoses = []

# Top-1 dominance
if dominant_percentage > 50:
    diagnoses.append(
        "HIGH TOP-1 DOMINANCE: "
        f"{dominant_crop} accounts for "
        f"{dominant_percentage:.2f}% of top-1 recommendations."
    )
else:
    diagnoses.append(
        "Top-1 dominance is below 50%."
    )


# Top-3 diversity
if unique_combinations <= 5:
    diagnoses.append(
        "LOW TOP-3 DIVERSITY: "
        "Few unique crop combinations occur across locations."
    )
else:
    diagnoses.append(
        "Top-3 recommendation diversity is acceptable."
    )


# Crop encoding
if "crop_encoded" in feature_importance.columns:

    crop_imp = feature_importance[
        feature_importance["feature"]
        == "crop_encoded"
    ]["importance"]

    if not crop_imp.empty:

        crop_imp_value = crop_imp.iloc[0]

        if crop_imp_value > 0.50:
            diagnoses.append(
                "HIGH CROP-ENCODING IMPORTANCE: "
                "crop_encoded contributes more than 50% "
                "of tree importance."
            )

        elif crop_imp_value > 0.30:
            diagnoses.append(
                "MODERATE/HIGH CROP-ENCODING IMPORTANCE."
            )

        else:
            diagnoses.append(
                "crop_encoded does not strongly dominate "
                "feature importance."
            )


# Probability concentration
high_probability_count = (
    recommendations[
        recommendations[
            "suitability_probability"
        ] >= 0.81
    ]
    .shape[0]
)

high_probability_percentage = (
    high_probability_count
    / len(recommendations)
    * 100
)

if high_probability_percentage > 50:

    diagnoses.append(
        "HIGH PROBABILITY CONCENTRATION: "
        f"{high_probability_percentage:.2f}% "
        "of predictions have probability >= 0.81."
    )

else:

    diagnoses.append(
        "Probability distribution is not excessively "
        "concentrated above 0.81."
    )


print()

for i, diagnosis in enumerate(
    diagnoses,
    start=1
):

    print(
        f"{i}. {diagnosis}"
    )


# ============================================================
# 18. SAVE DIAGNOSIS REPORT
# ============================================================

print_section(
    18,
    "Saving bias analysis report..."
)

report_path = os.path.join(
    OUTPUT_DIR,
    "recommendation_bias_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "CROP RECOMMENDATION BIAS ANALYSIS REPORT\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        "DATASET SUMMARY\n"
    )

    report.write(
        f"Locations: {recommendations['location_id'].nunique()}\n"
    )

    report.write(
        f"Crops: {recommendations['crop'].nunique()}\n"
    )

    report.write(
        f"Predictions: {len(recommendations)}\n\n"
    )

    report.write(
        "TOP-1 DOMINANCE\n"
    )

    report.write(
        f"Dominant crop: {dominant_crop}\n"
    )

    report.write(
        f"Top-1 count: {dominant_count}\n"
    )

    report.write(
        f"Top-1 dominance: {dominant_percentage:.2f}%\n\n"
    )

    report.write(
        "TOP-3 DIVERSITY\n"
    )

    report.write(
        f"Unique top-3 combinations: "
        f"{unique_combinations}\n"
    )

    report.write(
        f"Total locations: {total_locations}\n"
    )

    report.write(
        f"Diversity percentage: "
        f"{diversity_percentage:.2f}%\n\n"
    )

    report.write(
        "HIGH-PROBABILITY PREDICTIONS\n"
    )

    report.write(
        f"Count >= 0.81: "
        f"{high_probability_count}\n"
    )

    report.write(
        f"Percentage >= 0.81: "
        f"{high_probability_percentage:.2f}%\n\n"
    )

    report.write(
        "DIAGNOSIS\n"
    )

    for i, diagnosis in enumerate(
        diagnoses,
        start=1
    ):

        report.write(
            f"{i}. {diagnosis}\n"
        )

print(
    f"Report saved to:\n{report_path}"
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print_section(
    19,
    "FINAL BIAS ANALYSIS SUMMARY"
)

print(
    f"Locations analyzed       : "
    f"{recommendations['location_id'].nunique()}"
)

print(
    f"Crops analyzed           : "
    f"{recommendations['crop'].nunique()}"
)

print(
    f"Total predictions        : "
    f"{len(recommendations)}"
)

print(
    f"Dominant top-1 crop      : "
    f"{dominant_crop}"
)

print(
    f"Top-1 dominance          : "
    f"{dominant_percentage:.2f}%"
)

print(
    f"Unique top-3 combinations: "
    f"{unique_combinations}"
)

print(
    f"High probability >=81%   : "
    f"{high_probability_percentage:.2f}%"
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("=" * 70)
print("RECOMMENDATION BIAS ANALYSIS COMPLETED")
print("=" * 70)