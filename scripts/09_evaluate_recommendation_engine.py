"""
09_evaluate_recommendation_engine.py

Evaluate the crop recommendation engine.

This script analyzes:
    1. Full crop recommendation predictions
    2. Top-3 crop recommendations per LGA
    3. Recommendation category distribution
    4. Crop recommendation frequency
    5. Suitability probability statistics
    6. Top-1, Top-2 and Top-3 crop coverage
    7. Crop dominance
    8. LGA recommendation consistency
    9. Confidence distribution
   10. Duplicate recommendation patterns

Outputs are saved under:

data/processed/model_results/recommendation_evaluation/
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FULL_RECOMMENDATIONS = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recommendations",
    "crop_recommendations.csv"
)

TOP_RECOMMENDATIONS = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "recommendations",
    "top_crop_recommendations.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "model_results",
    "recommendation_evaluation"
)


# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def save_plot(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EVALUATING CROP RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# [1/12] CHECK INPUT FILES
# ============================================================

print_section("[1/12] Checking input files...")

if not os.path.exists(FULL_RECOMMENDATIONS):
    raise FileNotFoundError(
        f"Full recommendation dataset not found:\n{FULL_RECOMMENDATIONS}"
    )

print(f"Full recommendation dataset found:")
print(FULL_RECOMMENDATIONS)


if not os.path.exists(TOP_RECOMMENDATIONS):
    raise FileNotFoundError(
        f"Top recommendation dataset not found:\n{TOP_RECOMMENDATIONS}"
    )

print()
print("Top recommendation dataset found:")
print(TOP_RECOMMENDATIONS)


# ============================================================
# [2/12] LOAD DATASETS
# ============================================================

print_section("[2/12] Loading recommendation datasets...")

full_df = pd.read_csv(FULL_RECOMMENDATIONS)
top_df = pd.read_csv(TOP_RECOMMENDATIONS)

print(f"Full recommendation records: {len(full_df)}")
print(f"Top recommendation records:  {len(top_df)}")


# ============================================================
# [3/12] VALIDATE DATASETS
# ============================================================

print_section("[3/12] Validating recommendation datasets...")

required_full_columns = [
    "location_id",
    "state",
    "lga",
    "crop",
    "suitability_probability",
    "recommendation_rank",
    "recommendation_category"
]

required_top_columns = [
    "location_id",
    "state",
    "lga",
    "crop",
    "suitability_probability",
    "recommendation_rank",
    "recommendation_category"
]


missing_full = [
    col for col in required_full_columns
    if col not in full_df.columns
]

missing_top = [
    col for col in required_top_columns
    if col not in top_df.columns
]


if missing_full:
    raise ValueError(
        f"Missing columns in full recommendation dataset:\n{missing_full}"
    )

if missing_top:
    raise ValueError(
        f"Missing columns in top recommendation dataset:\n{missing_top}"
    )


print("SUCCESS: Full recommendation dataset is valid.")
print("SUCCESS: Top recommendation dataset is valid.")


# ============================================================
# [4/12] BASIC DATASET INFORMATION
# ============================================================

print_section("[4/12] Analyzing dataset structure...")

unique_locations = full_df["location_id"].nunique()
unique_crops = full_df["crop"].nunique()

print(f"Unique locations: {unique_locations}")
print(f"Unique crops:     {unique_crops}")
print(f"Total predictions:{len(full_df)}")


expected_predictions = unique_locations * unique_crops

print(f"Expected predictions: {expected_predictions}")
print(f"Actual predictions:   {len(full_df)}")


if len(full_df) == expected_predictions:
    print("SUCCESS: Every location has predictions for every crop.")
else:
    print("WARNING: Prediction count does not match expected combinations.")


# ============================================================
# [5/12] CHECK MISSING AND DUPLICATE VALUES
# ============================================================

print_section("[5/12] Checking data quality...")

missing_values = full_df.isnull().sum().sum()
duplicate_rows = full_df.duplicated().sum()

print(f"Total missing values: {missing_values}")
print(f"Duplicate rows:      {duplicate_rows}")


if missing_values == 0:
    print("SUCCESS: No missing values detected.")
else:
    print("WARNING: Missing values detected.")


if duplicate_rows == 0:
    print("SUCCESS: No duplicate rows detected.")
else:
    print("WARNING: Duplicate rows detected.")


# ============================================================
# [6/12] RECOMMENDATION CATEGORY ANALYSIS
# ============================================================

print_section("[6/12] Analyzing recommendation categories...")

category_counts = (
    full_df["recommendation_category"]
    .value_counts()
    .sort_values(ascending=False)
)

category_percentages = (
    full_df["recommendation_category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

category_table = pd.DataFrame({
    "count": category_counts,
    "percentage": category_percentages
})

print()
print(category_table)


category_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "recommendation_category_distribution.csv"
    )
)

print()
print(
    "Saved category analysis to:"
    f"\n{os.path.join(OUTPUT_DIR, 'recommendation_category_distribution.csv')}"
)


# Plot category distribution

plt.figure(figsize=(10, 6))

category_counts.plot(kind="bar")

plt.title("Crop Recommendation Category Distribution")
plt.xlabel("Recommendation Category")
plt.ylabel("Number of Predictions")
plt.xticks(rotation=30, ha="right")

save_plot("recommendation_category_distribution.png")


# ============================================================
# [7/12] CROP RECOMMENDATION FREQUENCY
# ============================================================

print_section("[7/12] Analyzing crop recommendation frequency...")

crop_counts = (
    full_df["crop"]
    .value_counts()
    .sort_values(ascending=False)
)

print()
print("Total predictions by crop:")
print(crop_counts)


# Count suitable recommendations

suitable_df = full_df[
    full_df["recommendation_category"] != "Not Recommended"
].copy()

suitable_crop_counts = (
    suitable_df["crop"]
    .value_counts()
    .sort_values(ascending=False)
)

print()
print("Suitable/recommended predictions by crop:")
print(suitable_crop_counts)


crop_analysis = pd.DataFrame({
    "total_predictions": crop_counts,
    "recommended_predictions": suitable_crop_counts
}).fillna(0)

crop_analysis["recommendation_percentage"] = (
    crop_analysis["recommended_predictions"]
    / crop_analysis["total_predictions"]
    * 100
).round(2)

crop_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "crop_recommendation_frequency.csv"
    )
)


print()
print(
    "Saved crop frequency analysis to:"
    f"\n{os.path.join(OUTPUT_DIR, 'crop_recommendation_frequency.csv')}"
)


# Plot crop recommendation frequency

plt.figure(figsize=(11, 6))

crop_analysis["recommended_predictions"].sort_values(
    ascending=False
).plot(kind="bar")

plt.title("Recommended Predictions by Crop")
plt.xlabel("Crop")
plt.ylabel("Number of Recommended Predictions")
plt.xticks(rotation=45, ha="right")

save_plot("crop_recommendation_frequency.png")


# ============================================================
# [8/12] SUITABILITY PROBABILITY ANALYSIS
# ============================================================

print_section("[8/12] Analyzing suitability probabilities...")

probability = full_df["suitability_probability"]

print()
print("Suitability probability statistics:")

print(probability.describe())


print()
print("Probability distribution:")

probability_ranges = pd.cut(
    probability,
    bins=[
        -0.001,
        0.20,
        0.40,
        0.60,
        0.80,
        1.00
    ],
    labels=[
        "0.00 - 0.20",
        "0.21 - 0.40",
        "0.41 - 0.60",
        "0.61 - 0.80",
        "0.81 - 1.00"
    ],
    include_lowest=True
)

probability_distribution = (
    probability_ranges
    .value_counts()
    .sort_index()
)

print(probability_distribution)


probability_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "probability_distribution.csv"
    )
)


# Probability histogram

plt.figure(figsize=(10, 6))

plt.hist(
    probability,
    bins=20,
    edgecolor="black"
)

plt.title("Suitability Probability Distribution")
plt.xlabel("Suitability Probability")
plt.ylabel("Number of Predictions")

save_plot("suitability_probability_distribution.png")


# ============================================================
# [9/12] TOP 1, TOP 2 AND TOP 3 ANALYSIS
# ============================================================

print_section("[9/12] Analyzing top crop recommendations...")

top1 = top_df[top_df["recommendation_rank"] == 1].copy()
top2 = top_df[top_df["recommendation_rank"] == 2].copy()
top3 = top_df[top_df["recommendation_rank"] == 3].copy()


print(f"Top-1 recommendations: {len(top1)}")
print(f"Top-2 recommendations: {len(top2)}")
print(f"Top-3 recommendations: {len(top3)}")


print()
print("Top-1 crop distribution:")

top1_distribution = (
    top1["crop"]
    .value_counts()
    .sort_values(ascending=False)
)

print(top1_distribution)


print()
print("Top-2 crop distribution:")

top2_distribution = (
    top2["crop"]
    .value_counts()
    .sort_values(ascending=False)
)

print(top2_distribution)


print()
print("Top-3 crop distribution:")

top3_distribution = (
    top3["crop"]
    .value_counts()
    .sort_values(ascending=False)
)

print(top3_distribution)


top_analysis = pd.DataFrame({
    "top_1": top1_distribution,
    "top_2": top2_distribution,
    "top_3": top3_distribution
}).fillna(0)

top_analysis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top_1_2_3_crop_distribution.csv"
    )
)


# ============================================================
# [10/12] CHECK CROP DOMINANCE
# ============================================================

print_section("[10/12] Checking crop dominance...")

if len(top1) > 0:

    top_crop = top1["crop"].value_counts().idxmax()
    top_crop_count = top1["crop"].value_counts().max()

    dominance_percentage = (
        top_crop_count / len(top1)
    ) * 100

    print(f"Most frequently recommended top-1 crop: {top_crop}")
    print(f"Top-1 appearances: {top_crop_count}")
    print(f"Top-1 dominance: {dominance_percentage:.2f}%")

    if dominance_percentage > 50:
        print()
        print("WARNING:")
        print(
            "One crop accounts for more than 50% "
            "of all top-1 recommendations."
        )

        print(
            "This should be investigated for possible "
            "model or data bias."
        )

    else:
        print()
        print(
            "SUCCESS: No single crop dominates more "
            "than 50% of top-1 recommendations."
        )


# ============================================================
# [11/12] LGA-LEVEL CONSISTENCY ANALYSIS
# ============================================================

print_section("[11/12] Analyzing LGA-level recommendations...")

# Number of unique top-3 crop combinations

top_df_sorted = top_df.sort_values(
    ["location_id", "recommendation_rank"]
)

lga_top3 = (
    top_df_sorted
    .groupby(["location_id", "state", "lga"])["crop"]
    .apply(lambda x: " | ".join(x))
    .reset_index()
)

lga_top3.columns = [
    "location_id",
    "state",
    "lga",
    "top_3_crops"
]


print()
print("First LGA recommendation combinations:")
print(lga_top3.head(15).to_string(index=False))


lga_top3.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lga_top_3_recommendations.csv"
    ),
    index=False
)


unique_combinations = lga_top3["top_3_crops"].nunique()

print()
print(f"Unique top-3 crop combinations: {unique_combinations}")
print(f"Total locations: {len(lga_top3)}")


if unique_combinations == 1:

    print()
    print("WARNING:")
    print(
        "Every location has exactly the same top-3 "
        "crop recommendation combination."
    )

elif unique_combinations < len(lga_top3) * 0.25:

    print()
    print("WARNING:")
    print(
        "Top-3 recommendations show relatively low "
        "variation between locations."
    )

else:

    print()
    print(
        "SUCCESS: Recommendations vary across locations."
    )


# ============================================================
# [12/12] FINAL EVALUATION SUMMARY
# ============================================================

print_section("[12/12] Final recommendation engine evaluation")


# Average probability

mean_probability = probability.mean()

# High confidence predictions

high_confidence = (
    probability >= 0.90
).sum()

high_confidence_percentage = (
    high_confidence / len(full_df)
) * 100


# Recommended predictions

recommended_count = len(suitable_df)

recommended_percentage = (
    recommended_count / len(full_df)
) * 100


summary = {
    "total_locations": unique_locations,
    "total_crops": unique_crops,
    "total_predictions": len(full_df),
    "recommended_predictions": recommended_count,
    "recommended_percentage": round(
        recommended_percentage,
        2
    ),
    "not_recommended_predictions": (
        len(full_df) - recommended_count
    ),
    "mean_suitability_probability": round(
        mean_probability,
        4
    ),
    "high_confidence_predictions": high_confidence,
    "high_confidence_percentage": round(
        high_confidence_percentage,
        2
    ),
    "unique_top3_combinations": unique_combinations,
    "top1_dominant_crop": (
        top_crop if len(top1) > 0 else "N/A"
    ),
    "top1_dominance_percentage": round(
        dominance_percentage,
        2
    ) if len(top1) > 0 else 0
}


summary_df = pd.DataFrame(
    [summary]
)

print()
print(summary_df.to_string(index=False))


summary_path = os.path.join(
    OUTPUT_DIR,
    "recommendation_engine_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


print()
print(f"Summary saved to:")
print(summary_path)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "recommendation_engine_evaluation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:

    file.write(
        "CROP RECOMMENDATION ENGINE EVALUATION REPORT\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Total locations: {unique_locations}\n"
    )

    file.write(
        f"Total crops: {unique_crops}\n"
    )

    file.write(
        f"Total predictions: {len(full_df)}\n"
    )

    file.write(
        f"Recommended predictions: {recommended_count}\n"
    )

    file.write(
        f"Recommended percentage: "
        f"{recommended_percentage:.2f}%\n"
    )

    file.write(
        f"Mean suitability probability: "
        f"{mean_probability:.4f}\n"
    )

    file.write(
        f"High-confidence predictions: "
        f"{high_confidence}\n"
    )

    file.write(
        f"High-confidence percentage: "
        f"{high_confidence_percentage:.2f}%\n"
    )

    file.write(
        f"Unique top-3 combinations: "
        f"{unique_combinations}\n"
    )

    file.write("\n")

    file.write("TOP-1 CROP DISTRIBUTION\n")
    file.write("-" * 40 + "\n")
    file.write(
        top1_distribution.to_string()
    )

    file.write("\n\n")

    file.write("RECOMMENDATION CATEGORY DISTRIBUTION\n")
    file.write("-" * 40 + "\n")
    file.write(
        category_table.to_string()
    )

    file.write("\n\n")

    file.write("SUITABILITY PROBABILITY STATISTICS\n")
    file.write("-" * 40 + "\n")
    file.write(
        probability.describe().to_string()
    )


print()
print(f"Text report saved to:")
print(report_path)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("RECOMMENDATION ENGINE EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Evaluation outputs saved to:")
print(OUTPUT_DIR)

print()
print("Important files generated:")

print("1. recommendation_engine_summary.csv")
print("2. recommendation_category_distribution.csv")
print("3. crop_recommendation_frequency.csv")
print("4. probability_distribution.csv")
print("5. top_1_2_3_crop_distribution.csv")
print("6. lga_top_3_recommendations.csv")
print("7. recommendation_engine_evaluation_report.txt")
print("8. recommendation_category_distribution.png")
print("9. crop_recommendation_frequency.png")
print("10. suitability_probability_distribution.png")

print()
print("=" * 70)