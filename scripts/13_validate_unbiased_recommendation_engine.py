"""
======================================================================
13_validate_unbiased_recommendation_engine.py

VALIDATE THE REBUILT UNBIASED CROP RECOMMENDATION ENGINE

Purpose:
    1. Load the original recommendation results.
    2. Load the rebuilt unbiased recommendation results.
    3. Compare Top-1 recommendation distributions.
    4. Measure crop coverage.
    5. Measure Top-1, Top-3 and Top-5 diversity.
    6. Calculate recommendation entropy.
    7. Calculate concentration metrics.
    8. Identify crops that are completely excluded.
    9. Compare recommendation dominance before and after correction.
   10. Analyze recommendation diversity by location.
   11. Generate plots.
   12. Generate CSV tables.
   13. Generate a JSON summary.
   14. Generate a human-readable validation report.

Project:
    CropRecommendation

Windows-compatible paths are constructed using pathlib,
so backslash escaping problems are avoided.
======================================================================
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ======================================================================
# 0. PROJECT PATHS
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_DATA_PATH = (
    DATA_DIR / "north_central_training_dataset.csv"
)

RECOMMENDATION_DATASET_PATH = (
    DATA_DIR
    / "recommendations"
    / "crop_recommendations.csv"
)

ORIGINAL_MODEL_PATH = (
    MODELS_DIR / "final_crop_suitability_model.joblib"
)

CORRECTED_MODEL_PATH = (
    MODELS_DIR
    / "corrected"
    / "crop_suitability_model_corrected.joblib"
)

UNBIASED_ENGINE_PATH = (
    MODELS_DIR
    / "recommendation_engine"
    / "unbiased_crop_recommendation_engine.joblib"
)


# ----------------------------------------------------------------------
# Possible locations for the original recommendation output
# ----------------------------------------------------------------------

ORIGINAL_RECOMMENDATION_CANDIDATES = [
    DATA_DIR
    / "model_results"
    / "bias_investigation"
    / "tables"
    / "top1_distribution.csv",

    DATA_DIR
    / "model_results"
    / "bias_investigation"
    / "tables"
    / "top1_distribution.csv",

    DATA_DIR
    / "model_results"
    / "bias_investigation"
    / "tables"
    / "top1_recommendation_distribution.csv",

    DATA_DIR
    / "model_results"
    / "recommendation_engine"
    / "tables"
    / "top1_recommendation_distribution.csv",
]


# ----------------------------------------------------------------------
# Script 12 output
# ----------------------------------------------------------------------

UNBIASED_RESULTS_DIR = (
    DATA_DIR
    / "model_results"
    / "unbiased_recommendation_engine"
)

UNBIASED_TABLES_DIR = (
    UNBIASED_RESULTS_DIR / "tables"
)

UNBIASED_ALL_SCORES_PATH = (
    UNBIASED_TABLES_DIR
    / "all_crop_recommendation_scores.csv"
)

UNBIASED_TOP1_PATH = (
    UNBIASED_TABLES_DIR
    / "top1_crop_recommendations.csv"
)

UNBIASED_TOP3_PATH = (
    UNBIASED_TABLES_DIR
    / "top3_crop_recommendations.csv"
)

UNBIASED_TOP5_PATH = (
    UNBIASED_TABLES_DIR
    / "top5_crop_recommendations.csv"
)

UNBIASED_TOP1_DISTRIBUTION_PATH = (
    UNBIASED_TABLES_DIR
    / "top1_recommendation_distribution.csv"
)

UNBIASED_PROBABILITY_SUMMARY_PATH = (
    UNBIASED_TABLES_DIR
    / "crop_probability_summary.csv"
)


# ----------------------------------------------------------------------
# Script 13 output directory
# ----------------------------------------------------------------------

VALIDATION_DIR = (
    DATA_DIR
    / "model_results"
    / "unbiased_recommendation_engine_validation"
)

TABLES_DIR = VALIDATION_DIR / "tables"
PLOTS_DIR = VALIDATION_DIR / "plots"

VALIDATION_SUMMARY_PATH = (
    VALIDATION_DIR
    / "unbiased_recommendation_validation_summary.json"
)

VALIDATION_REPORT_PATH = (
    VALIDATION_DIR
    / "unbiased_recommendation_validation_report.txt"
)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_step(number, total, title):
    print()
    print("=" * 70)
    print(f"[{number}/{total}] {title}")
    print("=" * 70)


def check_file(path, name):
    if path.exists():
        print(f"Found {name}:")
        print(path)
        return True

    print(f"WARNING: {name} not found:")
    print(path)
    return False


def ensure_directories():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_percentage(value, total):
    if total == 0:
        return 0.0
    return float((value / total) * 100.0)


def normalize_crop_name(value):
    if pd.isna(value):
        return None

    return str(value).strip()


def find_original_recommendation_distribution():
    """
    Search known output locations for the original Top-1 distribution.
    """

    for candidate in ORIGINAL_RECOMMENDATION_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def find_all_csv_files(root):
    """
    Find CSV files recursively.
    """

    if not root.exists():
        return []

    return list(root.rglob("*.csv"))


def detect_column(df, candidates):
    """
    Find the first matching column from a list of possible names.
    """

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def calculate_entropy(probabilities):
    """
    Shannon entropy using natural logarithm.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    probabilities = probabilities[
        probabilities > 0
    ]

    if len(probabilities) == 0:
        return 0.0

    return float(
        -np.sum(
            probabilities * np.log(probabilities)
        )
    )


def normalized_entropy(probabilities):
    """
    Entropy normalized to [0, 1].

    1.0 = perfectly even distribution.
    0.0 = complete concentration on one crop.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    if len(probabilities) <= 1:
        return 0.0

    entropy = calculate_entropy(probabilities)

    maximum_entropy = np.log(len(probabilities))

    if maximum_entropy == 0:
        return 0.0

    return float(entropy / maximum_entropy)


def calculate_hhi(counts):
    """
    Herfindahl-Hirschman Index.

    Lower = more diverse.
    Higher = more concentrated.
    """

    counts = np.asarray(
        counts,
        dtype=float
    )

    total = counts.sum()

    if total == 0:
        return 0.0

    shares = counts / total

    return float(np.sum(shares ** 2))


def calculate_effective_number(probabilities):
    """
    Effective number of crops:

        exp(entropy)

    Higher values indicate greater diversity.
    """

    entropy = calculate_entropy(probabilities)

    return float(np.exp(entropy))


def top_n_distribution(df, crop_column, rank_column, n):
    """
    Count how often each crop appears within Top-N.
    """

    filtered = df[df[rank_column] <= n].copy()

    counts = (
        filtered[crop_column]
        .value_counts()
        .rename("count")
        .reset_index()
    )

    counts.columns = [
        "crop",
        "count"
    ]

    return counts


def calculate_rank_metrics(
    recommendation_df,
    crop_column,
    rank_column,
    crops
):
    """
    Calculate Top-1, Top-3 and Top-5 metrics.
    """

    total_locations = (
        recommendation_df["location_id"]
        .nunique()
    )

    rows = []

    for n in [1, 3, 5]:

        distribution = top_n_distribution(
            recommendation_df,
            crop_column,
            rank_column,
            n
        )

        count_map = dict(
            zip(
                distribution["crop"],
                distribution["count"]
            )
        )

        counts = np.array(
            [
                count_map.get(crop, 0)
                for crop in crops
            ],
            dtype=float
        )

        total_recommendations = counts.sum()

        probabilities = (
            counts / total_recommendations
            if total_recommendations > 0
            else np.zeros(len(counts))
        )

        coverage = int(
            np.sum(counts > 0)
        )

        coverage_percentage = safe_percentage(
            coverage,
            len(crops)
        )

        entropy = calculate_entropy(
            probabilities
        )

        norm_entropy = normalized_entropy(
            probabilities
        )

        hhi = calculate_hhi(counts)

        effective_number = (
            calculate_effective_number(
                probabilities
            )
            if total_recommendations > 0
            else 0.0
        )

        dominant_index = (
            int(np.argmax(counts))
            if len(counts) > 0
            else 0
        )

        dominant_crop = (
            crops[dominant_index]
            if len(crops) > 0
            else None
        )

        dominant_count = (
            counts[dominant_index]
            if len(counts) > 0
            else 0
        )

        dominant_percentage = safe_percentage(
            dominant_count,
            total_recommendations
        )

        rows.append({
            "rank_level": f"Top-{n}",
            "total_recommendations": int(
                total_recommendations
            ),
            "crop_coverage": coverage,
            "crop_coverage_percentage":
                coverage_percentage,
            "entropy": entropy,
            "normalized_entropy":
                norm_entropy,
            "hhi": hhi,
            "effective_number_of_crops":
                effective_number,
            "dominant_crop": dominant_crop,
            "dominant_count": int(
                dominant_count
            ),
            "dominant_percentage":
                dominant_percentage,
        })

    return pd.DataFrame(rows)


def calculate_top1_distribution(
    recommendation_df,
    crop_column,
    rank_column,
    crops
):
    """
    Calculate Top-1 distribution.
    """

    top1 = recommendation_df[
        recommendation_df[rank_column] == 1
    ].copy()

    counts = (
        top1[crop_column]
        .value_counts()
    )

    rows = []

    total = len(top1)

    for crop in crops:

        count = int(
            counts.get(crop, 0)
        )

        percentage = safe_percentage(
            count,
            total
        )

        rows.append({
            "crop": crop,
            "top1_count": count,
            "top1_percentage": percentage
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "top1_count",
        ascending=False
    ).reset_index(drop=True)

    return result


def calculate_top_n_presence(
    recommendation_df,
    crop_column,
    rank_column,
    location_column,
    crops
):
    """
    Calculate the percentage of locations in which each crop
    appears in Top-1, Top-3 and Top-5.
    """

    locations = (
        recommendation_df[location_column]
        .drop_duplicates()
        .tolist()
    )

    total_locations = len(locations)

    rows = []

    for crop in crops:

        crop_rows = recommendation_df[
            recommendation_df[crop_column] == crop
        ]

        row = {
            "crop": crop
        }

        for n in [1, 3, 5]:

            selected = crop_rows[
                crop_rows[rank_column] <= n
            ]

            location_count = (
                selected[location_column]
                .nunique()
            )

            percentage = safe_percentage(
                location_count,
                total_locations
            )

            row[
                f"locations_in_top{n}_count"
            ] = int(location_count)

            row[
                f"locations_in_top{n}_percentage"
            ] = percentage

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_location_diversity(
    recommendation_df,
    crop_column,
    rank_column,
    location_column
):
    """
    Calculate how diverse the recommendations are for each location.
    """

    rows = []

    for location, group in recommendation_df.groupby(
        location_column
    ):

        top5 = group[
            group[rank_column] <= 5
        ]

        unique_top5 = (
            top5[crop_column]
            .nunique()
        )

        top3 = group[
            group[rank_column] <= 3
        ]

        unique_top3 = (
            top3[crop_column]
            .nunique()
        )

        top1 = group[
            group[rank_column] == 1
        ]

        top1_crop = (
            top1[crop_column].iloc[0]
            if len(top1) > 0
            else None
        )

        rows.append({
            location_column: location,
            "unique_crops_top1":
                int(top1[crop_column].nunique()),
            "unique_crops_top3":
                int(unique_top3),
            "unique_crops_top5":
                int(unique_top5),
            "top1_crop": top1_crop,
        })

    return pd.DataFrame(rows)


def save_dataframe(df, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved:")
    print(path)


def save_plot(fig, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved plot:")
    print(path)


# ======================================================================
# START
# ======================================================================

print_header(
    "VALIDATING UNBIASED CROP RECOMMENDATION ENGINE"
)


# ======================================================================
# [1] CHECK INPUT FILES
# ======================================================================

print_step(
    1,
    15,
    "CHECKING INPUT FILES"
)

required_files = {
    TRAINING_DATA_PATH:
        "Training dataset",

    RECOMMENDATION_DATASET_PATH:
        "Recommendation dataset",

    ORIGINAL_MODEL_PATH:
        "Original model",

    CORRECTED_MODEL_PATH:
        "Corrected model",

    UNBIASED_ENGINE_PATH:
        "Unbiased recommendation engine",

    UNBIASED_ALL_SCORES_PATH:
        "Unbiased recommendation scores",

    UNBIASED_TOP1_PATH:
        "Unbiased Top-1 recommendations",

    UNBIASED_TOP3_PATH:
        "Unbiased Top-3 recommendations",

    UNBIASED_TOP5_PATH:
        "Unbiased Top-5 recommendations",
}

for path, name in required_files.items():

    if not check_file(path, name):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n\n"
            f"Please run Script 12 first."
        )


# ======================================================================
# [2] CREATE OUTPUT DIRECTORIES
# ======================================================================

print_step(
    2,
    15,
    "CREATING VALIDATION OUTPUT DIRECTORIES"
)

ensure_directories()

print("Output directory:")
print(VALIDATION_DIR)


# ======================================================================
# [3] LOAD DATASETS
# ======================================================================

print_step(
    3,
    15,
    "LOADING DATASETS"
)

training_df = pd.read_csv(
    TRAINING_DATA_PATH
)

recommendation_df = pd.read_csv(
    RECOMMENDATION_DATASET_PATH
)

unbiased_scores_df = pd.read_csv(
    UNBIASED_ALL_SCORES_PATH
)

unbiased_top1_df = pd.read_csv(
    UNBIASED_TOP1_PATH
)

unbiased_top3_df = pd.read_csv(
    UNBIASED_TOP3_PATH
)

unbiased_top5_df = pd.read_csv(
    UNBIASED_TOP5_PATH
)

print(
    f"Training dataset shape: "
    f"{training_df.shape}"
)

print(
    f"Recommendation dataset shape: "
    f"{recommendation_df.shape}"
)

print(
    f"Unbiased scores shape: "
    f"{unbiased_scores_df.shape}"
)

print(
    f"Unbiased Top-1 shape: "
    f"{unbiased_top1_df.shape}"
)

print(
    f"Unbiased Top-3 shape: "
    f"{unbiased_top3_df.shape}"
)

print(
    f"Unbiased Top-5 shape: "
    f"{unbiased_top5_df.shape}"
)


# ======================================================================
# [4] LOAD MODELS
# ======================================================================

print_step(
    4,
    15,
    "LOADING MODELS"
)

original_model = joblib.load(
    ORIGINAL_MODEL_PATH
)

corrected_model = joblib.load(
    CORRECTED_MODEL_PATH
)

unbiased_engine = joblib.load(
    UNBIASED_ENGINE_PATH
)

print(
    "Original model:",
    type(original_model).__name__
)

print(
    "Corrected model:",
    type(corrected_model).__name__
)

print(
    "Unbiased engine:",
    type(unbiased_engine).__name__
)


# ======================================================================
# [5] VALIDATE UNBIASED SCORE DATASET
# ======================================================================

print_step(
    5,
    15,
    "VALIDATING UNBIASED RECOMMENDATION DATA"
)

crop_column = detect_column(
    unbiased_scores_df,
    ["crop"]
)

location_column = detect_column(
    unbiased_scores_df,
    [
        "location_id",
        "location",
        "locationid"
    ]
)

rank_column = detect_column(
    unbiased_scores_df,
    [
        "recommendation_rank",
        "rank"
    ]
)

probability_column = detect_column(
    unbiased_scores_df,
    [
        "suitability_probability",
        "probability",
        "score",
        "recommendation_score"
    ]
)

if crop_column is None:
    raise ValueError(
        "Could not identify crop column."
    )

if location_column is None:
    raise ValueError(
        "Could not identify location column."
    )

if rank_column is None:
    raise ValueError(
        "Could not identify recommendation rank column."
    )

if probability_column is None:
    raise ValueError(
        "Could not identify probability/score column."
    )

print(
    f"Crop column: {crop_column}"
)

print(
    f"Location column: {location_column}"
)

print(
    f"Rank column: {rank_column}"
)

print(
    f"Probability column: {probability_column}"
)

unbiased_scores_df[crop_column] = (
    unbiased_scores_df[crop_column]
    .apply(normalize_crop_name)
)

unbiased_scores_df[
    location_column
] = unbiased_scores_df[
    location_column
].astype(str)

crops = sorted(
    unbiased_scores_df[
        crop_column
    ].dropna().unique().tolist()
)

locations = sorted(
    unbiased_scores_df[
        location_column
    ].dropna().unique().tolist()
)

print(
    f"Unique crops: {len(crops)}"
)

print(
    f"Unique locations: {len(locations)}"
)

expected_records = (
    len(crops) * len(locations)
)

actual_records = len(
    unbiased_scores_df
)

print(
    f"Expected recommendation records: "
    f"{expected_records}"
)

print(
    f"Actual recommendation records: "
    f"{actual_records}"
)

if expected_records != actual_records:

    print(
        "WARNING: Recommendation record count "
        "does not equal location × crop coverage."
    )

else:

    print(
        "SUCCESS: Complete location × crop "
        "recommendation coverage."
    )


# ======================================================================
# [6] CALCULATE TOP-N METRICS
# ======================================================================

print_step(
    6,
    15,
    "CALCULATING TOP-1, TOP-3 AND TOP-5 DIVERSITY"
)

rank_metrics = calculate_rank_metrics(
    unbiased_scores_df,
    crop_column,
    rank_column,
    crops
)

print()
print(rank_metrics.to_string(index=False))

save_dataframe(
    rank_metrics,
    TABLES_DIR / "rank_level_diversity_metrics.csv"
)


# ======================================================================
# [7] TOP-1 DISTRIBUTION
# ======================================================================

print_step(
    7,
    15,
    "ANALYZING TOP-1 RECOMMENDATION DISTRIBUTION"
)

top1_distribution = calculate_top1_distribution(
    unbiased_scores_df,
    crop_column,
    rank_column,
    crops
)

print()
print(top1_distribution.to_string(index=False))

save_dataframe(
    top1_distribution,
    TABLES_DIR
    / "unbiased_top1_distribution.csv"
)


dominant_row = (
    top1_distribution.iloc[0]
    if len(top1_distribution) > 0
    else None
)

if dominant_row is not None:

    dominant_crop = dominant_row["crop"]

    dominant_percentage = float(
        dominant_row["top1_percentage"]
    )

else:

    dominant_crop = None
    dominant_percentage = 0.0

print()
print(
    f"Dominant Top-1 crop: "
    f"{dominant_crop}"
)

print(
    f"Top-1 dominance: "
    f"{dominant_percentage:.2f}%"
)


# ======================================================================
# [8] CROP COVERAGE
# ======================================================================

print_step(
    8,
    15,
    "ANALYZING CROP COVERAGE"
)

presence_df = calculate_top_n_presence(
    unbiased_scores_df,
    crop_column,
    rank_column,
    location_column,
    crops
)

print()
print(presence_df.to_string(index=False))

save_dataframe(
    presence_df,
    TABLES_DIR
    / "crop_top_n_presence.csv"
)

top1_coverage = int(
    (presence_df[
        "locations_in_top1_count"
    ] > 0).sum()
)

top3_coverage = int(
    (presence_df[
        "locations_in_top3_count"
    ] > 0).sum()
)

top5_coverage = int(
    (presence_df[
        "locations_in_top5_count"
    ] > 0).sum()
)

print()
print(
    f"Top-1 crop coverage: "
    f"{top1_coverage}/{len(crops)}"
)

print(
    f"Top-3 crop coverage: "
    f"{top3_coverage}/{len(crops)}"
)

print(
    f"Top-5 crop coverage: "
    f"{top5_coverage}/{len(crops)}"
)


# ======================================================================
# [9] IDENTIFY EXCLUDED CROPS
# ======================================================================

print_step(
    9,
    15,
    "IDENTIFYING COMPLETELY EXCLUDED CROPS"
)

top1_excluded = presence_df[
    presence_df[
        "locations_in_top1_count"
    ] == 0
]["crop"].tolist()

top3_excluded = presence_df[
    presence_df[
        "locations_in_top3_count"
    ] == 0
]["crop"].tolist()

top5_excluded = presence_df[
    presence_df[
        "locations_in_top5_count"
    ] == 0
]["crop"].tolist()

print(
    "Crops never appearing in Top-1:"
)

print(
    top1_excluded
    if top1_excluded
    else "None"
)

print()
print(
    "Crops never appearing in Top-3:"
)

print(
    top3_excluded
    if top3_excluded
    else "None"
)

print()
print(
    "Crops never appearing in Top-5:"
)

print(
    top5_excluded
    if top5_excluded
    else "None"
)


# ======================================================================
# [10] LOCATION-LEVEL DIVERSITY
# ======================================================================

print_step(
    10,
    15,
    "ANALYZING LOCATION-LEVEL RECOMMENDATION DIVERSITY"
)

location_diversity_df = calculate_location_diversity(
    unbiased_scores_df,
    crop_column,
    rank_column,
    location_column
)

print()

print(
    location_diversity_df.head(
        20
    ).to_string(index=False)
)

save_dataframe(
    location_diversity_df,
    TABLES_DIR
    / "location_level_recommendation_diversity.csv"
)

mean_top3_diversity = float(
    location_diversity_df[
        "unique_crops_top3"
    ].mean()
)

mean_top5_diversity = float(
    location_diversity_df[
        "unique_crops_top5"
    ].mean()
)

print()
print(
    f"Mean unique crops per location "
    f"in Top-3: {mean_top3_diversity:.3f}"
)

print(
    f"Mean unique crops per location "
    f"in Top-5: {mean_top5_diversity:.3f}"
)


# ======================================================================
# [11] COMPARE WITH ORIGINAL BIAS INVESTIGATION
# ======================================================================

print_step(
    11,
    15,
    "COMPARING ORIGINAL AND REBUILT RECOMMENDATION DISTRIBUTIONS"
)

original_distribution_path = (
    find_original_recommendation_distribution()
)

original_distribution_df = None

if original_distribution_path is not None:

    print(
        "Original recommendation distribution found:"
    )

    print(
        original_distribution_path
    )

    original_distribution_df = pd.read_csv(
        original_distribution_path
    )

    print()
    print(
        original_distribution_df.to_string(
            index=False
        )
    )

else:

    print(
        "WARNING: Original Top-1 distribution "
        "file could not be located."
    )

    print(
        "The validation will continue using "
        "the previously reported 80.99% baseline "
        "if necessary."
    )


# ----------------------------------------------------------------------
# Extract original dominance
# ----------------------------------------------------------------------

original_dominance = None
original_dominant_crop = None

if original_distribution_df is not None:

    original_crop_col = detect_column(
        original_distribution_df,
        [
            "crop",
            "recommended_crop"
        ]
    )

    original_percentage_col = detect_column(
        original_distribution_df,
        [
            "top1_percentage",
            "percentage",
            "proportion"
        ]
    )

    if (
        original_crop_col is not None
        and original_percentage_col is not None
    ):

        temp = (
            original_distribution_df
            .sort_values(
                original_percentage_col,
                ascending=False
            )
            .reset_index(drop=True)
        )

        if len(temp) > 0:

            original_dominant_crop = (
                temp.loc[
                    0,
                    original_crop_col
                ]
            )

            original_dominance = float(
                temp.loc[
                    0,
                    original_percentage_col
                ]
            )

            # Handle proportions such as 0.8099
            if original_dominance <= 1:
                original_dominance *= 100.0


# ----------------------------------------------------------------------
# Fallback baseline from Script 7/previous investigation
# ----------------------------------------------------------------------

if original_dominance is None:

    original_dominance = 80.991736

    original_dominant_crop = (
        "Bambara_groundnut"
    )

    print()
    print(
        "Using previously measured baseline:"
    )

    print(
        f"Original dominant crop: "
        f"{original_dominant_crop}"
    )

    print(
        f"Original dominance: "
        f"{original_dominance:.2f}%"
    )


# ----------------------------------------------------------------------
# Calculate reduction
# ----------------------------------------------------------------------

dominance_reduction = (
    original_dominance
    - dominant_percentage
)

if original_dominance > 0:

    dominance_reduction_percentage = (
        dominance_reduction
        / original_dominance
    ) * 100.0

else:

    dominance_reduction_percentage = 0.0


print()
print(
    "ORIGINAL ENGINE"
)

print(
    f"Dominant crop: "
    f"{original_dominant_crop}"
)

print(
    f"Dominance: "
    f"{original_dominance:.2f}%"
)

print()
print(
    "REBUILT ENGINE"
)

print(
    f"Dominant crop: "
    f"{dominant_crop}"
)

print(
    f"Dominance: "
    f"{dominant_percentage:.2f}%"
)

print()
print(
    f"Absolute dominance reduction: "
    f"{dominance_reduction:.2f} percentage points"
)

print(
    f"Relative dominance reduction: "
    f"{dominance_reduction_percentage:.2f}%"
)


# ======================================================================
# [12] GENERATE PLOTS
# ======================================================================

print_step(
    12,
    15,
    "GENERATING VALIDATION PLOTS"
)

# ----------------------------------------------------------------------
# Plot 1: Top-1 distribution
# ----------------------------------------------------------------------

fig = plt.figure(
    figsize=(12, 7)
)

plt.bar(
    top1_distribution["crop"],
    top1_distribution["top1_percentage"]
)

plt.xlabel(
    "Crop"
)

plt.ylabel(
    "Top-1 Recommendation Percentage"
)

plt.title(
    "Rebuilt Engine: Top-1 Crop Recommendation Distribution"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

save_plot(
    fig,
    PLOTS_DIR
    / "unbiased_top1_distribution.png"
)


# ----------------------------------------------------------------------
# Plot 2: Crop Top-N coverage
# ----------------------------------------------------------------------

coverage_plot_df = pd.DataFrame({
    "Top-1": [
        safe_percentage(
            x,
            len(locations)
        )
        for x in presence_df[
            "locations_in_top1_count"
        ]
    ],
    "Top-3": [
        safe_percentage(
            x,
            len(locations)
        )
        for x in presence_df[
            "locations_in_top3_count"
        ]
    ],
    "Top-5": [
        safe_percentage(
            x,
            len(locations)
        )
        for x in presence_df[
            "locations_in_top5_count"
        ]
    ],
}, index=presence_df["crop"])

fig = plt.figure(
    figsize=(13, 7)
)

coverage_plot_df.plot(
    kind="bar",
    ax=plt.gca()
)

plt.xlabel(
    "Crop"
)

plt.ylabel(
    "Percentage of Locations"
)

plt.title(
    "Crop Coverage Across Top-1, Top-3 and Top-5"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Recommendation Level"
)

plt.tight_layout()

save_plot(
    fig,
    PLOTS_DIR
    / "crop_top_n_coverage.png"
)


# ----------------------------------------------------------------------
# Plot 3: Original vs rebuilt dominance
# ----------------------------------------------------------------------

comparison_plot_df = pd.DataFrame({
    "Engine": [
        "Original",
        "Rebuilt"
    ],
    "Dominance": [
        original_dominance,
        dominant_percentage
    ]
})

fig = plt.figure(
    figsize=(9, 6)
)

plt.bar(
    comparison_plot_df["Engine"],
    comparison_plot_df["Dominance"]
)

plt.ylabel(
    "Dominant Top-1 Crop Percentage"
)

plt.title(
    "Recommendation Dominance Before and After Rebuilding"
)

plt.ylim(
    0,
    max(
        100,
        original_dominance + 10
    )
)

plt.tight_layout()

save_plot(
    fig,
    PLOTS_DIR
    / "dominance_before_after.png"
)


# ----------------------------------------------------------------------
# Plot 4: Rank-level diversity
# ----------------------------------------------------------------------

fig = plt.figure(
    figsize=(9, 6)
)

plt.plot(
    rank_metrics["rank_level"],
    rank_metrics["normalized_entropy"],
    marker="o"
)

plt.xlabel(
    "Recommendation Level"
)

plt.ylabel(
    "Normalized Entropy"
)

plt.title(
    "Recommendation Diversity by Ranking Level"
)

plt.ylim(
    0,
    1
)

plt.tight_layout()

save_plot(
    fig,
    PLOTS_DIR
    / "rank_level_entropy.png"
)


# ----------------------------------------------------------------------
# Plot 5: Location-level Top-5 diversity
# ----------------------------------------------------------------------

fig = plt.figure(
    figsize=(10, 6)
)

plt.hist(
    location_diversity_df[
        "unique_crops_top5"
    ],
    bins=range(
        1,
        len(crops) + 2
    )
)

plt.xlabel(
    "Number of Unique Crops in Top-5"
)

plt.ylabel(
    "Number of Locations"
)

plt.title(
    "Location-Level Top-5 Recommendation Diversity"
)

plt.tight_layout()

save_plot(
    fig,
    PLOTS_DIR
    / "location_top5_diversity.png"
)


# ======================================================================
# [13] BIAS ASSESSMENT
# ======================================================================

print_step(
    13,
    15,
    "PERFORMING FINAL BIAS ASSESSMENT"
)

warnings_list = []
positive_findings = []

# ----------------------------------------------------------------------
# Dominance
# ----------------------------------------------------------------------

if dominant_percentage > 50:

    warnings_list.append(
        "The rebuilt engine still has a "
        "Top-1 crop accounting for more than 50% "
        "of recommendations."
    )

else:

    positive_findings.append(
        "No single crop accounts for more than "
        "50% of Top-1 recommendations."
    )


# ----------------------------------------------------------------------
# Dominance reduction
# ----------------------------------------------------------------------

if dominance_reduction > 0:

    positive_findings.append(
        "Top-1 recommendation dominance decreased "
        "after rebuilding the recommendation engine."
    )

elif dominance_reduction == 0:

    warnings_list.append(
        "Top-1 recommendation dominance did not "
        "change compared with the original engine."
    )

else:

    warnings_list.append(
        "Top-1 recommendation dominance increased "
        "after rebuilding."
    )


# ----------------------------------------------------------------------
# Top-1 coverage
# ----------------------------------------------------------------------

if top1_coverage < len(crops):

    warnings_list.append(
        f"{len(crops) - top1_coverage} crop(s) "
        "never appear as a Top-1 recommendation."
    )

else:

    positive_findings.append(
        "All crops appear at least once as "
        "Top-1 recommendations."
    )


# ----------------------------------------------------------------------
# Top-3 coverage
# ----------------------------------------------------------------------

if top3_coverage < len(crops):

    warnings_list.append(
        f"{len(crops) - top3_coverage} crop(s) "
        "never appear in Top-3 recommendations."
    )

else:

    positive_findings.append(
        "All crops appear in Top-3 recommendations."
    )


# ----------------------------------------------------------------------
# Top-5 coverage
# ----------------------------------------------------------------------

if top5_coverage < len(crops):

    warnings_list.append(
        f"{len(crops) - top5_coverage} crop(s) "
        "never appear in Top-5 recommendations."
    )

else:

    positive_findings.append(
        "All crops appear in Top-5 recommendations."
    )


# ----------------------------------------------------------------------
# Entropy
# ----------------------------------------------------------------------

top1_entropy = float(
    rank_metrics.loc[
        rank_metrics["rank_level"] == "Top-1",
        "normalized_entropy"
    ].iloc[0]
)

top3_entropy = float(
    rank_metrics.loc[
        rank_metrics["rank_level"] == "Top-3",
        "normalized_entropy"
    ].iloc[0]
)

top5_entropy = float(
    rank_metrics.loc[
        rank_metrics["rank_level"] == "Top-5",
        "normalized_entropy"
    ].iloc[0]
)

if top1_entropy < 0.40:

    warnings_list.append(
        "Top-1 recommendation diversity remains low "
        "according to normalized entropy."
    )

elif top1_entropy >= 0.60:

    positive_findings.append(
        "Top-1 recommendation distribution has "
        "moderate-to-high diversity."
    )


# ----------------------------------------------------------------------
# HHI
# ----------------------------------------------------------------------

top1_hhi = float(
    rank_metrics.loc[
        rank_metrics["rank_level"] == "Top-1",
        "hhi"
    ].iloc[0]
)

if top1_hhi > 0.50:

    warnings_list.append(
        "Top-1 recommendations remain highly "
        "concentrated according to HHI."
    )

else:

    positive_findings.append(
        "Top-1 concentration is below the "
        "high-concentration threshold."
    )


# ----------------------------------------------------------------------
# Final status
# ----------------------------------------------------------------------

if len(warnings_list) == 0:

    overall_status = (
        "PASS - No major recommendation "
        "bias indicators detected."
    )

elif dominance_reduction > 20 and dominant_percentage < 50:

    overall_status = (
        "IMPROVED - Recommendation bias has "
        "been substantially reduced, although "
        "some crop-level limitations remain."
    )

else:

    overall_status = (
        "PARTIAL IMPROVEMENT - The rebuilt engine "
        "shows improvement but additional "
        "investigation may be required."
    )


print()
print(
    "OVERALL VALIDATION STATUS:"
)

print(
    overall_status
)

print()
print(
    "Positive findings:"
)

for item in positive_findings:
    print(
        f"  + {item}"
    )

print()
print(
    "Warnings:"
)

if warnings_list:

    for item in warnings_list:
        print(
            f"  - {item}"
        )

else:

    print(
        "  None"
    )


# ======================================================================
# [14] SAVE SUMMARY
# ======================================================================

print_step(
    14,
    15,
    "SAVING VALIDATION SUMMARY"
)

top1_metrics = rank_metrics[
    rank_metrics["rank_level"] == "Top-1"
].iloc[0]

top3_metrics = rank_metrics[
    rank_metrics["rank_level"] == "Top-3"
].iloc[0]

top5_metrics = rank_metrics[
    rank_metrics["rank_level"] == "Top-5"
].iloc[0]


summary = {
    "validation_status": overall_status,

    "project_root": str(
        PROJECT_ROOT
    ),

    "training_records": int(
        len(training_df)
    ),

    "recommendation_records": int(
        len(unbiased_scores_df)
    ),

    "locations_analyzed": int(
        len(locations)
    ),

    "crops_analyzed": int(
        len(crops)
    ),

    "crops": crops,

    "original_engine": {
        "dominant_crop":
            original_dominant_crop,
        "dominance_percentage":
            original_dominance,
    },

    "rebuilt_engine": {
        "dominant_crop":
            dominant_crop,
        "dominance_percentage":
            dominant_percentage,
    },

    "dominance_change": {
        "absolute_percentage_point_reduction":
            dominance_reduction,
        "relative_percentage_reduction":
            dominance_reduction_percentage,
    },

    "top1": {
        "coverage":
            int(top1_metrics[
                "crop_coverage"
            ]),
        "coverage_percentage":
            float(top1_metrics[
                "crop_coverage_percentage"
            ]),
        "entropy":
            float(top1_metrics[
                "entropy"
            ]),
        "normalized_entropy":
            float(top1_metrics[
                "normalized_entropy"
            ]),
        "hhi":
            float(top1_metrics[
                "hhi"
            ]),
        "effective_number_of_crops":
            float(top1_metrics[
                "effective_number_of_crops"
            ]),
        "dominant_crop":
            top1_metrics[
                "dominant_crop"
            ],
        "dominant_percentage":
            float(top1_metrics[
                "dominant_percentage"
            ]),
    },

    "top3": {
        "coverage":
            int(top3_metrics[
                "crop_coverage"
            ]),
        "coverage_percentage":
            float(top3_metrics[
                "crop_coverage_percentage"
            ]),
        "entropy":
            float(top3_metrics[
                "entropy"
            ]),
        "normalized_entropy":
            float(top3_metrics[
                "normalized_entropy"
            ]),
        "hhi":
            float(top3_metrics[
                "hhi"
            ]),
        "effective_number_of_crops":
            float(top3_metrics[
                "effective_number_of_crops"
            ]),
    },

    "top5": {
        "coverage":
            int(top5_metrics[
                "crop_coverage"
            ]),
        "coverage_percentage":
            float(top5_metrics[
                "crop_coverage_percentage"
            ]),
        "entropy":
            float(top5_metrics[
                "entropy"
            ]),
        "normalized_entropy":
            float(top5_metrics[
                "normalized_entropy"
            ]),
        "hhi":
            float(top5_metrics[
                "hhi"
            ]),
        "effective_number_of_crops":
            float(top5_metrics[
                "effective_number_of_crops"
            ]),
    },

    "excluded_crops": {
        "top1":
            top1_excluded,
        "top3":
            top3_excluded,
        "top5":
            top5_excluded,
    },

    "location_diversity": {
        "mean_unique_crops_top3":
            mean_top3_diversity,
        "mean_unique_crops_top5":
            mean_top5_diversity,
    },

    "positive_findings":
        positive_findings,

    "warnings":
        warnings_list,

    "output_directory":
        str(VALIDATION_DIR),
}

with open(
    VALIDATION_SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print()
print(
    "Saved:"
)

print(
    VALIDATION_SUMMARY_PATH
)


# ======================================================================
# [15] SAVE HUMAN-READABLE REPORT
# ======================================================================

print_step(
    15,
    15,
    "GENERATING FINAL VALIDATION REPORT"
)

report_lines = []

report_lines.append(
    "=" * 70
)

report_lines.append(
    "UNBIASED CROP RECOMMENDATION ENGINE VALIDATION REPORT"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "OVERALL STATUS"
)

report_lines.append(
    overall_status
)

report_lines.append("")

report_lines.append(
    "DATASET SUMMARY"
)

report_lines.append(
    f"Locations analyzed: {len(locations)}"
)

report_lines.append(
    f"Crops analyzed: {len(crops)}"
)

report_lines.append(
    f"Recommendation records: "
    f"{len(unbiased_scores_df)}"
)

report_lines.append("")

report_lines.append(
    "ORIGINAL ENGINE"
)

report_lines.append(
    f"Dominant crop: "
    f"{original_dominant_crop}"
)

report_lines.append(
    f"Dominance: "
    f"{original_dominance:.2f}%"
)

report_lines.append("")

report_lines.append(
    "REBUILT ENGINE"
)

report_lines.append(
    f"Dominant crop: "
    f"{dominant_crop}"
)

report_lines.append(
    f"Dominance: "
    f"{dominant_percentage:.2f}%"
)

report_lines.append("")

report_lines.append(
    "BIAS REDUCTION"
)

report_lines.append(
    f"Absolute reduction: "
    f"{dominance_reduction:.2f} percentage points"
)

report_lines.append(
    f"Relative reduction: "
    f"{dominance_reduction_percentage:.2f}%"
)

report_lines.append("")

report_lines.append(
    "TOP-1 METRICS"
)

report_lines.append(
    f"Crop coverage: "
    f"{top1_metrics['crop_coverage']}/"
    f"{len(crops)}"
)

report_lines.append(
    f"Coverage percentage: "
    f"{top1_metrics['crop_coverage_percentage']:.2f}%"
)

report_lines.append(
    f"Normalized entropy: "
    f"{top1_metrics['normalized_entropy']:.4f}"
)

report_lines.append(
    f"HHI: "
    f"{top1_metrics['hhi']:.4f}"
)

report_lines.append(
    f"Effective number of crops: "
    f"{top1_metrics['effective_number_of_crops']:.4f}"
)

report_lines.append("")

report_lines.append(
    "TOP-3 METRICS"
)

report_lines.append(
    f"Crop coverage: "
    f"{top3_metrics['crop_coverage']}/"
    f"{len(crops)}"
)

report_lines.append(
    f"Coverage percentage: "
    f"{top3_metrics['crop_coverage_percentage']:.2f}%"
)

report_lines.append(
    f"Normalized entropy: "
    f"{top3_metrics['normalized_entropy']:.4f}"
)

report_lines.append(
    f"HHI: "
    f"{top3_metrics['hhi']:.4f}"
)

report_lines.append("")

report_lines.append(
    "TOP-5 METRICS"
)

report_lines.append(
    f"Crop coverage: "
    f"{top5_metrics['crop_coverage']}/"
    f"{len(crops)}"
)

report_lines.append(
    f"Coverage percentage: "
    f"{top5_metrics['crop_coverage_percentage']:.2f}%"
)

report_lines.append(
    f"Normalized entropy: "
    f"{top5_metrics['normalized_entropy']:.4f}"
)

report_lines.append(
    f"HHI: "
    f"{top5_metrics['hhi']:.4f}"
)

report_lines.append("")

report_lines.append(
    "EXCLUDED CROPS"
)

report_lines.append(
    "Top-1: "
    + (
        ", ".join(top1_excluded)
        if top1_excluded
        else "None"
    )
)

report_lines.append(
    "Top-3: "
    + (
        ", ".join(top3_excluded)
        if top3_excluded
        else "None"
    )
)

report_lines.append(
    "Top-5: "
    + (
        ", ".join(top5_excluded)
        if top5_excluded
        else "None"
    )
)

report_lines.append("")

report_lines.append(
    "LOCATION-LEVEL DIVERSITY"
)

report_lines.append(
    f"Mean unique crops in Top-3: "
    f"{mean_top3_diversity:.4f}"
)

report_lines.append(
    f"Mean unique crops in Top-5: "
    f"{mean_top5_diversity:.4f}"
)

report_lines.append("")

report_lines.append(
    "POSITIVE FINDINGS"
)

if positive_findings:

    for item in positive_findings:
        report_lines.append(
            f"+ {item}"
        )

else:

    report_lines.append(
        "None"
    )

report_lines.append("")

report_lines.append(
    "WARNINGS"
)

if warnings_list:

    for item in warnings_list:
        report_lines.append(
            f"- {item}"
        )

else:

    report_lines.append(
        "None"
    )

report_lines.append("")

report_lines.append(
    "OUTPUT DIRECTORY"
)

report_lines.append(
    str(VALIDATION_DIR)
)

report_lines.append("")

report_lines.append(
    "=" * 70
)

report_lines.append(
    "END OF VALIDATION REPORT"
)

report_lines.append(
    "=" * 70
)

with open(
    VALIDATION_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )

print()
print(
    "Saved:"
)

print(
    VALIDATION_REPORT_PATH
)


# ======================================================================
# FINAL CONSOLE SUMMARY
# ======================================================================

print_header(
    "SCRIPT 13 COMPLETED SUCCESSFULLY"
)

print(
    f"Original dominant crop: "
    f"{original_dominant_crop}"
)

print(
    f"Original dominance: "
    f"{original_dominance:.2f}%"
)

print()

print(
    f"Rebuilt dominant crop: "
    f"{dominant_crop}"
)

print(
    f"Rebuilt dominance: "
    f"{dominant_percentage:.2f}%"
)

print()

print(
    f"Dominance reduction: "
    f"{dominance_reduction:.2f} percentage points"
)

print(
    f"Relative reduction: "
    f"{dominance_reduction_percentage:.2f}%"
)

print()

print(
    f"Top-1 crop coverage: "
    f"{top1_coverage}/{len(crops)}"
)

print(
    f"Top-3 crop coverage: "
    f"{top3_coverage}/{len(crops)}"
)

print(
    f"Top-5 crop coverage: "
    f"{top5_coverage}/{len(crops)}"
)

print()

print(
    f"Top-1 normalized entropy: "
    f"{top1_entropy:.4f}"
)

print(
    f"Top-1 HHI: "
    f"{top1_hhi:.4f}"
)

print()

print(
    "Validation status:"
)

print(
    overall_status
)

print()

print(
    "Important outputs:"
)

print(
    f"1. {VALIDATION_SUMMARY_PATH}"
)

print(
    f"2. {VALIDATION_REPORT_PATH}"
)

print(
    f"3. {TABLES_DIR}"
)

print(
    f"4. {PLOTS_DIR}"
)

print()

print(
    "The rebuilt recommendation engine has now "
    "been quantitatively validated."
)

print(
    "=" * 70
)