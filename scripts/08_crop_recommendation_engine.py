from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

ENVIRONMENTAL_FILE = (
    PROCESSED_DIR / "north_central_environmental.csv"
)

CROP_REQUIREMENTS_FILE = (
    PROCESSED_DIR / "north_central_crop_requirements.csv"
)

CROP_MAPPING_FILE = (
    PROCESSED_DIR / "crop_label_mapping.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "final_crop_suitability_model.joblib"
)

OUTPUT_DIR = (
    PROCESSED_DIR / "recommendations"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "crop_recommendations.csv"
)

TOP_OUTPUT_FILE = (
    OUTPUT_DIR / "top_crop_recommendations.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Your climate data covers 2015–2025 = 11 years.
CLIMATE_YEARS = 11


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("BUILDING CROP RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print("\n[1/10] Checking input files...")

required_files = {
    "Environmental dataset": ENVIRONMENTAL_FILE,
    "Crop requirements": CROP_REQUIREMENTS_FILE,
    "Crop mapping": CROP_MAPPING_FILE,
    "Final model": MODEL_FILE,
}

for name, path in required_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    print(f"{name} found:")
    print(path)


# ============================================================
# 2. LOAD DATASETS AND MODEL
# ============================================================

print("\n[2/10] Loading datasets and model...")

environmental = pd.read_csv(
    ENVIRONMENTAL_FILE
)

crop_requirements = pd.read_csv(
    CROP_REQUIREMENTS_FILE
)

crop_mapping = pd.read_csv(
    CROP_MAPPING_FILE
)

model = joblib.load(
    MODEL_FILE
)

print(
    f"Environmental records: "
    f"{len(environmental)}"
)

print(
    f"Crop requirements: "
    f"{len(crop_requirements)}"
)

print(
    f"Crop mappings: "
    f"{len(crop_mapping)}"
)

print(
    f"Model loaded: "
    f"{type(model).__name__}"
)


# ============================================================
# 3. VALIDATE ENVIRONMENTAL DATASET
# ============================================================

print("\n[3/10] Validating environmental dataset...")

required_environmental_columns = [

    "location_id",
    "state",
    "lga",

    "latitude_climate",
    "longitude_climate",

    "mean_daily_precipitation_mm",
    "std_daily_precipitation_mm",
    "min_daily_precipitation_mm",
    "max_daily_precipitation_mm",
    "total_precipitation_mm",

    "mean_temperature_c",
    "std_temperature_c",
    "min_temperature_c",
    "max_temperature_c",
    "mean_max_temperature_c",
    "absolute_max_temperature_c",
    "mean_min_temperature_c",
    "absolute_min_temperature_c",

    "mean_relative_humidity_percent",
    "minimum_relative_humidity_percent",
    "maximum_relative_humidity_percent",

    "mean_solar_radiation",
    "minimum_solar_radiation",
    "maximum_solar_radiation",

    "mean_wind_speed",
    "minimum_wind_speed",
    "maximum_wind_speed",

    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "clay_percent",
    "sand_percent",
    "silt_percent",
    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]

missing_environmental_columns = [
    column
    for column in required_environmental_columns
    if column not in environmental.columns
]

if missing_environmental_columns:

    print(
        "\nMissing environmental columns:"
    )

    for column in missing_environmental_columns:
        print(f"- {column}")

    raise ValueError(
        "Environmental dataset does not contain "
        "all required columns."
    )

print(
    "SUCCESS: Environmental dataset "
    "contains all required columns."
)


# ============================================================
# 4. CREATE MODEL-READY ENVIRONMENTAL FEATURES
# ============================================================

print(
    "\n[4/10] Preparing model-ready "
    "environmental features..."
)


# ------------------------------------------------------------
# Latitude and longitude
# ------------------------------------------------------------

environmental["latitude"] = (
    environmental["latitude_climate"]
)

environmental["longitude"] = (
    environmental["longitude_climate"]
)


# ------------------------------------------------------------
# Mean annual precipitation
# ------------------------------------------------------------

print(
    "\nCalculating mean annual precipitation..."
)

print(
    f"Climate period: 2015–2025 "
    f"({CLIMATE_YEARS} years)"
)

environmental[
    "mean_annual_precipitation_mm"
] = (
    environmental[
        "total_precipitation_mm"
    ]
    / CLIMATE_YEARS
)

print(
    "Mean annual precipitation "
    "calculated successfully."
)


# ------------------------------------------------------------
# Environmental features used during training
# ------------------------------------------------------------

environmental_features = [

    "latitude",
    "longitude",

    "mean_daily_precipitation_mm",
    "std_daily_precipitation_mm",
    "min_daily_precipitation_mm",
    "max_daily_precipitation_mm",
    "total_precipitation_mm",
    "mean_annual_precipitation_mm",

    "mean_temperature_c",
    "std_temperature_c",
    "min_temperature_c",
    "max_temperature_c",
    "mean_max_temperature_c",
    "absolute_max_temperature_c",
    "mean_min_temperature_c",
    "absolute_min_temperature_c",

    "mean_relative_humidity_percent",
    "minimum_relative_humidity_percent",
    "maximum_relative_humidity_percent",

    "mean_solar_radiation",
    "minimum_solar_radiation",
    "maximum_solar_radiation",

    "mean_wind_speed",
    "minimum_wind_speed",
    "maximum_wind_speed",

    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "clay_percent",
    "sand_percent",
    "silt_percent",
    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]


print(
    f"\nEnvironmental features: "
    f"{len(environmental_features)}"
)


# ------------------------------------------------------------
# Check environmental features
# ------------------------------------------------------------

missing_features = [
    column
    for column in environmental_features
    if column not in environmental.columns
]

if missing_features:

    print(
        "\nMissing model features:"
    )

    for column in missing_features:
        print(f"- {column}")

    raise ValueError(
        "Required model features are missing."
    )


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

feature_missing = (
    environmental[
        environmental_features
    ]
    .isna()
    .sum()
    .sum()
)

if feature_missing > 0:

    raise ValueError(
        "Environmental features contain "
        f"{feature_missing} missing values."
    )

print(
    "SUCCESS: Environmental features "
    "are ready."
)


# ============================================================
# 5. VALIDATE CROP MAPPING
# ============================================================

print("\n[5/10] Validating crop mapping...")

required_mapping_columns = [
    "crop",
    "crop_encoded"
]

for column in required_mapping_columns:

    if column not in crop_mapping.columns:

        raise ValueError(
            f"Missing crop mapping column: {column}"
        )


# Ensure crop encoding is integer

crop_mapping["crop_encoded"] = (
    crop_mapping["crop_encoded"]
    .astype(int)
)


print(
    f"Number of crops: "
    f"{len(crop_mapping)}"
)

print("\nCrop mapping:")

print(
    crop_mapping.to_string(index=False)
)


# ============================================================
# 6. CHECK MODEL FEATURE COUNT
# ============================================================

print(
    "\n[6/10] Checking model feature structure..."
)

expected_feature_count = (
    len(environmental_features) + 1
)

print(
    f"Environmental features: "
    f"{len(environmental_features)}"
)

print(
    "Crop feature: crop_encoded"
)

print(
    f"Expected model features: "
    f"{expected_feature_count}"
)

if hasattr(model, "n_features_in_"):

    print(
        f"Model expects: "
        f"{model.n_features_in_} features"
    )

    if (
        model.n_features_in_
        != expected_feature_count
    ):

        raise ValueError(
            "Feature count mismatch!\n"
            f"Model expects "
            f"{model.n_features_in_} features, "
            f"but recommendation engine "
            f"provides "
            f"{expected_feature_count}."
        )

print(
    "SUCCESS: Model feature count matches."
)


# ============================================================
# 7. GENERATE CROP PREDICTIONS
# ============================================================

print(
    "\n[7/10] Generating crop predictions..."
)

recommendations = []


for _, location in environmental.iterrows():

    location_id = location["location_id"]

    state = location["state"]

    lga = location["lga"]


    # --------------------------------------------------------
    # Environmental values
    # --------------------------------------------------------

    base_features = (
        location[
            environmental_features
        ]
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # Evaluate every crop
    # --------------------------------------------------------

    for _, crop_row in crop_mapping.iterrows():

        crop_name = crop_row["crop"]

        crop_encoded = int(
            crop_row["crop_encoded"]
        )


        # ----------------------------------------------------
        # Add crop encoding
        # ----------------------------------------------------

        model_features = np.append(
            base_features,
            crop_encoded
        )


        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        feature_names = (
            environmental_features
            + ["crop_encoded"]
        )

        X_input = pd.DataFrame(
            [model_features],
            columns=feature_names
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(X_input)[0]
        )


        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = (
                model.predict_proba(X_input)[0]
            )

            classes = model.classes_

            if 1 in classes:

                suitable_index = (
                    list(classes).index(1)
                )

                suitable_probability = float(
                    probabilities[
                        suitable_index
                    ]
                )

            else:

                suitable_probability = 0.0

        else:

            suitable_probability = float(
                prediction
            )


        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        recommendations.append({

            "location_id":
                location_id,

            "state":
                state,

            "lga":
                lga,

            "crop":
                crop_name,

            "crop_encoded":
                crop_encoded,

            "prediction":
                prediction,

            "suitability_probability":
                suitable_probability,

            "suitability_label":
                (
                    "Suitable"
                    if prediction == 1
                    else "Not Suitable"
                )

        })


print(
    "SUCCESS: Crop predictions generated."
)


# ============================================================
# 8. CREATE AND RANK RECOMMENDATIONS
# ============================================================

print(
    "\n[8/10] Ranking crop recommendations..."
)

recommendations_df = pd.DataFrame(
    recommendations
)


expected_records = (
    len(environmental)
    * len(crop_mapping)
)

actual_records = (
    len(recommendations_df)
)

print(
    f"Expected predictions: "
    f"{expected_records}"
)

print(
    f"Generated predictions: "
    f"{actual_records}"
)

if actual_records != expected_records:

    raise ValueError(
        "Unexpected number of predictions."
    )


# ------------------------------------------------------------
# Rank crops within each location
# ------------------------------------------------------------

recommendations_df[
    "recommendation_rank"
] = (

    recommendations_df
    .groupby("location_id")[
        "suitability_probability"
    ]
    .rank(
        method="first",
        ascending=False
    )
    .astype(int)
)


# ------------------------------------------------------------
# Recommendation category
# ------------------------------------------------------------

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


recommendations_df[
    "recommendation_category"
] = (

    recommendations_df[
        "suitability_probability"
    ]
    .apply(
        recommendation_category
    )
)


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

recommendations_df = (
    recommendations_df
    .sort_values(
        [
            "location_id",
            "recommendation_rank"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 9. CREATE TOP 3 RECOMMENDATIONS
# ============================================================

print(
    "\n[9/10] Creating top recommendations..."
)

top_recommendations = (

    recommendations_df
    .groupby(
        [
            "location_id",
            "state",
            "lga"
        ],
        as_index=False
    )
    .head(3)
    .reset_index(drop=True)
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

print(
    "\n[10/10] Saving recommendation datasets..."
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Full recommendation dataset

recommendations_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# Top recommendations

top_recommendations.to_csv(
    TOP_OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(
    "CROP RECOMMENDATION ENGINE "
    "COMPLETED SUCCESSFULLY"
)
print("=" * 70)

print(
    f"Locations evaluated: "
    f"{len(environmental)}"
)

print(
    f"Crops evaluated: "
    f"{len(crop_mapping)}"
)

print(
    f"Total predictions: "
    f"{len(recommendations_df)}"
)

print(
    f"Suitable predictions: "
    f"{(recommendations_df['prediction'] == 1).sum()}"
)

print(
    f"Not suitable predictions: "
    f"{(recommendations_df['prediction'] == 0).sum()}"
)


# ------------------------------------------------------------
# Recommendation distribution
# ------------------------------------------------------------

print("\nRecommendation categories:")

print(
    recommendations_df[
        "recommendation_category"
    ]
    .value_counts()
)


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

print(
    "\nFull recommendation dataset:"
)

print(OUTPUT_FILE)

print(
    "\nTop 3 recommendations per location:"
)

print(TOP_OUTPUT_FILE)


# ============================================================
# SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TOP CROP RECOMMENDATIONS")
print("=" * 70)

sample_columns = [

    "location_id",
    "state",
    "lga",
    "crop",
    "suitability_probability",
    "recommendation_rank",
    "recommendation_category"

]

print(
    top_recommendations[
        sample_columns
    ]
    .head(30)
    .to_string(index=False)
)

print("\n" + "=" * 70)