from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

ENVIRONMENTAL_FILE = (
    PROCESSED_DIR /
    "north_central_environmental.csv"
)

CROP_FILE = (
    PROCESSED_DIR /
    "north_central_crop_requirements.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "north_central_training_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# NASA POWER climate data covers 2015–2025.
# Therefore, the total precipitation is converted to
# mean annual precipitation before comparing it with
# annual crop rainfall requirements.

CLIMATE_YEARS = 11


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CREATING CROP RECOMMENDATION TRAINING DATASET")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print("\n[1/8] Checking input files...")

if not ENVIRONMENTAL_FILE.exists():
    raise FileNotFoundError(
        f"Environmental file not found:\n"
        f"{ENVIRONMENTAL_FILE}"
    )

if not CROP_FILE.exists():
    raise FileNotFoundError(
        f"Crop requirements file not found:\n"
        f"{CROP_FILE}"
    )

print(
    f"Environmental file found:\n"
    f"{ENVIRONMENTAL_FILE}"
)

print(
    f"\nCrop requirements file found:\n"
    f"{CROP_FILE}"
)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\n[2/8] Loading datasets...")

environmental = pd.read_csv(
    ENVIRONMENTAL_FILE
)

crops = pd.read_csv(
    CROP_FILE
)

print(
    f"Environmental records: "
    f"{len(environmental)}"
)

print(
    f"Crop records: "
    f"{len(crops)}"
)


# ============================================================
# 3. VALIDATE INPUT DATA
# ============================================================

print("\n[3/8] Validating input datasets...")

required_environmental_columns = [
    "location_id",
    "state",
    "lga",

    "mean_temperature_c",

    "total_precipitation_mm",

    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",

    "clay_percent",
    "sand_percent",
    "silt_percent",

    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]


required_crop_columns = [
    "crop",

    "temp_min_c",
    "temp_max_c",

    "rainfall_min_mm",
    "rainfall_max_mm",

    "ph_min",
    "ph_max",
]


# ------------------------------------------------------------
# Check environmental columns
# ------------------------------------------------------------

missing_environmental = [
    col
    for col in required_environmental_columns
    if col not in environmental.columns
]


# ------------------------------------------------------------
# Check crop columns
# ------------------------------------------------------------

missing_crop = [
    col
    for col in required_crop_columns
    if col not in crops.columns
]


# ------------------------------------------------------------
# Raise errors if columns are missing
# ------------------------------------------------------------

if missing_environmental:

    raise ValueError(
        "Missing environmental columns:\n"
        + "\n".join(missing_environmental)
    )


if missing_crop:

    raise ValueError(
        "Missing crop requirement columns:\n"
        + "\n".join(missing_crop)
    )


print("Environmental dataset: VALID")
print("Crop requirements dataset: VALID")


# ============================================================
# 4. CREATE ALL LGA × CROP COMBINATIONS
# ============================================================

print("\n[4/8] Creating LGA × crop combinations...")


# Temporary merge key

environmental["_merge_key"] = 1
crops["_merge_key"] = 1


# Cartesian product

training = pd.merge(
    environmental,
    crops,
    on="_merge_key",
    how="inner"
)


# Remove temporary key

training = training.drop(
    columns=["_merge_key"]
)


# Expected number of records

expected_records = (
    len(environmental) *
    len(crops)
)


print(
    f"Expected records: "
    f"{expected_records}"
)

print(
    f"Generated records: "
    f"{len(training)}"
)


if len(training) != expected_records:

    raise ValueError(
        "Unexpected number of LGA × crop combinations."
    )


# ============================================================
# 5. CALCULATE SUITABILITY FEATURES
# ============================================================

print("\n[5/8] Calculating suitability features...")


# ============================================================
# 5A. MEAN ANNUAL PRECIPITATION
# ============================================================

print(
    "\nCalculating mean annual precipitation..."
)

print(
    f"Climate period: "
    f"2015–2025 ({CLIMATE_YEARS} years)"
)


# IMPORTANT:
#
# total_precipitation_mm represents accumulated precipitation
# over the entire climate period.
#
# Crop rainfall requirements are annual requirements.
#
# Therefore:
#
# Mean annual rainfall =
# total precipitation / number of years
#

training["mean_annual_precipitation_mm"] = (
    training["total_precipitation_mm"]
    / CLIMATE_YEARS
)


print(
    "Mean annual precipitation calculated successfully."
)


# ============================================================
# 5B. TEMPERATURE SUITABILITY
# ============================================================

print("\nCalculating temperature suitability...")


training["temperature_suitable"] = (
    (
        training["mean_temperature_c"]
        >= training["temp_min_c"]
    )
    &
    (
        training["mean_temperature_c"]
        <= training["temp_max_c"]
    )
).astype(int)


# ============================================================
# 5C. RAINFALL SUITABILITY
# ============================================================

print("Calculating rainfall suitability...")


training["rainfall_suitable"] = (
    (
        training["mean_annual_precipitation_mm"]
        >= training["rainfall_min_mm"]
    )
    &
    (
        training["mean_annual_precipitation_mm"]
        <= training["rainfall_max_mm"]
    )
).astype(int)


# ============================================================
# 5D. SOIL pH SUITABILITY
# ============================================================

print("Calculating soil-pH suitability...")


training["ph_suitable"] = (
    (
        training["soil_ph"]
        >= training["ph_min"]
    )
    &
    (
        training["soil_ph"]
        <= training["ph_max"]
    )
).astype(int)


# ============================================================
# 6. CALCULATE OVERALL SUITABILITY
# ============================================================

print("\n[6/8] Calculating overall suitability...")


# ------------------------------------------------------------
# Environmental match score
# ------------------------------------------------------------

training["environmental_match_score"] = (
    training[
        [
            "temperature_suitable",
            "rainfall_suitable",
            "ph_suitable"
        ]
    ]
    .mean(axis=1)
)


# ------------------------------------------------------------
# Overall suitability
# ------------------------------------------------------------
#
# 1 = suitable
# 0 = unsuitable
#
# A crop is considered suitable when all three
# primary ecological requirements are satisfied.
#

training["suitable"] = (
    training["environmental_match_score"] == 1.0
).astype(int)


# ------------------------------------------------------------
# Human-readable label
# ------------------------------------------------------------

training["suitability_label"] = np.where(
    training["suitable"] == 1,
    "Suitable",
    "Not Suitable"
)


# ============================================================
# 6A. DISPLAY SUITABILITY DIAGNOSTICS
# ============================================================

print("\nSuitability diagnostics:")

print(
    "\nTemperature suitability:"
)

print(
    training["temperature_suitable"]
    .value_counts()
    .sort_index()
)


print(
    "\nRainfall suitability:"
)

print(
    training["rainfall_suitable"]
    .value_counts()
    .sort_index()
)


print(
    "\nSoil-pH suitability:"
)

print(
    training["ph_suitable"]
    .value_counts()
    .sort_index()
)


print(
    "\nOverall suitability:"
)

print(
    training["suitable"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. CLEAN AND REORDER COLUMNS
# ============================================================

print("\n[7/8] Cleaning training dataset...")


# ------------------------------------------------------------
# Remove duplicated coordinate columns
# ------------------------------------------------------------

if (
    "latitude_climate" in training.columns
    and
    "latitude_soil" in training.columns
):

    training["latitude"] = (
        training["latitude_climate"]
    )

    training["longitude"] = (
        training["longitude_climate"]
    )

    training = training.drop(
        columns=[
            "latitude_climate",
            "longitude_climate",
            "latitude_soil",
            "longitude_soil",
        ]
    )


# ------------------------------------------------------------
# COLUMN ORDER
# ------------------------------------------------------------

preferred_columns = [

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    "location_id",
    "state",
    "lga",

    "latitude",
    "longitude",


    # --------------------------------------------------------
    # Climate variables
    # --------------------------------------------------------

    "mean_daily_precipitation_mm",
    "std_daily_precipitation_mm",
    "min_daily_precipitation_mm",
    "max_daily_precipitation_mm",

    "total_precipitation_mm",

    # NEW FEATURE
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


    # --------------------------------------------------------
    # Soil variables
    # --------------------------------------------------------

    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",

    "clay_percent",
    "sand_percent",
    "silt_percent",

    "cec_cmolc_kg",
    "bulk_density_g_cm3",


    # --------------------------------------------------------
    # Crop requirements
    # --------------------------------------------------------

    "crop",

    "temp_min_c",
    "temp_max_c",

    "rainfall_min_mm",
    "rainfall_max_mm",

    "ph_min",
    "ph_max",


    # --------------------------------------------------------
    # Calculated suitability features
    # --------------------------------------------------------

    "temperature_suitable",
    "rainfall_suitable",
    "ph_suitable",

    "environmental_match_score",

    "suitable",

    "suitability_label",
]


# ------------------------------------------------------------
# Keep only columns that exist
# ------------------------------------------------------------

final_columns = [
    col
    for col in preferred_columns
    if col in training.columns
]


training = training[
    final_columns
]


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing = training.isna().sum()

total_missing = missing.sum()


print(
    f"\nTotal missing values: "
    f"{total_missing}"
)


if total_missing > 0:

    print(
        "\nMissing values:"
    )

    print(
        missing[
            missing > 0
        ]
    )

    raise ValueError(
        "Training dataset contains missing values."
    )


print(
    "SUCCESS: No missing values detected."
)


# ============================================================
# 7A. CHECK TARGET CLASS BALANCE
# ============================================================

print(
    "\nChecking target class balance..."
)


class_counts = (
    training["suitable"]
    .value_counts()
    .sort_index()
)


print(
    "\nTarget distribution:"
)

print(class_counts)


class_percentages = (
    training["suitable"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


print(
    "\nTarget percentages:"
)

print(
    class_percentages.round(2)
)


# ------------------------------------------------------------
# Ensure both classes exist
# ------------------------------------------------------------

unique_target_classes = (
    training["suitable"]
    .nunique()
)


if unique_target_classes < 2:

    raise ValueError(
        "\nCRITICAL ERROR:\n"
        "The suitability target contains only one class.\n"
        "The dataset cannot be used for binary classification "
        "until both suitable and unsuitable records exist."
    )


print(
    "\nSUCCESS:"
)

print(
    "Both suitable (1) and unsuitable (0) "
    "classes are present."
)


# ============================================================
# 8. SAVE DATASET
# ============================================================

print(
    "\n[8/8] Saving training dataset..."
)


training.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRAINING DATASET CREATED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    f"Total records: "
    f"{len(training)}"
)


print(
    f"Unique LGAs: "
    f"{training['location_id'].nunique()}"
)


print(
    f"Unique crops: "
    f"{training['crop'].nunique()}"
)


print(
    f"Suitable records: "
    f"{training['suitable'].sum()}"
)


print(
    f"Not suitable records: "
    f"{(training['suitable'] == 0).sum()}"
)


print(
    "\nSuitability distribution:"
)


print(
    training["suitability_label"]
    .value_counts()
)


print(
    "\nSuitability percentages:"
)


print(
    (
        training["suitability_label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print(
    "\nCrop distribution:"
)


print(
    training["crop"]
    .value_counts()
)


print(
    "\nMean annual rainfall statistics:"
)


print(
    training[
        "mean_annual_precipitation_mm"
    ].describe()
)


print(
    "\nFirst 10 records:"
)


print(
    training
    .head(10)
    .to_string(index=False)
)


print(
    "\nColumns:"
)


print(
    list(training.columns)
)


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)


print(
    "\n" + "=" * 70
)