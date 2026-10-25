from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "north_central_training_dataset.csv"
)

X_TRAIN_FILE = PROCESSED_DIR / "X_train.csv"
X_TEST_FILE = PROCESSED_DIR / "X_test.csv"
Y_TRAIN_FILE = PROCESSED_DIR / "y_train.csv"
Y_TEST_FILE = PROCESSED_DIR / "y_test.csv"

CROP_MAPPING_FILE = (
    PROCESSED_DIR /
    "crop_label_mapping.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PREPARING MACHINE LEARNING DATA")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILE
# ============================================================

print("\n[1/10] Checking input dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"""
Training dataset not found:

{INPUT_FILE}

Make sure the training dataset has been created first.
"""
    )

print(f"Training dataset found:\n{INPUT_FILE}")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[2/10] Loading training dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

print("\n[3/10] Validating dataset...")

required_columns = [
    "crop",
    "suitable"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

print("Required columns are present.")

print(
    f"Duplicate rows: {df.duplicated().sum()}"
)

print(
    f"Unique locations: "
    f"{df['location_id'].nunique()}"
)

print(
    f"Unique crops: "
    f"{df['crop'].nunique()}"
)

print("\nCrop distribution:")

print(
    df["crop"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. CHECK TARGET
# ============================================================

print("\n[4/10] Checking target variable...")

print("Target: suitable")

print("\nTarget distribution:")

target_distribution = df["suitable"].value_counts().sort_index()

print(target_distribution)

print("\nTarget percentages:")

print(
    (df["suitable"].value_counts(normalize=True) * 100)
    .round(2)
)

# Make sure both classes exist

unique_targets = sorted(
    df["suitable"].dropna().unique()
)

if len(unique_targets) != 2:
    raise ValueError(
        f"""
Expected binary target with classes 0 and 1.

Found:
{unique_targets}

The model requires both suitable and unsuitable
classes.
"""
    )

if set(unique_targets) != {0, 1}:
    raise ValueError(
        f"""
Target must contain exactly 0 and 1.

Found:
{unique_targets}
"""
    )

print("\nSUCCESS: Both target classes are present.")


# ============================================================
# 5. SELECT ENVIRONMENTAL FEATURES
# ============================================================

print("\n[5/10] Selecting machine learning features...")


# These are genuine environmental variables.
#
# IMPORTANT:
# We intentionally DO NOT include:
#
# temp_min_c
# temp_max_c
# rainfall_min_mm
# rainfall_max_mm
# ph_min
# ph_max
# temperature_suitable
# rainfall_suitable
# ph_suitable
# environmental_match_score
# suitable
# suitability_label
#
# Those variables either describe the crop requirements
# or are directly used to calculate the target.
#
# Including them would cause DATA LEAKAGE.

environmental_features = [

    # Location
    "latitude",
    "longitude",

    # Rainfall
    "mean_daily_precipitation_mm",
    "std_daily_precipitation_mm",
    "min_daily_precipitation_mm",
    "max_daily_precipitation_mm",
    "total_precipitation_mm",
    "mean_annual_precipitation_mm",

    # Temperature
    "mean_temperature_c",
    "std_temperature_c",
    "min_temperature_c",
    "max_temperature_c",
    "mean_max_temperature_c",
    "absolute_max_temperature_c",
    "mean_min_temperature_c",
    "absolute_min_temperature_c",

    # Humidity
    "mean_relative_humidity_percent",
    "minimum_relative_humidity_percent",
    "maximum_relative_humidity_percent",

    # Solar radiation
    "mean_solar_radiation",
    "minimum_solar_radiation",
    "maximum_solar_radiation",

    # Wind
    "mean_wind_speed",
    "minimum_wind_speed",
    "maximum_wind_speed",

    # Soil
    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "clay_percent",
    "sand_percent",
    "silt_percent",
    "cec_cmolc_kg",
    "bulk_density_g_cm3"
]


# Check that all environmental features exist

missing_features = [
    feature
    for feature in environmental_features
    if feature not in df.columns
]

if missing_features:

    print("\nMissing environmental features:")

    for feature in missing_features:
        print(f" - {feature}")

    raise ValueError(
        "One or more required environmental features "
        "are missing from the dataset."
    )


print("\nEnvironmental features selected:")

for feature in environmental_features:
    print(f" - {feature}")

print(
    f"\nNumber of environmental features: "
    f"{len(environmental_features)}"
)


# ============================================================
# 6. ENCODE CROP
# ============================================================

print("\n[6/10] Encoding crop variable...")

print("\nOriginal crops:")

crops = sorted(
    df["crop"]
    .astype(str)
    .unique()
)

for crop in crops:
    print(f" - {crop}")


crop_encoder = LabelEncoder()

df["crop_encoded"] = crop_encoder.fit_transform(
    df["crop"].astype(str)
)


# Create mapping table

crop_mapping = pd.DataFrame({
    "crop": crop_encoder.classes_,
    "crop_encoded": range(
        len(crop_encoder.classes_)
    )
})


crop_mapping.to_csv(
    CROP_MAPPING_FILE,
    index=False
)


print("\nCrop encoding:")

print(crop_mapping.to_string(index=False))

print(
    f"\nCrop mapping saved to:\n"
    f"{CROP_MAPPING_FILE}"
)


# ============================================================
# 7. CREATE X AND y
# ============================================================

print("\n[7/10] Creating model features and target...")


# Environmental variables + encoded crop

feature_columns = (
    environmental_features +
    ["crop_encoded"]
)

X = df[feature_columns].copy()

y = df["suitable"].astype(int).copy()


print(
    f"Total model features: {X.shape[1]}"
)

print(
    f"Total samples: {len(X)}"
)

print("\nFinal model features:")

for i, feature in enumerate(
    feature_columns,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# 8. CHECK DATA QUALITY
# ============================================================

print("\n[8/10] Checking feature data quality...")


# Check missing values

missing_features_count = X.isna().sum().sum()

if missing_features_count > 0:

    print(
        "\nMissing values found:"
    )

    print(
        X.isna()
        .sum()
        [X.isna().sum() > 0]
    )

    raise ValueError(
        "Missing values exist in model features."
    )

print("SUCCESS: No missing feature values.")


# Check infinite values

infinite_count = np.isinf(
    X.select_dtypes(include=np.number)
).sum().sum()

if infinite_count > 0:

    raise ValueError(
        f"Dataset contains {infinite_count} "
        "infinite values."
    )

print("SUCCESS: No infinite values.")


# Check numeric types

non_numeric = X.select_dtypes(
    exclude=np.number
).columns.tolist()

if non_numeric:

    raise ValueError(
        f"Non-numeric model features found:\n"
        f"{non_numeric}"
    )

print(
    "SUCCESS: All model features are numeric."
)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\n[9/10] Splitting dataset...")


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)

print(
    f"Training percentage: "
    f"{len(X_train) / len(X) * 100:.2f}%"
)

print(
    f"Testing percentage: "
    f"{len(X_test) / len(X) * 100:.2f}%"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTraining target distribution:")

print(
    y_train.value_counts()
    .sort_index()
)


print("\nTesting target distribution:")

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 10. SAVE DATASETS
# ============================================================

print("\n[10/10] Saving machine learning datasets...")


X_train.to_csv(
    X_TRAIN_FILE,
    index=False
)

X_test.to_csv(
    X_TEST_FILE,
    index=False
)

y_train.to_csv(
    Y_TRAIN_FILE,
    index=False,
    header=True
)

y_test.to_csv(
    Y_TEST_FILE,
    index=False,
    header=True
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING DATA PREPARATION COMPLETED")
print("=" * 70)

print(
    f"Original records: {len(df)}"
)

print(
    f"Model features: {X.shape[1]}"
)

print(
    f"Training records: {len(X_train)}"
)

print(
    f"Testing records: {len(X_test)}"
)

print(
    f"Training features: {X_train.shape[1]}"
)

print(
    f"Testing features: {X_test.shape[1]}"
)

print(
    f"Target classes: {sorted(y.unique())}"
)

print("\nFiles saved:")

print(
    f"X_train:\n{X_TRAIN_FILE}"
)

print(
    f"X_test:\n{X_TEST_FILE}"
)

print(
    f"y_train:\n{Y_TRAIN_FILE}"
)

print(
    f"y_test:\n{Y_TEST_FILE}"
)

print(
    f"Crop mapping:\n{CROP_MAPPING_FILE}"
)

print("\n" + "=" * 70)