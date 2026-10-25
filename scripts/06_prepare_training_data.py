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

TRAINING_FILE = (
    PROCESSED_DIR /
    "north_central_training_dataset.csv"
)

X_TRAIN_FILE = (
    PROCESSED_DIR /
    "X_train.csv"
)

X_TEST_FILE = (
    PROCESSED_DIR /
    "X_test.csv"
)

Y_TRAIN_FILE = (
    PROCESSED_DIR /
    "y_train.csv"
)

Y_TEST_FILE = (
    PROCESSED_DIR /
    "y_test.csv"
)

LABEL_ENCODER_FILE = (
    PROCESSED_DIR /
    "crop_label_mapping.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PREPARING TRAINING DATA")
print("=" * 70)

# ============================================================
# 1. CHECK INPUT FILE
# ============================================================

print("\n[1/10] Checking training dataset...")

if not TRAINING_FILE.exists():
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAINING_FILE}"
    )

print(f"Training file found:\n{TRAINING_FILE}")

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2/10] Loading training dataset...")

df = pd.read_csv(TRAINING_FILE)

print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# ============================================================
# 3. BASIC VALIDATION
# ============================================================

print("\n[3/10] Validating dataset...")

print(f"Duplicate rows: {df.duplicated().sum()}")

if df.duplicated().sum() > 0:
    print("WARNING: Duplicate rows detected.")

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
# 4. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n[4/10] Checking suitability distribution...")

if "suitable" not in df.columns:
    raise ValueError(
        "Target column 'suitable' was not found."
    )

print("\nSuitable distribution:")

print(
    df["suitable"]
    .value_counts()
    .sort_index()
)

print("\nSuitability percentages:")

print(
    (
        df["suitable"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

# ============================================================
# 5. CHECK SUITABILITY LABEL
# ============================================================

print("\n[5/10] Checking suitability labels...")

if "suitability_label" in df.columns:

    print(
        df["suitability_label"]
        .value_counts()
    )

# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n[6/10] Checking missing values...")

missing = df.isna().sum()

total_missing = missing.sum()

print(f"Total missing values: {total_missing}")

if total_missing > 0:

    print("\nColumns containing missing values:")

    print(
        missing[missing > 0]
    )

    raise ValueError(
        "Missing values detected. "
        "Resolve them before model training."
    )

print("SUCCESS: No missing values.")

# ============================================================
# 7. DEFINE FEATURES
# ============================================================

print("\n[7/10] Preparing model features...")

# ------------------------------------------------------------
# Columns that should NOT be used as predictors
# ------------------------------------------------------------

DROP_COLUMNS = [

    # Location identifiers
    "location_id",

    # Administrative/location text
    "state",
    "lga",

    # Crop name
    "crop",

    # Crop requirement information
    "temp_min_c",
    "temp_max_c",
    "rainfall_min_mm",
    "rainfall_max_mm",
    "ph_min",
    "ph_max",

    # Rule-based suitability outputs
    "temperature_suitable",
    "rainfall_suitable",
    "ph_suitable",
    "environmental_match_score",
    "suitability_label",

    # Target
    "suitable",
]

# Keep only columns that actually exist
DROP_COLUMNS = [
    column
    for column in DROP_COLUMNS
    if column in df.columns
]

X = df.drop(
    columns=DROP_COLUMNS
)

y = df["suitable"]

print("\nFeatures selected:")

print(list(X.columns))

print(
    f"\nNumber of features: {X.shape[1]}"
)

# ============================================================
# 8. VALIDATE FEATURE TYPES
# ============================================================

print("\n[8/10] Checking feature data types...")

non_numeric = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

if non_numeric:

    print(
        "\nNon-numeric features found:"
    )

    print(non_numeric)

    raise ValueError(
        "All model features must be numeric."
    )

print(
    "SUCCESS: All model features are numeric."
)

# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

print("\n[9/10] Splitting dataset...")

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# We stratify using the suitability target so that the
# proportion of suitable/unsuitable samples is maintained
# in both training and testing datasets.
# ------------------------------------------------------------

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
# 10. SAVE PREPARED DATA
# ============================================================

print("\n[10/10] Saving prepared datasets...")

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
    index=False
)

y_test.to_csv(
    Y_TEST_FILE,
    index=False
)

# ============================================================
# SAVE TARGET MAPPING
# ============================================================

mapping = pd.DataFrame({
    "original_value": sorted(
        df["suitable"].unique()
    )
})

mapping["description"] = mapping[
    "original_value"
].map({
    0: "Unsuitable",
    1: "Suitable"
})

mapping.to_csv(
    LABEL_ENCODER_FILE,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING DATA PREPARATION COMPLETED")
print("=" * 70)

print(
    f"Original records: {len(df)}"
)

print(
    f"Features: {X.shape[1]}"
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

print("\nFiles saved:")

print(X_TRAIN_FILE)
print(X_TEST_FILE)
print(Y_TRAIN_FILE)
print(Y_TEST_FILE)
print(LABEL_ENCODER_FILE)

print("\n" + "=" * 70)