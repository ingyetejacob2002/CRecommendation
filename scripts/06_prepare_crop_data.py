from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

CROP_FILE = PROCESSED_DIR / "north_central_crop_requirements.csv"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PREPARING CROP REQUIREMENTS DATASET")
print("=" * 70)


# ============================================================
# CROP DATA
# ============================================================
#
# IMPORTANT:
# The values in this file should come from a documented
# agricultural/ecological source such as FAO ECOCROP.
#
# Do not invent crop requirements.
#
# This script assumes that the crop requirements CSV has
# already been prepared.
# ============================================================


REQUIRED_COLUMNS = [
    "crop",
    "temp_min_c",
    "temp_max_c",
    "rainfall_min_mm",
    "rainfall_max_mm",
    "ph_min",
    "ph_max",
]


# ============================================================
# 1. CHECK FILE
# ============================================================

print("\n[1/6] Checking crop requirements file...")

if not CROP_FILE.exists():
    raise FileNotFoundError(
        f"""
Crop requirements file not found:

{CROP_FILE}

Create the file:

data/processed/north_central_crop_requirements.csv
"""
    )

print(f"Crop file found:\n{CROP_FILE}")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2/6] Loading crop requirements...")

crops = pd.read_csv(CROP_FILE)

print(f"Crop records: {len(crops)}")


# ============================================================
# 3. CHECK COLUMNS
# ============================================================

print("\n[3/6] Checking required columns...")

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in crops.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )

print("All required columns are present.")


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\n[4/6] Checking duplicate crops...")

duplicate_crops = crops[
    crops["crop"].duplicated(keep=False)
]

if len(duplicate_crops) > 0:

    print("\nWARNING: Duplicate crop names found:")

    print(
        duplicate_crops[
            ["crop"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Crop dataset contains duplicate crop names."
    )

print("Crop names are unique.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n[5/6] Checking missing values...")

missing = crops.isna().sum()

total_missing = missing.sum()

print(f"Total missing values: {total_missing}")

if total_missing > 0:

    print("\nMissing values by column:")

    print(
        missing[
            missing > 0
        ]
    )

    raise ValueError(
        "Crop dataset contains missing values."
    )

print("SUCCESS: No missing values detected.")


# ============================================================
# CHECK NUMERICAL RANGES
# ============================================================

print("\nChecking crop requirement ranges...")

if (crops["temp_min_c"] >= crops["temp_max_c"]).any():
    raise ValueError(
        "Some crops have invalid temperature ranges."
    )

if (crops["rainfall_min_mm"] >= crops["rainfall_max_mm"]).any():
    raise ValueError(
        "Some crops have invalid rainfall ranges."
    )

if (crops["ph_min"] >= crops["ph_max"]).any():
    raise ValueError(
        "Some crops have invalid soil-pH ranges."
    )

print("Temperature ranges: valid")
print("Rainfall ranges: valid")
print("Soil-pH ranges: valid")


# ============================================================
# 6. SAVE
# ============================================================

print("\n[6/6] Saving crop requirements dataset...")

crops.to_csv(
    CROP_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CROP REQUIREMENTS DATASET COMPLETED")
print("=" * 70)

print(f"Total crops: {len(crops)}")

print("\nCrops:")

print(
    crops["crop"].tolist()
)

print("\nColumns:")

print(
    list(crops.columns)
)

print("\nDataset:")

print(
    crops.to_string(index=False)
)

print(f"\nSaved to:\n{CROP_FILE}")

print("\n" + "=" * 70)