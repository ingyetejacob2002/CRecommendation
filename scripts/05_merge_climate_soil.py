from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

# IMPORTANT:
# Use the cleaned climate dataset, not the original dataset.
CLIMATE_FILE = PROCESSED_DIR / "north_central_climate_clean.csv"

SOIL_FILE = PROCESSED_DIR / "north_central_soil.csv"

OUTPUT_FILE = (
    PROCESSED_DIR /
    "north_central_environmental.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MERGING CLIMATE + SOIL DATA")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1/7] Checking input files...")

if not CLIMATE_FILE.exists():
    raise FileNotFoundError(
        f"Clean climate file not found:\n{CLIMATE_FILE}\n\n"
        "Make sure you have run:\n"
        "05a_remove_climate_duplicates.py"
    )

if not SOIL_FILE.exists():
    raise FileNotFoundError(
        f"Soil file not found:\n{SOIL_FILE}"
    )

print(f"Climate file found:")
print(CLIMATE_FILE)

print(f"\nSoil file found:")
print(SOIL_FILE)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2/7] Loading datasets...")

climate = pd.read_csv(CLIMATE_FILE)
soil = pd.read_csv(SOIL_FILE)

print(f"Climate records: {len(climate)}")
print(f"Soil records: {len(soil)}")


# ============================================================
# 3. VALIDATE LOCATION IDs
# ============================================================

print("\n[3/7] Validating location IDs...")

print(
    f"Climate unique locations: "
    f"{climate['location_id'].nunique()}"
)

print(
    f"Soil unique locations: "
    f"{soil['location_id'].nunique()}"
)


# ------------------------------------------------------------
# Check climate duplicates
# ------------------------------------------------------------

climate_duplicates = climate[
    climate["location_id"].duplicated(keep=False)
]

if len(climate_duplicates) > 0:

    print("\nERROR: Duplicate climate location IDs found:")

    print(
        climate_duplicates[
            ["location_id", "state", "lga"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Climate dataset still contains duplicate location IDs."
    )


# ------------------------------------------------------------
# Check soil duplicates
# ------------------------------------------------------------

soil_duplicates = soil[
    soil["location_id"].duplicated(keep=False)
]

if len(soil_duplicates) > 0:

    print("\nERROR: Duplicate soil location IDs found:")

    print(
        soil_duplicates[
            ["location_id", "state", "lga"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Soil dataset contains duplicate location IDs."
    )


print("\nSUCCESS!")
print("Climate location IDs are unique.")
print("Soil location IDs are unique.")


# ============================================================
# 4. CHECK LOCATION COVERAGE
# ============================================================

print("\n[4/7] Checking location coverage...")


climate_ids = set(
    climate["location_id"]
)

soil_ids = set(
    soil["location_id"]
)


# Locations with climate but no soil
climate_missing_soil = climate_ids - soil_ids

# Locations with soil but no climate
soil_missing_climate = soil_ids - climate_ids


if climate_missing_soil:

    print(
        "\nLocations with climate data "
        "but no soil data:"
    )

    print(
        sorted(climate_missing_soil)
    )


if soil_missing_climate:

    print(
        "\nLocations with soil data "
        "but no climate data:"
    )

    print(
        sorted(soil_missing_climate)
    )


if climate_missing_soil or soil_missing_climate:

    raise ValueError(
        "Climate and soil datasets do not contain "
        "the same location IDs."
    )


print(
    "Climate and soil location coverage matches."
)


# ============================================================
# 5. MERGE
# ============================================================

print("\n[5/7] Merging climate and soil data...")


environmental = pd.merge(
    climate,
    soil,
    on="location_id",
    how="inner",
    suffixes=("_climate", "_soil"),
    validate="one_to_one"
)


print(
    f"Merged records: {len(environmental)}"
)


# ============================================================
# 6. CHECK AND CLEAN MERGED DATASET
# ============================================================

print("\n[6/7] Checking merged dataset...")


# ------------------------------------------------------------
# Check state and LGA consistency
# ------------------------------------------------------------

if (
    "state_climate" in environmental.columns
    and "state_soil" in environmental.columns
):

    state_difference = (
        environmental["state_climate"]
        != environmental["state_soil"]
    ).sum()

    print(
        f"State differences: {state_difference}"
    )

    if state_difference > 0:

        print("\nWARNING:")
        print(
            "Some state names differ between "
            "climate and soil datasets."
        )


if (
    "lga_climate" in environmental.columns
    and "lga_soil" in environmental.columns
):

    lga_difference = (
        environmental["lga_climate"]
        != environmental["lga_soil"]
    ).sum()

    print(
        f"LGA differences: {lga_difference}"
    )

    if lga_difference > 0:

        print("\nWARNING:")
        print(
            "Some LGA names differ between "
            "climate and soil datasets."
        )


# ------------------------------------------------------------
# Keep climate state/LGA as the main location labels
# ------------------------------------------------------------

if "state_climate" in environmental.columns:

    environmental = environmental.rename(
        columns={
            "state_climate": "state",
            "lga_climate": "lga"
        }
    )


# ------------------------------------------------------------
# Remove duplicate soil location labels
# ------------------------------------------------------------

columns_to_drop = []

if "state_soil" in environmental.columns:
    columns_to_drop.append("state_soil")

if "lga_soil" in environmental.columns:
    columns_to_drop.append("lga_soil")


if columns_to_drop:

    environmental = environmental.drop(
        columns=columns_to_drop
    )


# ------------------------------------------------------------
# Check final duplicate location IDs
# ------------------------------------------------------------

duplicate_final = environmental[
    environmental["location_id"].duplicated(keep=False)
]


if len(duplicate_final) > 0:

    raise ValueError(
        "Duplicate location IDs exist in the merged dataset."
    )


print(
    "\nFinal location IDs are unique."
)


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

missing = environmental.isna().sum()

total_missing = missing.sum()


print(
    f"\nTotal missing values: {total_missing}"
)


if total_missing > 0:

    print("\nMissing values by column:")

    print(
        missing[
            missing > 0
        ]
    )

else:

    print(
        "SUCCESS: No missing values detected."
    )


# ============================================================
# 7. SAVE
# ============================================================

print("\n[7/7] Saving integrated dataset...")


environmental.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLIMATE + SOIL MERGE COMPLETED")
print("=" * 70)


print(
    f"Records: {len(environmental)}"
)

print(
    f"Columns: {len(environmental.columns)}"
)

print(
    f"Unique locations: "
    f"{environmental['location_id'].nunique()}"
)

print(
    f"Total missing values: "
    f"{environmental.isna().sum().sum()}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


print("\nFirst 5 records:")

print(
    environmental.head().to_string(index=False)
)


print("\nColumns:")

print(
    list(environmental.columns)
)


print("\n" + "=" * 70)
print("READY FOR NEXT DATA INTEGRATION STAGE")
print("=" * 70)