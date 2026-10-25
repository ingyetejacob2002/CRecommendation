from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLIMATE_FILE = PROCESSED_DIR / "north_central_climate.csv"
OUTPUT_FILE = PROCESSED_DIR / "north_central_climate_clean.csv"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("REMOVING SAFE CLIMATE DUPLICATES")
print("=" * 70)


# ============================================================
# [1] LOAD DATA
# ============================================================

print("\n[1] Loading climate dataset...")

if not CLIMATE_FILE.exists():
    raise FileNotFoundError(
        f"Climate dataset not found:\n{CLIMATE_FILE}"
    )

climate = pd.read_csv(CLIMATE_FILE)

print(f"Original records: {len(climate)}")
print(f"Original unique locations: {climate['location_id'].nunique()}")


# ============================================================
# [2] CHECK DUPLICATES
# ============================================================

print("\n[2] Checking duplicate location IDs...")

duplicate_mask = climate.duplicated(
    subset="location_id",
    keep=False
)

duplicates = climate[duplicate_mask]

print(f"Duplicate rows: {len(duplicates)}")
print(
    f"Duplicate location IDs: "
    f"{duplicates['location_id'].nunique()}"
)


# ============================================================
# [3] VERIFY DUPLICATES HAVE IDENTICAL CLIMATE VALUES
# ============================================================

print("\n[3] Verifying duplicate climate values...")

climate_columns = [
    column
    for column in climate.columns
    if column not in ["location_id", "state", "lga"]
]

unsafe_duplicates = []

for location_id, group in climate.groupby("location_id"):

    if len(group) > 1:

        # Check whether all climate variables are identical
        if not group[climate_columns].nunique().eq(1).all():

            unsafe_duplicates.append(location_id)


if unsafe_duplicates:

    print("\nERROR!")
    print("Some duplicate location IDs have different climate values.")

    print("\nUnsafe duplicate location IDs:")
    print(unsafe_duplicates)

    raise ValueError(
        "Duplicate climate records are not identical. "
        "Do NOT automatically remove them."
    )

else:

    print(
        "SUCCESS: All duplicate location IDs "
        "contain identical climate values."
    )


# ============================================================
# [4] REMOVE DUPLICATES
# ============================================================

print("\n[4] Removing duplicate records...")

climate_clean = climate.drop_duplicates(
    subset="location_id",
    keep="first"
).copy()

print(f"Records after cleaning: {len(climate_clean)}")
print(
    f"Unique locations after cleaning: "
    f"{climate_clean['location_id'].nunique()}"
)


# ============================================================
# [5] FINAL VALIDATION
# ============================================================

print("\n[5] Final validation...")

if climate_clean["location_id"].duplicated().any():

    raise ValueError(
        "Duplicate location IDs still exist after cleaning."
    )

if len(climate_clean) != climate_clean["location_id"].nunique():

    raise ValueError(
        "Number of records does not match number of unique locations."
    )

print("No duplicate location IDs remain.")


# ============================================================
# [6] SAVE
# ============================================================

print("\n[6] Saving cleaned climate dataset...")

climate_clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to:")
print(OUTPUT_FILE)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLIMATE DATA CLEANING COMPLETED")
print("=" * 70)

print(f"Original records:       {len(climate)}")
print(f"Clean records:          {len(climate_clean)}")
print(
    f"Records removed:        "
    f"{len(climate) - len(climate_clean)}"
)
print(
    f"Unique locations:       "
    f"{climate_clean['location_id'].nunique()}"
)

print("\nSUCCESS!")
print("Climate dataset is now ready for merging with soil data.")

print("=" * 70)