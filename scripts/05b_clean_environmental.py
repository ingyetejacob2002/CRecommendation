from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "north_central_environmental.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "north_central_environmental_clean.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CLEANING INTEGRATED ENVIRONMENTAL DATASET")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading environmental dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Environmental dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Records: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. CHECK LOCATION IDs
# ============================================================

print("\n[2/6] Validating location IDs...")

print(
    f"Unique location IDs: "
    f"{df['location_id'].nunique()}"
)

if df["location_id"].duplicated().any():

    raise ValueError(
        "Duplicate location IDs found."
    )

print("Location IDs are unique.")


# ============================================================
# 3. CHECK COORDINATES
# ============================================================

print("\n[3/6] Checking coordinates...")

coordinate_columns = [
    "latitude_climate",
    "longitude_climate",
    "latitude_soil",
    "longitude_soil"
]

for column in coordinate_columns:

    if column not in df.columns:
        raise ValueError(
            f"Expected coordinate column missing: {column}"
        )


# Check whether climate and soil coordinates agree

latitude_difference = (
    df["latitude_climate"]
    - df["latitude_soil"]
).abs()

longitude_difference = (
    df["longitude_climate"]
    - df["longitude_soil"]
).abs()


print(
    f"Maximum latitude difference: "
    f"{latitude_difference.max()}"
)

print(
    f"Maximum longitude difference: "
    f"{longitude_difference.max()}"
)


if (
    latitude_difference > 0.000001
).any():

    raise ValueError(
        "Climate and soil latitude coordinates differ."
    )


if (
    longitude_difference > 0.000001
).any():

    raise ValueError(
        "Climate and soil longitude coordinates differ."
    )


print(
    "Climate and soil coordinates agree."
)


# ============================================================
# 4. KEEP ONE SET OF COORDINATES
# ============================================================

print("\n[4/6] Removing duplicate coordinate columns...")

# Use climate coordinates as the main coordinates

df = df.rename(
    columns={
        "latitude_climate": "latitude",
        "longitude_climate": "longitude"
    }
)

# Remove duplicate soil coordinates

df = df.drop(
    columns=[
        "latitude_soil",
        "longitude_soil"
    ]
)


print("Coordinate columns cleaned.")


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n[5/6] Checking missing values...")

missing = df.isna().sum()

total_missing = missing.sum()

print(
    f"Total missing values: {total_missing}"
)

if total_missing > 0:

    print("\nMissing values:")

    print(
        missing[
            missing > 0
        ]
    )

    raise ValueError(
        "Environmental dataset contains missing values."
    )

print(
    "SUCCESS: No missing values."
)


# ============================================================
# 6. SAVE
# ============================================================

print("\n[6/6] Saving cleaned environmental dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ENVIRONMENTAL DATASET CLEANING COMPLETED")
print("=" * 70)

print(
    f"Records: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"Unique locations: "
    f"{df['location_id'].nunique()}"
)

print(
    f"Missing values: "
    f"{df.isna().sum().sum()}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print("\nFinal columns:")

print(
    list(df.columns)
)

print("\nFirst 5 records:")

print(
    df.head().to_string(index=False)
)

print("\n" + "=" * 70)