from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

SOIL_FILE = PROCESSED_DIR / "north_central_soil.csv"
OUTPUT_FILE = PROCESSED_DIR / "north_central_soil.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# Number of nearest valid LGAs used for spatial imputation
N_NEIGHBORS = 5

SOIL_COLUMNS = [
    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "clay_percent",
    "sand_percent",
    "silt_percent",
    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]

ID_COLUMNS = [
    "location_id",
    "state",
    "lga",
    "latitude",
    "longitude",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SPATIAL IMPUTATION OF MISSING SOIL DATA")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/6] Loading soil dataset...")

if not SOIL_FILE.exists():
    raise FileNotFoundError(
        f"\nSoil file not found:\n{SOIL_FILE}"
    )

df = pd.read_csv(SOIL_FILE)

print(f"Total records: {len(df)}")
print(f"Unique locations: {df['location_id'].nunique()}")


# ============================================================
# 2. IDENTIFY INCOMPLETE RECORDS
# ============================================================

print("\n[2/6] Identifying incomplete soil records...")

missing_mask = df[SOIL_COLUMNS].isna().any(axis=1)

incomplete = df.loc[
    missing_mask,
    ID_COLUMNS
].copy()

print(f"Incomplete records: {len(incomplete)}")

if len(incomplete) == 0:
    print("\nNo missing soil records found.")
    print("Nothing to impute.")
    raise SystemExit


print("\nRecords requiring imputation:")
print(
    incomplete[
        ["location_id", "state", "lga"]
    ].to_string(index=False)
)


# ============================================================
# 3. PREPARE VALID RECORDS
# ============================================================

print("\n[3/6] Preparing valid soil records...")

valid_mask = ~missing_mask

valid_df = df.loc[valid_mask].copy()

print(f"Valid records available: {len(valid_df)}")

if len(valid_df) < N_NEIGHBORS:
    raise RuntimeError(
        "Not enough valid soil records for spatial imputation."
    )


# ============================================================
# 4. FIND NEAREST NEIGHBORS
# ============================================================

print("\n[4/6] Finding nearest valid LGAs...")

# Coordinates
valid_coordinates = valid_df[
    ["latitude", "longitude"]
].values

missing_coordinates = incomplete[
    ["latitude", "longitude"]
].values


# ------------------------------------------------------------
# Nearest-neighbor model
# ------------------------------------------------------------

knn = NearestNeighbors(
    n_neighbors=N_NEIGHBORS,
    metric="euclidean"
)

knn.fit(valid_coordinates)

distances, indices = knn.kneighbors(
    missing_coordinates
)


# ============================================================
# 5. IMPUTE MISSING VALUES
# ============================================================

print("\n[5/6] Performing spatial imputation...")


for row_number, (
    original_index,
    distance_row,
    index_row
) in enumerate(
    zip(
        incomplete.index,
        distances,
        indices
    ),
    start=1
):

    target = df.loc[original_index]

    print("\n" + "-" * 60)
    print(
        f"Location {row_number}/{len(incomplete)}"
    )

    print(
        f"Target: {target['state']} - {target['lga']}"
    )

    print(
        f"Location ID: {target['location_id']}"
    )

    print(
        f"Coordinates: "
        f"{target['latitude']}, "
        f"{target['longitude']}"
    )


    # --------------------------------------------------------
    # Get nearest valid locations
    # --------------------------------------------------------

    neighbors = valid_df.iloc[index_row].copy()

    print("\nNearest valid LGAs:")

    print(
        neighbors[
            ["location_id", "state", "lga",
             "latitude", "longitude"]
        ].to_string(index=False)
    )


    # --------------------------------------------------------
    # Distance-weighted averaging
    # --------------------------------------------------------

    # Prevent division by zero if two locations have
    # identical coordinates.
    safe_distances = np.maximum(
        distance_row,
        1e-8
    )

    weights = 1 / safe_distances

    weights = weights / weights.sum()


    # --------------------------------------------------------
    # Impute each soil variable
    # --------------------------------------------------------

    for column in SOIL_COLUMNS:

        # Only impute if the target value is missing
        if pd.isna(df.loc[original_index, column]):

            values = neighbors[column].astype(float).values

            # Safety check
            valid_values = ~np.isnan(values)

            if not valid_values.any():
                continue

            column_weights = weights[valid_values]
            column_values = values[valid_values]

            column_weights = (
                column_weights /
                column_weights.sum()
            )

            estimated_value = np.sum(
                column_values * column_weights
            )

            df.loc[
                original_index,
                column
            ] = estimated_value

            print(
                f"  {column}: "
                f"{estimated_value:.4f}"
            )


# ============================================================
# 6. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


remaining_missing = df[
    SOIL_COLUMNS
].isna().sum()


print("\nMissing values by column:")

print(remaining_missing)


total_missing = remaining_missing.sum()


print(
    f"\nTotal remaining missing soil values: "
    f"{total_missing}"
)


print(
    f"Total records: {len(df)}"
)

print(
    f"Unique location IDs: "
    f"{df['location_id'].nunique()}"
)


# ============================================================
# SAVE
# ============================================================

if total_missing > 0:

    print(
        "\nWARNING: Some missing values remain."
    )

else:

    print(
        "\nSUCCESS: All soil variables are complete."
    )


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved soil dataset to:\n{OUTPUT_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SOIL IMPUTATION COMPLETED")
print("=" * 70)

print(
    f"Records: {len(df)}"
)

print(
    f"Unique locations: "
    f"{df['location_id'].nunique()}"
)

print(
    f"Remaining missing values: "
    f"{total_missing}"
)

print("=" * 70)