from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLIMATE_FILE = (
    PROCESSED_DIR /
    "north_central_climate.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CLIMATE DATA DUPLICATE DIAGNOSTIC")
print("=" * 70)

print("\n[1] Loading climate dataset...")

if not CLIMATE_FILE.exists():
    raise FileNotFoundError(
        f"Climate file not found:\n{CLIMATE_FILE}"
    )

df = pd.read_csv(CLIMATE_FILE)

print(f"Total records: {len(df)}")
print(
    f"Unique location IDs: "
    f"{df['location_id'].nunique()}"
)


# ============================================================
# FIND DUPLICATES
# ============================================================

print("\n[2] Finding duplicate location IDs...")

duplicate_mask = df["location_id"].duplicated(
    keep=False
)

duplicates = df.loc[
    duplicate_mask
].copy()

print(
    f"Duplicate rows: {len(duplicates)}"
)

print(
    f"Duplicate location IDs: "
    f"{duplicates['location_id'].nunique()}"
)


# ============================================================
# DISPLAY DUPLICATES
# ============================================================

print("\n[3] Duplicate locations:")

print(
    duplicates[
        [
            "location_id",
            "state",
            "lga"
        ]
    ].sort_values(
        "location_id"
    ).to_string(index=False)
)


# ============================================================
# CHECK WHETHER DUPLICATES HAVE DIFFERENT VALUES
# ============================================================

print("\n[4] Checking climate values...")

# Columns that describe the location
location_columns = [
    "location_id",
    "state",
    "lga",
    "latitude",
    "longitude"
]

# Everything else is climate data
climate_columns = [
    column
    for column in df.columns
    if column not in location_columns
]


print(
    f"Climate variables checked: "
    f"{len(climate_columns)}"
)


# ============================================================
# COMPARE DUPLICATE RECORDS
# ============================================================

problem_locations = []

for location_id, group in duplicates.groupby(
    "location_id"
):

    # Number of unique climate records
    unique_climate_records = (
        group[climate_columns]
        .drop_duplicates()
        .shape[0]
    )

    if unique_climate_records > 1:

        problem_locations.append(
            location_id
        )


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC RESULT")
print("=" * 70)

if len(problem_locations) == 0:

    print(
        "\nSUCCESS!"
    )

    print(
        "All duplicate location IDs contain "
        "identical climate values."
    )

    print(
        "\nThe duplicates can safely be removed."
    )

else:

    print(
        "\nWARNING!"
    )

    print(
        "Some duplicate location IDs contain "
        "different climate values."
    )

    print(
        f"\nProblematic locations: "
        f"{len(problem_locations)}"
    )

    for location_id in problem_locations:

        print(
            f"\nLocation ID: {location_id}"
        )

        print(
            duplicates[
                duplicates["location_id"]
                == location_id
            ][
                [
                    "location_id",
                    "state",
                    "lga"
                ] + climate_columns
            ].to_string(index=False)
        )


print("\n" + "=" * 70)