from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "north_central_lgas.csv"
)

SOIL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "north_central_soil.csv"
)


print("\n[1] Loading locations...")

locations = pd.read_csv(LOCATIONS_FILE)

print(f"Total locations: {len(locations)}")


print("\n[2] Loading soil data...")

soil = pd.read_csv(SOIL_FILE)

print(f"Soil records: {len(soil)}")


print("\n[3] Checking missing location IDs...")

location_ids = set(
    locations["location_id"]
)

soil_ids = set(
    soil["location_id"]
)

missing_ids = sorted(
    location_ids - soil_ids
)

print(
    f"Locations without soil records: "
    f"{len(missing_ids)}"
)

if missing_ids:

    missing = locations[
        locations["location_id"].isin(missing_ids)
    ]

    print("\nMissing locations:")

    print(
        missing[
            [
                "location_id",
                "state",
                "lga",
                "latitude",
                "longitude"
            ]
        ].to_string(index=False)
    )


print("\n[4] Checking records containing missing soil values...")

soil_columns = [
    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "clay_percent",
    "sand_percent",
    "silt_percent",
    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]

missing_values = soil[
    soil[soil_columns]
    .isnull()
    .any(axis=1)
]

print(
    f"Records with missing soil values: "
    f"{len(missing_values)}"
)

if len(missing_values) > 0:

    print("\nAffected records:")

    print(
        missing_values[
            [
                "location_id",
                "state",
                "lga"
            ]
        ].to_string(index=False)
    )


print("\n========================================")
print("SOIL DATA DIAGNOSTIC COMPLETED")
print("========================================")