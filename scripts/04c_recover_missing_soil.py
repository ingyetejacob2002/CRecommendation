"""
04c_recover_missing_soil.py

Recover missing SoilGrids records for North-Central Nigeria LGAs.

This script:
1. Loads the verified LGA locations.
2. Loads the existing soil dataset.
3. Identifies locations with no soil record.
4. Retries SoilGrids for only those locations.
5. Saves recovered records.
6. Rebuilds the complete soil dataset.
7. Reports locations that still failed.

Project:
Crop Recommendation System
"""

from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_SOIL_DIR = DATA_DIR / "raw" / "soil"

LOCATIONS_FILE = PROCESSED_DIR / "north_central_lgas.csv"
SOIL_FILE = PROCESSED_DIR / "north_central_soil.csv"

RAW_SOIL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SOILGRIDS CONFIGURATION
# ============================================================

SOILGRIDS_URL = (
    "https://rest.isric.org/soilgrids/v2.0/properties/query"
)

DEPTH = "0-5cm"

# SoilGrids properties
PROPERTIES = {
    "phh2o": "soil_ph",
    "soc": "soil_organic_carbon_g_kg",
    "nitrogen": "soil_nitrogen_g_kg",
    "clay": "clay_percent",
    "sand": "sand_percent",
    "silt": "silt_percent",
    "cec": "cec_cmolc_kg",
    "bdod": "bulk_density_g_cm3",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_filename(text):
    """
    Convert LGA name into a safe filename.
    """
    text = str(text)
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace('"', "")
    )


def get_soil_value(data, property_name):
    """
    Extract the mean SoilGrids value for a property.
    """

    try:
        layers = data["properties"]["layers"]

        for layer in layers:

            if layer["name"] != property_name:
                continue

            depths = layer.get("depths", [])

            for depth_info in depths:

                if depth_info.get("label") == DEPTH:

                    values = depth_info.get("values", {})

                    value = values.get("mean")

                    if value is None:
                        return np.nan

                    return value

        return np.nan

    except Exception:
        return np.nan


def convert_soil_value(property_name, value):
    """
    Convert SoilGrids units into the units used by the project.

    SoilGrids commonly returns:
        pH          -> pH * 10
        SOC         -> g/kg * 10
        nitrogen    -> cg/kg
        clay/sand   -> g/kg
        CEC         -> mmol(c)/kg
        bulk density -> cg/cm3
    """

    if pd.isna(value):
        return np.nan

    if property_name == "phh2o":
        return value / 10

    elif property_name == "soc":
        return value / 10

    elif property_name == "nitrogen":
        return value / 100

    elif property_name in ["clay", "sand", "silt"]:
        return value / 10

    elif property_name == "cec":
        return value / 10

    elif property_name == "bdod":
        return value / 100

    return value


def download_soil(location):
    """
    Download SoilGrids data for one location.
    """

    location_id = int(location["location_id"])
    state = str(location["state"])
    lga = str(location["lga"])

    latitude = float(location["latitude"])
    longitude = float(location["longitude"])

    print()
    print("=" * 60)
    print(f"Location ID : {location_id}")
    print(f"State       : {state}")
    print(f"LGA         : {lga}")
    print(f"Latitude    : {latitude}")
    print(f"Longitude   : {longitude}")
    print("=" * 60)

    params = {
        "lon": longitude,
        "lat": latitude,
        "property": list(PROPERTIES.keys()),
        "depth": DEPTH,
        "value": "mean",
    }

    filename = (
        f"{location_id:04d}_{safe_filename(state)}_"
        f"{safe_filename(lga)}.json"
    )

    output_file = RAW_SOIL_DIR / filename

    for attempt in range(1, 4):

        try:

            print(f"Attempt {attempt}/3...")

            response = requests.get(
                SOILGRIDS_URL,
                params=params,
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            # Save raw response
            output_file.write_text(
                response.text,
                encoding="utf-8"
            )

            print("SoilGrids request successful.")

            record = {
                "location_id": location_id,
                "state": state,
                "lga": lga,
                "latitude": latitude,
                "longitude": longitude,
            }

            for property_name, column_name in PROPERTIES.items():

                raw_value = get_soil_value(
                    data,
                    property_name
                )

                converted_value = convert_soil_value(
                    property_name,
                    raw_value
                )

                record[column_name] = converted_value

            return record

        except Exception as error:

            print(
                f"Request failed on attempt {attempt}/3:"
            )

            print(error)

            if attempt < 3:
                time.sleep(5 * attempt)

    print("All attempts failed.")

    return None


# ============================================================
# 4. LOAD LOCATIONS
# ============================================================

print()
print("=" * 70)
print("RECOVERING MISSING SOIL DATA")
print("=" * 70)

print()
print("[1/6] Loading verified LGA locations...")

if not LOCATIONS_FILE.exists():

    raise FileNotFoundError(
        f"""
Location file not found:

{LOCATIONS_FILE}

Expected:
data/processed/north_central_lgas.csv
"""
    )

locations = pd.read_csv(LOCATIONS_FILE)

print(f"Total locations: {len(locations)}")


# ============================================================
# 5. LOAD EXISTING SOIL DATA
# ============================================================

print()
print("[2/6] Loading existing soil data...")

if SOIL_FILE.exists():

    soil_existing = pd.read_csv(SOIL_FILE)

    print(
        f"Existing soil records: "
        f"{len(soil_existing)}"
    )

else:

    print(
        "Existing soil dataset was not found."
    )

    soil_existing = pd.DataFrame()


# ============================================================
# 6. IDENTIFY MISSING LOCATIONS
# ============================================================

print()
print("[3/6] Identifying missing soil locations...")

if len(soil_existing) > 0:

    existing_ids = set(
        pd.to_numeric(
            soil_existing["location_id"],
            errors="coerce"
        )
        .dropna()
        .astype(int)
    )

else:

    existing_ids = set()


locations["location_id"] = (
    pd.to_numeric(
        locations["location_id"],
        errors="coerce"
    )
    .astype(int)
)

missing_locations = locations[
    ~locations["location_id"].isin(existing_ids)
].copy()

print(
    f"Locations requiring recovery: "
    f"{len(missing_locations)}"
)

if len(missing_locations) > 0:

    print()
    print("Locations to recover:")

    print(
        missing_locations[
            ["location_id", "state", "lga"]
        ].to_string(index=False)
    )

else:

    print()
    print("No completely missing locations found.")


# ============================================================
# 7. RETRY SOILGRIDS
# ============================================================

print()
print("[4/6] Retrying SoilGrids requests...")

recovered_records = []

failed_locations = []

for _, location in missing_locations.iterrows():

    record = download_soil(location)

    if record is not None:

        recovered_records.append(record)

        print(
            f"Recovered: "
            f"{location['state']} - {location['lga']}"
        )

    else:

        failed_locations.append(
            location["location_id"]
        )

    # Small delay to avoid hammering API
    time.sleep(2)


# ============================================================
# 8. COMBINE OLD + RECOVERED DATA
# ============================================================

print()
print("[5/6] Rebuilding soil dataset...")

if recovered_records:

    recovered_df = pd.DataFrame(
        recovered_records
    )

    print(
        f"Newly recovered records: "
        f"{len(recovered_df)}"
    )

else:

    recovered_df = pd.DataFrame()

    print("No new records were recovered.")


# Combine
if len(soil_existing) > 0 and len(recovered_df) > 0:

    soil_combined = pd.concat(
        [
            soil_existing,
            recovered_df
        ],
        ignore_index=True
    )

elif len(soil_existing) > 0:

    soil_combined = soil_existing.copy()

else:

    soil_combined = recovered_df.copy()


# Remove duplicate location IDs
if len(soil_combined) > 0:

    soil_combined = (
        soil_combined
        .drop_duplicates(
            subset=["location_id"],
            keep="last"
        )
        .sort_values("location_id")
        .reset_index(drop=True)
    )


# ============================================================
# 9. SAVE
# ============================================================

soil_combined.to_csv(
    SOIL_FILE,
    index=False
)

print()
print(f"Soil dataset saved to:")
print(SOIL_FILE)


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

print()
print("[6/6] Final validation...")

print()
print("=" * 70)
print("SOIL RECOVERY COMPLETED")
print("=" * 70)

print()
print(
    f"Total LGA locations: "
    f"{len(locations)}"
)

print(
    f"Soil records now: "
    f"{len(soil_combined)}"
)

print(
    f"Unique location IDs: "
    f"{soil_combined['location_id'].nunique()}"
)


# ------------------------------------------------------------
# Locations still missing completely
# ------------------------------------------------------------

all_ids = set(
    locations["location_id"]
)

final_soil_ids = set(
    soil_combined["location_id"]
)

still_missing = sorted(
    all_ids - final_soil_ids
)

print()
print(
    f"Locations still without soil records: "
    f"{len(still_missing)}"
)

if still_missing:

    remaining = locations[
        locations["location_id"].isin(
            still_missing
        )
    ]

    print()
    print("Still missing:")

    print(
        remaining[
            ["location_id", "state", "lga"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Missing values inside existing records
# ------------------------------------------------------------

soil_columns = list(PROPERTIES.values())

missing_counts = (
    soil_combined[soil_columns]
    .isna()
    .sum()
)

records_with_missing_values = (
    soil_combined[soil_columns]
    .isna()
    .any(axis=1)
    .sum()
)

print()
print(
    "Records containing missing soil values: "
    f"{records_with_missing_values}"
)

if records_with_missing_values > 0:

    print()
    print("Missing values by column:")

    print(
        missing_counts[
            missing_counts > 0
        ]
        .to_string()
    )


print()
print("=" * 70)
print("NEXT STEP:")
print("Review remaining missing soil data before merging")
print("climate + soil datasets.")
print("=" * 70)