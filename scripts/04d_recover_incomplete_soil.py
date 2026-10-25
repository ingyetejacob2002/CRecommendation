"""
04d_recover_incomplete_soil.py

Recover soil records that exist in the soil dataset but contain
missing soil variables.

Target locations:
    13  Benue - Makurdi
    58  Kwara - Ilorin West
    82  Niger - Bida
    85  Niger - Chanchaga
    109 Plateau - Jos North

The script retries SoilGrids for these locations only.
"""

from pathlib import Path
import time
import requests
import pandas as pd
import numpy as np


# ============================================================
# 1. PROJECT PATHS
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
# 3. TARGET MISSING LOCATION IDs
# ============================================================

TARGET_IDS = [13, 58, 82, 85, 109]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_filename(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace('"', "")
    )


def get_soil_value(data, property_name):

    try:

        layers = data["properties"]["layers"]

        for layer in layers:

            if layer["name"] != property_name:
                continue

            depths = layer.get("depths", [])

            for depth_info in depths:

                if depth_info.get("label") == DEPTH:

                    values = depth_info.get(
                        "values",
                        {}
                    )

                    return values.get(
                        "mean",
                        np.nan
                    )

        return np.nan

    except Exception:

        return np.nan


def convert_soil_value(
    property_name,
    value
):

    if pd.isna(value):
        return np.nan

    if property_name == "phh2o":
        return value / 10

    if property_name == "soc":
        return value / 10

    if property_name == "nitrogen":
        return value / 100

    if property_name in [
        "clay",
        "sand",
        "silt"
    ]:
        return value / 10

    if property_name == "cec":
        return value / 10

    if property_name == "bdod":
        return value / 100

    return value


def download_soil(location):

    location_id = int(
        location["location_id"]
    )

    state = str(
        location["state"]
    )

    lga = str(
        location["lga"]
    )

    latitude = float(
        location["latitude"]
    )

    longitude = float(
        location["longitude"]
    )

    print()
    print("=" * 60)
    print(
        f"Location: {location_id} | "
        f"{state} - {lga}"
    )
    print(
        f"Coordinates: "
        f"{latitude}, {longitude}"
    )
    print("=" * 60)

    params = {
        "lon": longitude,
        "lat": latitude,
        "property": list(
            PROPERTIES.keys()
        ),
        "depth": DEPTH,
        "value": "mean",
    }

    filename = (
        f"{location_id:04d}_"
        f"{safe_filename(state)}_"
        f"{safe_filename(lga)}.json"
    )

    output_file = (
        RAW_SOIL_DIR / filename
    )

    for attempt in range(1, 4):

        try:

            print(
                f"Attempt {attempt}/3..."
            )

            response = requests.get(
                SOILGRIDS_URL,
                params=params,
                timeout=180
            )

            response.raise_for_status()

            data = response.json()

            # Save raw response
            output_file.write_text(
                response.text,
                encoding="utf-8"
            )

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

                value = convert_soil_value(
                    property_name,
                    raw_value
                )

                record[column_name] = value

            return record

        except Exception as error:

            print(
                f"Attempt {attempt} failed:"
            )

            print(error)

            if attempt < 3:

                wait_time = 10 * attempt

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

    return None


# ============================================================
# 5. LOAD DATA
# ============================================================

print()
print("=" * 70)
print("RECOVERING INCOMPLETE SOIL RECORDS")
print("=" * 70)

print()
print("[1/5] Loading locations...")

if not LOCATIONS_FILE.exists():

    raise FileNotFoundError(
        f"Location file not found:\n"
        f"{LOCATIONS_FILE}"
    )

locations = pd.read_csv(
    LOCATIONS_FILE
)

locations["location_id"] = (
    pd.to_numeric(
        locations["location_id"],
        errors="coerce"
    )
    .astype(int)
)

print(
    f"Total locations: {len(locations)}"
)


# ============================================================
# 6. LOAD SOIL DATA
# ============================================================

print()
print("[2/5] Loading soil dataset...")

if not SOIL_FILE.exists():

    raise FileNotFoundError(
        f"Soil file not found:\n"
        f"{SOIL_FILE}"
    )

soil = pd.read_csv(
    SOIL_FILE
)

print(
    f"Soil records: {len(soil)}"
)


# ============================================================
# 7. IDENTIFY INCOMPLETE RECORDS
# ============================================================

print()
print("[3/5] Checking incomplete records...")

soil_columns = list(
    PROPERTIES.values()
)

incomplete_mask = (
    soil[soil_columns]
    .isna()
    .any(axis=1)
)

incomplete = soil[
    incomplete_mask
].copy()

print(
    f"Incomplete records found: "
    f"{len(incomplete)}"
)

print()

print(
    incomplete[
        [
            "location_id",
            "state",
            "lga"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. RETRY SOILGRIDS
# ============================================================

print()
print("[4/5] Retrying SoilGrids...")

target_locations = locations[
    locations["location_id"].isin(
        TARGET_IDS
    )
].copy()

recovered = []

for _, location in target_locations.iterrows():

    record = download_soil(
        location
    )

    if record is not None:

        recovered.append(record)

        print(
            f"Successfully downloaded: "
            f"{location['state']} - "
            f"{location['lga']}"
        )

    else:

        print(
            f"FAILED: "
            f"{location['state']} - "
            f"{location['lga']}"
        )

    time.sleep(3)


# ============================================================
# 9. UPDATE DATASET
# ============================================================

print()
print("[5/5] Updating soil dataset...")

if recovered:

    recovered_df = pd.DataFrame(
        recovered
    )

    print(
        f"Recovered records: "
        f"{len(recovered_df)}"
    )

    # Remove old versions of recovered IDs
    soil = soil[
        ~soil["location_id"].isin(
            recovered_df["location_id"]
        )
    ]

    soil = pd.concat(
        [
            soil,
            recovered_df
        ],
        ignore_index=True
    )

else:

    print(
        "No records were recovered."
    )


# Sort and remove duplicates
soil = (
    soil
    .drop_duplicates(
        subset=["location_id"],
        keep="last"
    )
    .sort_values(
        "location_id"
    )
    .reset_index(drop=True)
)


# Save
soil.to_csv(
    SOIL_FILE,
    index=False
)


# ============================================================
# 10. FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("INCOMPLETE SOIL RECOVERY COMPLETED")
print("=" * 70)

print()
print(
    f"Total soil records: "
    f"{len(soil)}"
)

print(
    f"Unique locations: "
    f"{soil['location_id'].nunique()}"
)

remaining_mask = (
    soil[soil_columns]
    .isna()
    .any(axis=1)
)

remaining = soil[
    remaining_mask
]

print()
print(
    f"Records still containing "
    f"missing soil values: "
    f"{len(remaining)}"
)

if len(remaining) > 0:

    print()

    print(
        remaining[
            [
                "location_id",
                "state",
                "lga"
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "These records will need "
        "careful spatial imputation."
    )

else:

    print()
    print(
        "SUCCESS: All soil records "
        "are complete."
    )

print()
print("Saved to:")
print(SOIL_FILE)

print()
print("=" * 70)