from pathlib import Path
import time
import requests
import pandas as pd
from tqdm import tqdm

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "north_central_lgas.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "soil"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "north_central_soil.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# SOILGRIDS CONFIGURATION
# ============================================================

SOILGRIDS_URL = (
    "https://rest.isric.org/soilgrids/v2.0/properties/query"
)

# Soil properties important for crop suitability.
#
# phh2o  = soil pH
# soc    = soil organic carbon
# nitrogen = total nitrogen
# clay   = clay content
# sand   = sand content
# silt   = silt content
# cec    = cation exchange capacity
# bdod   = bulk density
#
# We use the 0-5 cm depth as the first soil layer.
# Additional depths can be incorporated later if required
# by the journal methodology.

PROPERTIES = [
    "phh2o",
    "soc",
    "nitrogen",
    "clay",
    "sand",
    "silt",
    "cec",
    "bdod",
]

DEPTH = "0-5cm"
VALUE = "mean"


# ============================================================
# LOAD LOCATIONS
# ============================================================

print("\n[1/6] Loading verified LGA locations...")

if not LOCATIONS_FILE.exists():
    raise FileNotFoundError(
        f"""
Location file not found:

{LOCATIONS_FILE}

Make sure 01_download_locations.py created:
data/processed/north_central_lga_locations.csv
"""
    )

locations = pd.read_csv(LOCATIONS_FILE)

print(f"Locations loaded: {len(locations)}")

required_columns = [
    "location_id",
    "state",
    "lga",
    "latitude",
    "longitude",
]

missing_columns = [
    col for col in required_columns
    if col not in locations.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required location columns: {missing_columns}"
    )


# ============================================================
# REMOVE DUPLICATE LOCATIONS
# ============================================================

locations = (
    locations
    .drop_duplicates(subset=["location_id"])
    .reset_index(drop=True)
)

print(
    f"Unique locations to process: "
    f"{locations['location_id'].nunique()}"
)


# ============================================================
# FUNCTION TO QUERY SOILGRIDS
# ============================================================

def get_soil_data(latitude, longitude):
    """
    Query SoilGrids for a single coordinate.
    """

    params = {
        "lon": longitude,
        "lat": latitude,
        "property": PROPERTIES,
        "depth": DEPTH,
        "value": VALUE,
    }

    response = requests.get(
        SOILGRIDS_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# EXTRACT SOIL VALUES
# ============================================================

def extract_soil_values(data):
    """
    Extract SoilGrids property values from API response.
    """

    result = {}

    layers = data.get("properties", {}).get("layers", [])

    for layer in layers:

        property_name = layer.get("name")

        depths = layer.get("depths", [])

        value = None

        for depth in depths:

            depth_label = depth.get("label")

            if depth_label == DEPTH:

                values = depth.get("values", {})

                value = values.get(VALUE)

                break

        result[property_name] = value

    return result


# ============================================================
# DOWNLOAD SOIL DATA
# ============================================================

print("\n[2/6] Downloading SoilGrids data...")

records = []

failed = []

for _, row in tqdm(
    locations.iterrows(),
    total=len(locations),
    desc="SoilGrids",
):

    location_id = int(row["location_id"])

    state = row["state"]
    lga = row["lga"]

    latitude = float(row["latitude"])
    longitude = float(row["longitude"])

    success = False

    for attempt in range(1, 4):

        try:

            data = get_soil_data(
                latitude,
                longitude,
            )

            soil_values = extract_soil_values(data)

            record = {
                "location_id": location_id,
                "state": state,
                "lga": lga,
                "latitude": latitude,
                "longitude": longitude,
            }

            for property_name in PROPERTIES:

                record[property_name] = soil_values.get(
                    property_name
                )

            records.append(record)

            success = True

            break

        except Exception as e:

            print(
                f"\nRequest error for "
                f"{state} - {lga} "
                f"(attempt {attempt}/3): {e}"
            )

            time.sleep(2)

    if not success:

        failed.append(
            {
                "location_id": location_id,
                "state": state,
                "lga": lga,
            }
        )


# ============================================================
# CHECK DOWNLOAD RESULTS
# ============================================================

print("\n[3/6] Soil download summary...")

print(
    f"Successfully processed: {len(records)}"
)

print(
    f"Failed: {len(failed)}"
)

if failed:

    print("\nFailed locations:")

    for item in failed:

        print(
            f"{item['location_id']}: "
            f"{item['state']} - "
            f"{item['lga']}"
        )


if not records:

    raise RuntimeError(
        """
No SoilGrids records were successfully retrieved.

Check your internet connection and
try the script again.
"""
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n[4/6] Creating soil dataset...")

soil_df = pd.DataFrame(records)

# ============================================================
# CONVERT SOIL UNITS
# ============================================================
#
# SoilGrids commonly reports:
#
# pH:
#   pH is stored as pH × 10
#
# SOC:
#   dg/kg
#
# Nitrogen:
#   cg/kg
#
# Clay/Sand/Silt:
#   g/kg
#
# CEC:
#   mmol(c)/kg
#
# Bulk density:
#   cg/cm3
#
# We convert these to more interpretable units.

if "phh2o" in soil_df.columns:
    soil_df["soil_ph"] = (
        soil_df["phh2o"] / 10
    )

if "soc" in soil_df.columns:
    soil_df["soil_organic_carbon_g_kg"] = (
        soil_df["soc"] / 10
    )

if "nitrogen" in soil_df.columns:
    soil_df["soil_nitrogen_g_kg"] = (
        soil_df["nitrogen"] / 100
    )

if "clay" in soil_df.columns:
    soil_df["clay_percent"] = (
        soil_df["clay"] / 10
    )

if "sand" in soil_df.columns:
    soil_df["sand_percent"] = (
        soil_df["sand"] / 10
    )

if "silt" in soil_df.columns:
    soil_df["silt_percent"] = (
        soil_df["silt"] / 10
    )

if "cec" in soil_df.columns:
    soil_df["cec_cmolc_kg"] = (
        soil_df["cec"] / 10
    )

if "bdod" in soil_df.columns:
    soil_df["bulk_density_g_cm3"] = (
        soil_df["bdod"] / 100
    )


# ============================================================
# SELECT FINAL FEATURES
# ============================================================

final_columns = [
    "location_id",
    "state",
    "lga",
    "latitude",
    "longitude",

    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",

    "clay_percent",
    "sand_percent",
    "silt_percent",

    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]

# Keep only columns that actually exist.

final_columns = [
    col
    for col in final_columns
    if col in soil_df.columns
]

soil_df = soil_df[final_columns]


# ============================================================
# REMOVE DUPLICATES
# ============================================================

soil_df = (
    soil_df
    .drop_duplicates(
        subset=["location_id"]
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE DATASET
# ============================================================

print("\n[5/6] Saving soil dataset...")

soil_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n[6/6] Validating soil dataset...")

print("\n========================================")
print("SOIL DATASET COMPLETED")
print("========================================")

print(
    f"Records: {len(soil_df)}"
)

print(
    f"Unique locations: "
    f"{soil_df['location_id'].nunique()}"
)

print("\nColumns:")

print(
    soil_df.columns.tolist()
)

print("\nFirst 5 records:")

print(
    soil_df.head().to_string(index=False)
)

print("\nMissing values:")

print(
    soil_df.isnull().sum()
)

print("\nSaved to:")

print(OUTPUT_FILE)