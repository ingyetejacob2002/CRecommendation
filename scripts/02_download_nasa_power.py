"""
02_download_nasa_power.py

Download and process NASA POWER climate data for verified
North-Central Nigeria LGA locations.

Period:
    2015-01-01 to 2025-12-31

Parameters:
    PRECTOTCORR
    T2M
    T2M_MAX
    T2M_MIN
    RH2M
    ALLSKY_SFC_SW_DWN
    WS2M
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

CLIMATE_DIR = RAW_DIR / "climate"

PROCESSED_DIR = DATA_DIR / "processed"

LOCATIONS_FILE = PROCESSED_DIR / "north_central_lgas.csv"

OUTPUT_FILE = PROCESSED_DIR / "north_central_climate.csv"


START_DATE = "20150101"
END_DATE = "20251231"

COMMUNITY = "AG"


PARAMETERS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]


NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

CLIMATE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD VERIFIED LOCATIONS
# ============================================================

print("\n[1/6] Loading verified LGA locations...")

if not LOCATIONS_FILE.exists():
    raise FileNotFoundError(
        f"Location file not found:\n{LOCATIONS_FILE}"
    )


locations = pd.read_csv(LOCATIONS_FILE)

print(f"Locations loaded: {len(locations)}")


# ============================================================
# CHECK REQUIRED LOCATION COLUMNS
# ============================================================

required_location_columns = [
    "location_id",
    "state",
    "lga",
    "latitude",
    "longitude",
]


missing_columns = [
    col
    for col in required_location_columns
    if col not in locations.columns
]


if missing_columns:

    raise ValueError(
        "The location CSV is missing required columns:\n"
        + "\n".join(missing_columns)
    )


# Make location_id integer/string-safe
locations["location_id"] = (
    locations["location_id"]
    .astype(int)
)


# ============================================================
# DISPLAY PARAMETERS
# ============================================================

print("\nNASA POWER parameters:")

for parameter in PARAMETERS:
    print(f"- {parameter}")


print(f"\nPeriod: {START_DATE} to {END_DATE}")
print(f"Community: {COMMUNITY}")


# ============================================================
# NASA POWER DOWNLOAD FUNCTION
# ============================================================

def download_nasa_power(
    latitude,
    longitude,
    output_file,
    max_retries=3,
):
    """
    Download NASA POWER daily data.

    Returns:
        True  -> successful download
        False -> failed download
    """

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": COMMUNITY,
        "longitude": longitude,
        "latitude": latitude,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "LST",
    }

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.get(
                NASA_URL,
                params=params,
                timeout=60,
            )

            response.raise_for_status()

            data = response.json()

            # Basic validation
            if "properties" not in data:
                raise ValueError(
                    "NASA response does not contain 'properties'."
                )

            if "parameter" not in data["properties"]:
                raise ValueError(
                    "NASA response does not contain "
                    "'properties.parameter'."
                )

            # Save response
            with open(
                output_file,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2,
                )

            return True

        except Exception as e:

            print(
                f"Request error "
                f"(attempt {attempt}/{max_retries}): {e}"
            )

            if attempt < max_retries:
                time.sleep(5)

    return False


# ============================================================
# DOWNLOAD ONLY MISSING FILES
# ============================================================

print("\n[2/6] Checking/downloading climate data...")


downloaded = 0
already_exists = 0
failed = 0

failed_locations = []


for _, row in tqdm(
    locations.iterrows(),
    total=len(locations),
    desc="NASA POWER",
):

    location_id = int(row["location_id"])

    state = str(row["state"])

    lga = str(row["lga"])

    latitude = float(row["latitude"])

    longitude = float(row["longitude"])


    filename = (
        f"{location_id:04d}_"
        f"{state}_"
        f"{lga}.json"
    )

    output_file = CLIMATE_DIR / filename


    # --------------------------------------------------------
    # DO NOT DOWNLOAD AGAIN IF FILE EXISTS
    # --------------------------------------------------------

    if output_file.exists():

        already_exists += 1

        continue


    success = download_nasa_power(
        latitude=latitude,
        longitude=longitude,
        output_file=output_file,
    )


    if success:

        downloaded += 1

    else:

        failed += 1

        failed_locations.append(
            location_id
        )


print("\nDownload summary:")

print(f"Already downloaded: {already_exists}")
print(f"Newly downloaded:   {downloaded}")
print(f"Failed:             {failed}")


if failed_locations:

    print("\nFailed locations:")

    for location_id in failed_locations:

        match = locations[
            locations["location_id"] == location_id
        ]

        if len(match):

            row = match.iloc[0]

            print(
                f"  {location_id}: "
                f"{row['state']} - {row['lga']}"
            )

else:

    print("\nAll locations have climate files.")


# ============================================================
# PARSE NASA POWER JSON
# ============================================================

print(
    "\n[3/6] Processing downloaded NASA POWER files..."
)


def parse_nasa_file(
    json_file,
):
    """
    Parse a NASA POWER JSON file.

    NASA POWER structure:

        geometry
        properties
            parameter
                PRECTOTCORR
                T2M
                T2M_MAX
                ...

    Returns:
        pandas DataFrame
    """

    with open(
        json_file,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)


    # --------------------------------------------------------
    # LOCATION INFORMATION COMES FROM OUR OWN FILE NAME
    # --------------------------------------------------------

    filename = json_file.stem

    match = re.match(
        r"^(\d+)_([^_]+)_(.+)$",
        filename,
    )


    if not match:

        raise ValueError(
            f"Could not identify location from filename: "
            f"{json_file.name}"
        )


    location_id = int(match.group(1))

    state = match.group(2)

    lga = match.group(3)


    # --------------------------------------------------------
    # NASA DATA
    # --------------------------------------------------------

    properties = data.get(
        "properties",
        {},
    )


    parameters = properties.get(
        "parameter",
        {},
    )


    if not parameters:

        raise ValueError(
            "No climate parameters found."
        )


    # --------------------------------------------------------
    # GET ALL DATES
    # --------------------------------------------------------

    all_dates = set()


    for parameter_name in PARAMETERS:

        parameter_data = parameters.get(
            parameter_name,
            {},
        )

        all_dates.update(
            parameter_data.keys()
        )


    if not all_dates:

        raise ValueError(
            "No climate dates found."
        )


    # --------------------------------------------------------
    # BUILD DAILY DATAFRAME
    # --------------------------------------------------------

    records = []


    for date in sorted(all_dates):

        record = {

            "location_id": location_id,

            "state": state,

            "lga": lga,

            "date": pd.to_datetime(
                date,
                format="%Y%m%d",
            ),

        }


        for parameter_name in PARAMETERS:

            parameter_data = parameters.get(
                parameter_name,
                {},
            )


            value = parameter_data.get(
                date,
                None,
            )


            record[
                parameter_name
            ] = value


        records.append(record)


    df = pd.DataFrame(records)


    return df


# ============================================================
# PROCESS FILES
# ============================================================

json_files = sorted(
    CLIMATE_DIR.glob("*.json")
)


print(
    f"JSON files found: {len(json_files)}"
)


climate_frames = []

parse_failures = []


for json_file in tqdm(
    json_files,
    desc="Parsing climate files",
):

    try:

        df = parse_nasa_file(
            json_file
        )

        climate_frames.append(df)

    except Exception as e:

        print(
            f"\nCould not parse "
            f"{json_file.name}:"
        )

        print(e)

        parse_failures.append(
            json_file.name
        )


# ============================================================
# VALIDATE PARSING
# ============================================================

if not climate_frames:

    raise RuntimeError(
        "\nNo NASA POWER climate files "
        "could be processed.\n\n"
        "The downloaded JSON structure "
        "could not be parsed."
    )


# ============================================================
# COMBINE DAILY RECORDS
# ============================================================

print(
    "\n[4/6] Combining climate records..."
)


climate_daily = pd.concat(
    climate_frames,
    ignore_index=True,
)


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

for parameter in PARAMETERS:

    climate_daily[
        parameter
    ] = pd.to_numeric(
        climate_daily[
            parameter
        ],
        errors="coerce",
    )


# ============================================================
# SORT DATA
# ============================================================

climate_daily = climate_daily.sort_values(
    [
        "location_id",
        "date",
    ]
)


# ============================================================
# SAVE DAILY DATA
# ============================================================

DAILY_OUTPUT = (
    PROCESSED_DIR
    / "north_central_climate_daily.csv"
)


climate_daily.to_csv(
    DAILY_OUTPUT,
    index=False,
)


print(
    f"Daily climate data saved:\n"
    f"{DAILY_OUTPUT}"
)


print(
    f"\nDaily records: "
    f"{len(climate_daily):,}"
)


# ============================================================
# CREATE CLIMATE FEATURES
# ============================================================

print(
    "\n[5/6] Creating climate features..."
)


# Add year/month
climate_daily["year"] = (
    climate_daily["date"].dt.year
)

climate_daily["month"] = (
    climate_daily["date"].dt.month
)


# ------------------------------------------------------------
# MONTHLY AGGREGATION
# ------------------------------------------------------------

monthly = (
    climate_daily
    .groupby(
        [
            "location_id",
            "state",
            "lga",
            "year",
            "month",
        ],
        as_index=False,
    )
    .agg(
        {

            "PRECTOTCORR": "sum",

            "T2M": "mean",

            "T2M_MAX": "mean",

            "T2M_MIN": "mean",

            "RH2M": "mean",

            "ALLSKY_SFC_SW_DWN": "mean",

            "WS2M": "mean",

        }
    )
)


# ------------------------------------------------------------
# YEARLY AGGREGATION
# ------------------------------------------------------------

annual = (
    climate_daily
    .groupby(
        [
            "location_id",
            "state",
            "lga",
            "year",
        ],
        as_index=False,
    )
    .agg(
        {

            "PRECTOTCORR": "sum",

            "T2M": "mean",

            "T2M_MAX": "mean",

            "T2M_MIN": "mean",

            "RH2M": "mean",

            "ALLSKY_SFC_SW_DWN": "mean",

            "WS2M": "mean",

        }
    )
)


# ============================================================
# MULTI-YEAR LOCATION FEATURES
# ============================================================

location_features = (
    climate_daily
    .groupby(
        [
            "location_id",
            "state",
            "lga",
        ],
        as_index=False,
    )
    .agg(
        {

            "PRECTOTCORR": [
                "mean",
                "std",
                "min",
                "max",
                "sum",
            ],

            "T2M": [
                "mean",
                "std",
                "min",
                "max",
            ],

            "T2M_MAX": [
                "mean",
                "max",
            ],

            "T2M_MIN": [
                "mean",
                "min",
            ],

            "RH2M": [
                "mean",
                "min",
                "max",
            ],

            "ALLSKY_SFC_SW_DWN": [
                "mean",
                "min",
                "max",
            ],

            "WS2M": [
                "mean",
                "min",
                "max",
            ],

        }
    )
)


# Flatten multi-level column names
location_features.columns = [

    "_".join(
        [str(x) for x in col if str(x) != ""]
    ).strip("_")

    for col in location_features.columns
]


# ============================================================
# ADD LOCATION COORDINATES
# ============================================================

coordinate_columns = [
    "location_id",
    "latitude",
    "longitude",
]


coordinates = locations[
    coordinate_columns
].copy()


location_features = location_features.merge(
    coordinates,
    on="location_id",
    how="left",
)


# ============================================================
# RENAME FEATURES
# ============================================================

location_features = location_features.rename(

    columns={

        "PRECTOTCORR_mean":
            "mean_daily_precipitation_mm",

        "PRECTOTCORR_std":
            "std_daily_precipitation_mm",

        "PRECTOTCORR_min":
            "min_daily_precipitation_mm",

        "PRECTOTCORR_max":
            "max_daily_precipitation_mm",

        "PRECTOTCORR_sum":
            "total_precipitation_mm",

        "T2M_mean":
            "mean_temperature_c",

        "T2M_std":
            "std_temperature_c",

        "T2M_min":
            "min_temperature_c",

        "T2M_max":
            "max_temperature_c",

        "T2M_MAX_mean":
            "mean_max_temperature_c",

        "T2M_MAX_max":
            "absolute_max_temperature_c",

        "T2M_MIN_mean":
            "mean_min_temperature_c",

        "T2M_MIN_min":
            "absolute_min_temperature_c",

        "RH2M_mean":
            "mean_relative_humidity_percent",

        "RH2M_min":
            "minimum_relative_humidity_percent",

        "RH2M_max":
            "maximum_relative_humidity_percent",

        "ALLSKY_SFC_SW_DWN_mean":
            "mean_solar_radiation",

        "ALLSKY_SFC_SW_DWN_min":
            "minimum_solar_radiation",

        "ALLSKY_SFC_SW_DWN_max":
            "maximum_solar_radiation",

        "WS2M_mean":
            "mean_wind_speed",

        "WS2M_min":
            "minimum_wind_speed",

        "WS2M_max":
            "maximum_wind_speed",

    }

)


# ============================================================
# SAVE LOCATION CLIMATE FEATURES
# ============================================================

print(
    "\n[6/6] Saving climate feature dataset..."
)


location_features.to_csv(
    OUTPUT_FILE,
    index=False,
)


print(
    "\n=========================================="
)

print(
    "NASA POWER PROCESSING COMPLETE"
)

print(
    "=========================================="
)

print(
    f"\nLocations successfully processed: "
    f"{location_features['location_id'].nunique()}"
)

print(
    f"Climate feature records: "
    f"{len(location_features)}"
)

print(
    f"\nFinal climate feature file:"
)

print(
    OUTPUT_FILE
)


if parse_failures:

    print(
        "\nFiles that could not be parsed:"
    )

    for filename in parse_failures:

        print(
            f" - {filename}"
        )