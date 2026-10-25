
"""
01_download_locations.py

Purpose
-------
1. Download Nigeria ADM1 (states) and ADM2 (LGAs) boundaries.
2. Identify the North-Central states.
3. Spatially join each LGA to its parent state.
4. Calculate a representative coordinate for each LGA.
5. Save a clean LGA coordinate dataset.

Output
------
data/processed/north_central_lgas.csv
"""

from pathlib import Path
import io
import zipfile

import requests
import pandas as pd
import geopandas as gpd


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. NORTH-CENTRAL STATES
# ============================================================

NORTH_CENTRAL_STATES = {
    "Benue",
    "Kogi",
    "Kwara",
    "Nasarawa",
    "Niger",
    "Plateau",
    "Federal Capital Territory",
    "FCT",
}


# ============================================================
# 3. GEOBOUNDARIES URLS
# ============================================================

ADM1_URL = (
    "https://www.geoboundaries.org/"
    "data/1_3_3/zip/geojson/NGA/"
    "NGA_ADM1.geojson.zip"
)

ADM2_URL = (
    "https://www.geoboundaries.org/"
    "data/1_3_3/zip/geojson/NGA/"
    "NGA_ADM2.geojson.zip"
)


# ============================================================
# 4. DOWNLOAD FUNCTION
# ============================================================

def download_and_extract(url, output_directory, label):
    """
    Download and extract a GeoBoundaries ZIP file.
    """

    print(f"\nDownloading {label} boundaries...")

    response = requests.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    zip_file = output_directory / f"{label}.zip"

    with open(zip_file, "wb") as file:
        file.write(response.content)

    extract_directory = (
        output_directory / label
    )

    extract_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    with zipfile.ZipFile(
        io.BytesIO(response.content)
    ) as archive:

        archive.extractall(
            extract_directory
        )

    geojson_files = list(
        extract_directory.glob("*.geojson")
    )

    if not geojson_files:
        raise FileNotFoundError(
            f"No GeoJSON file found for {label}."
        )

    print(
        f"{label} downloaded successfully."
    )

    return geojson_files[0]


# ============================================================
# 5. DOWNLOAD ADM1 AND ADM2
# ============================================================

print("=" * 70)
print("NORTH-CENTRAL NIGERIA LOCATION DATASET")
print("=" * 70)

print("\n[1/7] Downloading administrative boundaries...")

adm1_file = download_and_extract(
    ADM1_URL,
    RAW_DIR,
    "NGA_ADM1"
)

adm2_file = download_and_extract(
    ADM2_URL,
    RAW_DIR,
    "NGA_ADM2"
)


# ============================================================
# 6. READ FILES
# ============================================================

print("\n[2/7] Reading administrative boundaries...")

states = gpd.read_file(adm1_file)

lgas = gpd.read_file(adm2_file)

print(
    f"Number of states/ADM1 records: {len(states)}"
)

print(
    f"Number of LGAs/ADM2 records: {len(lgas)}"
)

print("\nADM1 columns:")
print(states.columns.tolist())

print("\nADM2 columns:")
print(lgas.columns.tolist())


# ============================================================
# 7. INSPECT STATE NAMES
# ============================================================

print("\n[3/7] Inspecting Nigerian state names...")

print(
    "\nState names found in ADM1:"
)

print(
    states["Name"]
    .sort_values()
    .to_string(index=False)
)


# ============================================================
# 8. FILTER NORTH-CENTRAL STATES
# ============================================================

north_central_states = states[
    states["Name"].isin(
        NORTH_CENTRAL_STATES
    )
].copy()


if north_central_states.empty:

    print(
        "\nERROR: No North-Central states matched."
    )

    print(
        "\nActual state names in the dataset:"
    )

    print(
        states["Name"]
        .unique()
    )

    raise RuntimeError(
        "North-Central state names need to be "
        "mapped to the names used by the boundary file."
    )


print(
    "\nNorth-Central states found:"
)

print(
    north_central_states[
        ["Name"]
    ].sort_values("Name")
    .to_string(index=False)
)


# ============================================================
# 9. PREPARE COORDINATE SYSTEM
# ============================================================

print(
    "\n[4/7] Preparing coordinate systems..."
)

states = states.to_crs(
    "EPSG:4326"
)

lgas = lgas.to_crs(
    "EPSG:4326"
)

north_central_states = (
    north_central_states.to_crs(
        "EPSG:4326"
    )
)


# ============================================================
# 10. CREATE REPRESENTATIVE POINTS
# ============================================================

print(
    "\n[5/7] Calculating LGA representative points..."
)

lgas["representative_point"] = (
    lgas.geometry.representative_point()
)


# ============================================================
# 11. SPATIAL JOIN LGAs TO STATES
# ============================================================

print(
    "\nPerforming spatial join between "
    "LGAs and states..."
)

lga_points = lgas[
    [
        "id",
        "Name",
        "geometry",
        "representative_point",
    ]
].copy()


# Make representative points the active geometry.

lga_points = lga_points.set_geometry(
    "representative_point"
)


joined = gpd.sjoin(
    lga_points,
    north_central_states[
        ["Name", "geometry"]
    ],
    how="inner",
    predicate="within",
    lsuffix="lga",
    rsuffix="state",
)


# ============================================================
# 12. CHECK SPATIAL JOIN
# ============================================================

print(
    f"\nLGAs belonging to North-Central states: "
    f"{len(joined)}"
)

if joined.empty:

    raise RuntimeError(
        "Spatial join returned zero LGAs. "
        "The boundary datasets may have incompatible "
        "coordinate systems or geometries."
    )


# ============================================================
# 13. EXTRACT COORDINATES
# ============================================================

joined["longitude"] = (
    joined.geometry.x
)

joined["latitude"] = (
    joined.geometry.y
)


# ============================================================
# 14. CREATE CLEAN DATAFRAME
# ============================================================

locations = joined[
    [
        "Name_state",
        "Name_lga",
        "latitude",
        "longitude",
    ]
].copy()


locations.columns = [
    "state",
    "lga",
    "latitude",
    "longitude",
]


# ============================================================
# 15. CLEAN TEXT
# ============================================================

locations["state"] = (
    locations["state"]
    .astype(str)
    .str.strip()
)

locations["lga"] = (
    locations["lga"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 16. REMOVE DUPLICATES
# ============================================================

locations = locations.drop_duplicates(
    subset=[
        "state",
        "lga",
    ]
)


# ============================================================
# 17. SORT DATA
# ============================================================

locations = locations.sort_values(
    [
        "state",
        "lga",
    ]
).reset_index(drop=True)


# ============================================================
# 18. CREATE LOCATION ID
# ============================================================

locations.insert(
    0,
    "location_id",
    range(
        1,
        len(locations) + 1
    )
)


# ============================================================
# 19. SAVE CSV
# ============================================================

output_file = (
    PROCESSED_DIR /
    "north_central_lgas.csv"
)

locations.to_csv(
    output_file,
    index=False
)


# ============================================================
# 20. FINAL VALIDATION
# ============================================================

print(
    "\n[6/7] Validating dataset..."
)

print(
    f"\nTotal North-Central LGAs: "
    f"{len(locations)}"
)

print(
    "\nLGAs per state:"
)

print(
    locations.groupby("state")
    .size()
    .sort_values(
        ascending=False
    )
)


print(
    "\nMissing values:"
)

print(
    locations.isna()
    .sum()
)


# ============================================================
# 21. DISPLAY SAMPLE
# ============================================================

print(
    "\nFirst 15 records:"
)

print(
    locations.head(15)
    .to_string(index=False)
)


# ============================================================
# 22. COMPLETION
# ============================================================

print(
    "\n[7/7] Dataset saved successfully."
)

print(
    f"\nFile:"
)

print(
    output_file
)

print(
    "\n" + "=" * 70
)

print(
    "SCRIPT 1 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)