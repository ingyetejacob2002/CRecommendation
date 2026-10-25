from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "north_central_climate.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "north_central_climate_clean.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/5] Loading climate feature dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nClimate file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"Rows before cleaning: {len(df)}")
print(f"Unique location IDs: {df['location_id'].nunique()}")

# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\n[2/5] Checking duplicate location IDs...")

duplicates = df[df.duplicated("location_id", keep=False)].sort_values(
    "location_id"
)

if len(duplicates) > 0:
    print("\nDuplicate records found:")
    print(
        duplicates[
            ["location_id", "state", "lga"]
        ].to_string(index=False)
    )
else:
    print("No duplicate location IDs found.")

# ============================================================
# REMOVE DUPLICATES
# ============================================================

print("\n[3/5] Removing duplicate location records...")

# The location_id is our authoritative identifier.
# Keep the first valid record for each location_id.

df_clean = (
    df.sort_values("location_id")
      .drop_duplicates(
          subset=["location_id"],
          keep="first"
      )
      .reset_index(drop=True)
)

print(f"Rows after removing duplicates: {len(df_clean)}")
print(
    f"Unique location IDs after cleaning: "
    f"{df_clean['location_id'].nunique()}"
)

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n[4/5] Checking missing values...")

missing = df_clean.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print("Missing values:")
    print(missing)

# ============================================================
# SAVE
# ============================================================

print("\n[5/5] Saving cleaned climate dataset...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df_clean.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("CLIMATE DATA CLEANING COMPLETED")
print("========================================")

print(f"Original records : {len(df)}")
print(f"Clean records    : {len(df_clean)}")
print(f"Unique locations : {df_clean['location_id'].nunique()}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(df_clean.columns.tolist())

print("\nFirst 5 records:")
print(df_clean.head())