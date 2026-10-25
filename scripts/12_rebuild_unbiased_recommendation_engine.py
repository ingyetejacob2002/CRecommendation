
"""
Seven-Crop Recommendation Engine
North-Central Nigeria

Uses the trained Gradient Boosting model and the exact feature order
stored in models/model_metadata.json.

Outputs:
1. data/processed/recommendations_7_crops.csv
2. data/processed/top_crop_recommendations.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ENVIRONMENT_FILE = (
    ROOT / "data" / "processed" / "north_central_environmental.csv"
)
MAPPING_FILE = (
    ROOT / "data" / "processed" / "crop_label_mapping.csv"
)
METADATA_FILE = ROOT / "models" / "model_metadata.json"
MODEL_FILE = ROOT / "models" / "best_crop_suitability_model.joblib"

OUTPUT_DIR = ROOT / "data" / "processed"
ALL_RECOMMENDATIONS_FILE = OUTPUT_DIR / "recommendations_7_crops.csv"
TOP_RECOMMENDATIONS_FILE = OUTPUT_DIR / "top_crop_recommendations.csv"


# ---------------------------------------------------------
# 2. CHECK REQUIRED FILES
# ---------------------------------------------------------

required_files = [
    ENVIRONMENT_FILE,
    MAPPING_FILE,
    METADATA_FILE,
    MODEL_FILE,
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    raise FileNotFoundError(
        "The following required files were not found:\n"
        + "\n".join(missing_files)
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 3. LOAD METADATA, MODEL, ENVIRONMENT DATA AND CROP MAPPING
# ---------------------------------------------------------

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

model = joblib.load(MODEL_FILE)

environment = pd.read_csv(ENVIRONMENT_FILE)
crop_mapping = pd.read_csv(MAPPING_FILE)

# Recover model features from the training dataset using location_id.
TRAINING_FILE = (
    ROOT / "data" / "processed" /
    "north_central_training_dataset_7_crops.csv"
)

training_data = pd.read_csv(TRAINING_FILE)

required_recovery_columns = [
    "location_id",
    "latitude",
    "longitude",
    "mean_annual_precipitation_mm",
]

missing_recovery_columns = [
    col for col in required_recovery_columns
    if col not in training_data.columns
]

if missing_recovery_columns:
    raise ValueError(
        "Training dataset is missing required recovery columns: "
        + ", ".join(missing_recovery_columns)
    )

# Each location should have the same environmental values across crops.
# Verify this before using the training dataset to recover the fields.
recovery_data = training_data[required_recovery_columns].copy()

for col in ["latitude", "longitude", "mean_annual_precipitation_mm"]:
    variation = recovery_data.groupby("location_id")[col].nunique(
        dropna=False
    )
    inconsistent = variation[variation > 1]

    if not inconsistent.empty:
        raise ValueError(
            f"Inconsistent {col} values across crop records for "
            f"locations: {inconsistent.index.tolist()[:10]}"
        )

recovery_data = recovery_data.drop_duplicates(
    subset=["location_id"]
)

if recovery_data["location_id"].duplicated().any():
    raise ValueError(
        "Training location identifiers are not unique after deduplication."
    )

environment = environment.merge(
    recovery_data,
    on="location_id",
    how="left",
    validate="one_to_one",
)

if environment[
    ["latitude", "longitude", "mean_annual_precipitation_mm"]
].isna().any().any():
    raise ValueError(
        "Some environmental locations could not be matched to the "
        "training dataset. Check location_id values before proceeding."
    )

features = metadata["features"]
environment_features = metadata["environment_features"]
crop_order = metadata["crop_order"]

print("=" * 65)
print("SEVEN-CROP RECOMMENDATION ENGINE")
print("=" * 65)

print(f"Environmental records loaded: {len(environment)}")
print(f"Model: {metadata.get('best_model', 'Loaded model')}")
print(f"Model input features: {len(features)}")


# ---------------------------------------------------------
# 4. VALIDATE THE CROP MAPPING
# ---------------------------------------------------------

required_mapping_columns = {"crop", "crop_encoded"}

if not required_mapping_columns.issubset(crop_mapping.columns):
    raise ValueError(
        "Crop mapping must contain the columns: "
        "'crop' and 'crop_encoded'."
    )

crop_mapping["crop_encoded"] = pd.to_numeric(
    crop_mapping["crop_encoded"], errors="raise"
).astype(int)

crop_mapping = crop_mapping.sort_values("crop_encoded").reset_index(
    drop=True
)

if crop_mapping["crop"].duplicated().any():
    raise ValueError("Duplicate crop names were found in the crop mapping.")

if len(crop_mapping) != 7:
    raise ValueError(
        f"Expected exactly 7 crops, but found {len(crop_mapping)}."
    )

if crop_mapping["crop"].tolist() != crop_order:
    raise ValueError(
        "The crop mapping order does not match model_metadata.json.\n"
        f"Metadata order: {crop_order}\n"
        f"Mapping order: {crop_mapping['crop'].tolist()}"
    )

if "crop_encoded" not in features:
    raise ValueError(
        "The model metadata does not include crop_encoded as a feature."
    )

print("\nCrops loaded:")
for _, crop_row in crop_mapping.iterrows():
    print(f"  {crop_row['crop_encoded']}: {crop_row['crop']}")


# ---------------------------------------------------------
# 5. VALIDATE ENVIRONMENTAL FEATURES
# ---------------------------------------------------------

if environment.empty:
    raise ValueError("The environmental dataset contains no records.")

missing_features = [
    column
    for column in environment_features
    if column not in environment.columns
]

if missing_features:
    raise ValueError(
        "The environmental dataset is missing model input columns:\n"
        + "\n".join(missing_features)
    )

# Check that the metadata has the expected environmental features.
expected_feature_count = len(environment_features) + 1

if len(features) != expected_feature_count:
    raise ValueError(
        "Unexpected feature count in metadata. "
        f"Expected {expected_feature_count}, found {len(features)}."
    )

# Convert model input columns to numeric values.
for column in environment_features:
    environment[column] = pd.to_numeric(
        environment[column], errors="coerce"
    )

missing_counts = environment[environment_features].isna().sum()
missing_counts = missing_counts[missing_counts > 0]

if not missing_counts.empty:
    raise ValueError(
        "Missing or non-numeric values were found in model input features:\n"
        + missing_counts.to_string()
        + "\nCorrect these values before generating recommendations."
    )

if not np.isfinite(
    environment[environment_features].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Infinite values were found in environmental model inputs."
    )

print("\nEnvironmental feature validation passed.")


# ---------------------------------------------------------
# 6. PREPARE LOCATION IDENTIFIERS
# ---------------------------------------------------------

# Preserve useful identifiers from the source dataset.
identifier_candidates = [
    "location_id",
    "location_name",
    "location",
    "name",
    "state",
    "lga",
]

identifier_columns = [
    column
    for column in identifier_candidates
    if column in environment.columns
]

if "location_id" not in environment.columns:
    environment.insert(
        0,
        "location_id",
        [f"location_{i:03d}" for i in range(1, len(environment) + 1)],
    )
    identifier_columns.insert(0, "location_id")

# Ensure every source row can be ranked independently, even if
# a location identifier happens to be duplicated.
environment = environment.reset_index(drop=True)
environment["__location_row_id"] = np.arange(len(environment))

# Include latitude and longitude in the exported results when available.
for column in ["latitude", "longitude"]:
    if column in environment.columns and column not in identifier_columns:
        identifier_columns.append(column)


# ---------------------------------------------------------
# 7. CREATE ONE RECORD PER LOCATION-CROP COMBINATION
# ---------------------------------------------------------

# Each location is repeated seven times, once for every crop.
expanded = environment.loc[
    environment.index.repeat(len(crop_mapping))
].reset_index(drop=True)

expanded["crop"] = np.tile(
    crop_mapping["crop"].to_numpy(),
    len(environment),
)

expanded["crop_encoded"] = np.tile(
    crop_mapping["crop_encoded"].to_numpy(),
    len(environment),
)

# Construct the model matrix in the exact metadata feature order.
X = expanded[features].copy()

if list(X.columns) != features:
    raise ValueError("Model feature order does not match the metadata.")

print(f"\nLocation-crop combinations: {len(X)}")
print("Generating suitability predictions...")


# ---------------------------------------------------------
# 8. PREDICT SUITABILITY
# ---------------------------------------------------------

predicted_classes = model.predict(X)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)

    model_classes = list(model.classes_)

    if 1 not in model_classes:
        raise ValueError(
            "The loaded model has no class 1 for suitable predictions."
        )

    suitable_class_index = model_classes.index(1)
    suitability_scores = probabilities[:, suitable_class_index]

else:
    raise AttributeError(
        "The loaded model does not support predict_proba(). "
        "This script requires a probability-capable classifier."
    )

expanded["predicted_suitable"] = np.asarray(
    predicted_classes
).astype(int)

expanded["suitability_score_percent"] = (
    suitability_scores * 100
).round(2)

expanded["predicted_status"] = np.where(
    expanded["predicted_suitable"] == 1,
    "Suitable",
    "Not suitable",
)


# ---------------------------------------------------------
# 9. RANK CROPS WITHIN EACH LOCATION
# ---------------------------------------------------------

expanded["recommendation_rank"] = (
    expanded.groupby("__location_row_id")["suitability_score_percent"]
    .rank(method="first", ascending=False)
    .astype(int)
)

# Keep useful columns in the full output.
output_columns = (
    identifier_columns
    + [
        "crop",
        "crop_encoded",
        "predicted_status",
        "predicted_suitable",
        "suitability_score_percent",
        "recommendation_rank",
    ]
)

# Avoid accidentally exporting duplicate column names.
output_columns = list(dict.fromkeys(output_columns))

all_results = expanded[output_columns].copy()

all_results = all_results.sort_values(
    ["location_id", "recommendation_rank"],
    kind="stable",
).reset_index(drop=True)


# ---------------------------------------------------------
# 10. SAVE ALL SEVEN CROPS AND TOP THREE PER LOCATION
# ---------------------------------------------------------

all_results.to_csv(
    ALL_RECOMMENDATIONS_FILE,
    index=False,
)

top_results = all_results[
    all_results["recommendation_rank"] <= 3
].copy()

top_results.to_csv(
    TOP_RECOMMENDATIONS_FILE,
    index=False,
)


# ---------------------------------------------------------
# 11. PRINT A SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("RECOMMENDATION ENGINE COMPLETED")
print("=" * 65)

print(f"Locations processed: {len(environment)}")
print(f"Crops evaluated: {len(crop_mapping)}")
print(f"Total evaluations: {len(all_results)}")
print(f"All recommendations saved to:\n  {ALL_RECOMMENDATIONS_FILE}")
print(f"Top-three recommendations saved to:\n  {TOP_RECOMMENDATIONS_FILE}")

print("\nSample top recommendations:")

display_columns = [
    column
    for column in [
        "location_id",
        "location_name",
        "state",
        "crop",
        "suitability_score_percent",
        "predicted_status",
        "recommendation_rank",
    ]
    if column in top_results.columns
]

print(top_results[display_columns].head(15).to_string(index=False))

print("\nSuitability status counts:")
print(all_results["predicted_status"].value_counts().to_string())

print(
    "\nIMPORTANT: Scores are model-predicted suitability scores, "
    "not measured crop yields or independently validated probabilities."
)

if len(all_results) != len(environment) * 7:
    raise RuntimeError(
        "Unexpected number of recommendation rows. "
        "Check the environmental dataset and crop mapping."
    )

print("\nValidation passed: all locations were evaluated against seven crops.")
