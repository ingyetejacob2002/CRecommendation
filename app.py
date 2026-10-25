
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "north_central_training_dataset_7_crops.csv"
)

RECOMMENDATION_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "recommendations_7_crops.csv"
)

TOP_RECOMMENDATIONS_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "top_crop_recommendations.csv"
)

MAPPING_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "crop_label_mapping.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models"
    / "best_crop_suitability_model.joblib"
)

# ============================================================
# ENVIRONMENTAL FEATURES
# Keep these names identical to the model's training features.
# ============================================================

ENVIRONMENTAL_FEATURES = [
    "latitude",
    "longitude",
    "mean_daily_precipitation_mm",
    "std_daily_precipitation_mm",
    "min_daily_precipitation_mm",
    "max_daily_precipitation_mm",
    "total_precipitation_mm",
    "mean_annual_precipitation_mm",
    "mean_temperature_c",
    "std_temperature_c",
    "min_temperature_c",
    "max_temperature_c",
    "mean_max_temperature_c",
    "absolute_max_temperature_c",
    "mean_min_temperature_c",
    "absolute_min_temperature_c",
    "mean_relative_humidity_percent",
    "minimum_relative_humidity_percent",
    "maximum_relative_humidity_percent",
    "mean_solar_radiation",
    "minimum_solar_radiation",
    "maximum_solar_radiation",
    "mean_wind_speed",
    "minimum_wind_speed",
    "maximum_wind_speed",
    "soil_ph",
    "soil_organic_carbon_g_kg",
    "soil_nitrogen_g_kg",
    "clay_percent",
    "sand_percent",
    "silt_percent",
    "cec_cmolc_kg",
    "bulk_density_g_cm3",
]

MODEL_FEATURES = ENVIRONMENTAL_FEATURES + ["crop_encoded"]

# The new model contains exactly these seven crops.
CROPS = [
    "Bambara_groundnut",
    "Cassava",
    "Cowpea",
    "Groundnut",
    "Rice",
    "Soybean",
    "Yam",
]

# ============================================================
# GLOBAL RESOURCES
# ============================================================

model = None
training_df = None
recommendation_df = None
top_recommendations_df = None

crop_to_code = {}
code_to_crop = {}
location_profiles = []

startup_error = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def model_feature_names():
    """Use the feature order recorded by the trained model."""
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return list(names)

    return MODEL_FEATURES.copy()


def get_suitability_scores(model_object, frame):
    """
    Return the model's estimated suitability scores.

    For classifiers with predict_proba, use the probability
    associated with class 1 (Suitable).
    """
    if hasattr(model_object, "predict_proba"):
        probabilities = model_object.predict_proba(frame)
        classes = list(model_object.classes_)

        if 1 in classes:
            return probabilities[:, classes.index(1)]

        return probabilities[:, -1]

    if hasattr(model_object, "decision_function"):
        scores = np.asarray(
            model_object.decision_function(frame),
            dtype=float
        )

        if scores.ndim > 1:
            scores = scores[:, -1]

        # Convert decision scores to a 0-1 scale.
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))

    return np.asarray(model_object.predict(frame), dtype=float)


# ============================================================
# LOAD MODEL, DATASETS AND CROP MAPPING
# ============================================================

def load_resources():
    global model
    global training_df
    global recommendation_df
    global top_recommendations_df
    global crop_to_code
    global code_to_crop
    global location_profiles
    global startup_error

    try:
        # Check required files.
        required_files = [MODEL_PATH, DATASET_PATH, MAPPING_PATH]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file not found:\n{file_path}"
                )

        # Load the new trained model.
        model = joblib.load(MODEL_PATH)

        # Load datasets.
        training_df = pd.read_csv(DATASET_PATH)

        if RECOMMENDATION_DATASET_PATH.exists():
            recommendation_df = pd.read_csv(
                RECOMMENDATION_DATASET_PATH
            )

        if TOP_RECOMMENDATIONS_PATH.exists():
            top_recommendations_df = pd.read_csv(
                TOP_RECOMMENDATIONS_PATH
            )

        # Load the actual crop encoding used during training.
        mapping_df = pd.read_csv(MAPPING_PATH)

        if not {"crop", "crop_encoded"}.issubset(mapping_df.columns):
            raise ValueError(
                "crop_label_mapping.csv must contain "
                "'crop' and 'crop_encoded' columns."
            )

        mapping_df = mapping_df.drop_duplicates(
            subset=["crop"]
        )

        crop_to_code = {
            str(row["crop"]): int(row["crop_encoded"])
            for _, row in mapping_df.iterrows()
        }

        if set(crop_to_code) != set(CROPS):
            raise ValueError(
                "The crop mapping does not match the seven crops "
                "configured in app.py. Check crop_label_mapping.csv."
            )

        code_to_crop = {
            code: crop for crop, code in crop_to_code.items()
        }

        # Check that the model features match this application.
        expected_features = set(model_feature_names())

        missing_features = expected_features - set(MODEL_FEATURES)

        if missing_features:
            raise ValueError(
                "The model expects features not configured in app.py: "
                f"{sorted(missing_features)}"
            )

        # Build saved location profiles for the dropdown/API.
        location_column = find_column(
            training_df,
            ["location_id", "location", "locationid", "id"]
        )

        if location_column:
            profiles = []

            for location_id, group in training_df.groupby(
                location_column, sort=True
            ):
                row = group.iloc[0]

                profile = {"location_id": str(location_id)}

                for feature in ENVIRONMENTAL_FEATURES:
                    if feature in training_df.columns:
                        value = row[feature]
                        if pd.notna(value):
                            profile[feature] = float(value)

                for extra in ["state", "lga"]:
                    if extra in training_df.columns and pd.notna(row[extra]):
                        profile[extra] = str(row[extra])

                profiles.append(profile)

            location_profiles = profiles

        startup_error = None

    except Exception as exc:
        startup_error = str(exc)


load_resources()


# ============================================================
# BUILD INPUT FOR ALL SEVEN CROPS
# ============================================================

def build_input_frame(environment):
    features = model_feature_names()
    rows = []

    for crop in CROPS:
        row = {}

        for feature in ENVIRONMENTAL_FEATURES:
            if feature not in environment:
                raise ValueError(
                    f"Missing environmental feature: {feature}"
                )

            value = float(environment[feature])

            if not np.isfinite(value):
                raise ValueError(
                    f"{feature} must be a finite numeric value."
                )

            row[feature] = value

        if "crop_encoded" in features:
            row["crop_encoded"] = crop_to_code[crop]

        rows.append(row)

    frame = pd.DataFrame(rows)

    missing = [feature for feature in features if feature not in frame]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return frame[features]


# ============================================================
# GENERATE CROP RECOMMENDATIONS
# ============================================================

def make_recommendations(environment):
    if model is None:
        raise RuntimeError("The new crop suitability model is not loaded.")

    frame = build_input_frame(environment)

    scores = get_suitability_scores(model, frame)
    predictions = model.predict(frame)

    results = []

    for index, crop in enumerate(CROPS):
        score = float(np.clip(scores[index], 0.0, 1.0))
        prediction = int(predictions[index])

        results.append({
            "crop": crop,
            "prediction": prediction,
            "suitability_probability": score,
            "suitability_percentage": round(score * 100, 2),
            "suitability_label": (
                "Suitable" if prediction == 1 else "Not Suitable"
            ),
        })

    # Rank the crops by their estimated suitability score.
    results.sort(
        key=lambda item: item["suitability_probability"],
        reverse=True
    )

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

        if rank == 1:
            item["category"] = "Best Recommendation"
        elif rank <= 3:
            item["category"] = "Strong Alternative"
        elif rank <= 5:
            item["category"] = "Alternative"
        else:
            item["category"] = "Lower Priority"

    return results


# ============================================================
# PARSE FORM DATA
# ============================================================

def parse_environment(form):
    values = {}

    for feature in ENVIRONMENTAL_FEATURES:
        raw = form.get(feature, "").strip()

        if not raw:
            raise ValueError(f"Please provide a value for {feature}.")

        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be numeric.")

        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")

        values[feature] = value

    return values


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        crops=CROPS,
        environmental_features=ENVIRONMENTAL_FEATURES,
        locations=location_profiles,
        startup_error=startup_error,
        form_data={}
    )


# ============================================================
# HTML RECOMMENDATION
# ============================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    if startup_error:
        return render_template(
            "index.html",
            crops=CROPS,
            environmental_features=ENVIRONMENTAL_FEATURES,
            locations=location_profiles,
            startup_error=startup_error,
            form_data=request.form
        ), 500

    try:
        environment = parse_environment(request.form)
        results = make_recommendations(environment)

        return render_template(
            "recommendation.html",
            results=results,
            environment=environment,
            top1=results[0],
            top3=results[:3],
            top5=results[:5]
        )

    except Exception as exc:
        return render_template(
            "index.html",
            crops=CROPS,
            environmental_features=ENVIRONMENTAL_FEATURES,
            locations=location_profiles,
            startup_error=str(exc),
            form_data=request.form
        ), 400


# ============================================================
# JSON RECOMMENDATION API
# ============================================================

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    if startup_error:
        return jsonify({
            "success": False,
            "error": startup_error
        }), 500

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError("Send the environmental values as a JSON object.")

        environment = {}

        for feature in ENVIRONMENTAL_FEATURES:
            if feature not in data:
                raise ValueError(f"Missing feature: {feature}")

            try:
                value = float(data[feature])
            except (TypeError, ValueError):
                raise ValueError(f"{feature} must be numeric.")

            if not np.isfinite(value):
                raise ValueError(f"{feature} must be a finite number.")

            environment[feature] = value

        results = make_recommendations(environment)

        return jsonify({
            "success": True,
            "recommendations": results,
            "top1": results[0],
            "top3": results[:3],
            "top5": results[:5]
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


# ============================================================
# LOCATION PROFILE API
# ============================================================

@app.route("/api/profile/<location_id>")
def profile(location_id):
    for item in location_profiles:
        if str(item["location_id"]) == str(location_id):
            return jsonify({
                "success": True,
                "profile": item
            })

    return jsonify({
        "success": False,
        "error": "Location not found"
    }), 404


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# SYSTEM HEALTH
# ============================================================

@app.route("/health")
def health():
    payload = {
        "status": "ok" if startup_error is None else "error",
        "model_loaded": model is not None,
        "training_dataset_loaded": training_df is not None,
        "recommendation_dataset_loaded": recommendation_df is not None,
        "top_recommendations_loaded": top_recommendations_df is not None,
        "number_of_crops": len(CROPS),
        "number_of_location_profiles": len(location_profiles),
        "error": startup_error
    }

    wants_json = (
        request.args.get("format") == "json"
        or request.accept_mimetypes.best_match(
            ["application/json", "text/html"]
        ) == "application/json"
    )

    if wants_json:
        return jsonify(payload), (
            200 if startup_error is None else 500
        )

    return render_template("status.html", health=payload), (
        200 if startup_error is None else 500
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("NORTH-CENTRAL SEVEN-CROP RECOMMENDATION SYSTEM")
    print("=" * 65)
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print(f"Training dataset loaded: {training_df is not None}")
    print(f"Number of crops: {len(CROPS)}")

    if startup_error:
        print("\nSTARTUP ERROR:")
        print(startup_error)
    else:
        print(f"Location profiles loaded: {len(location_profiles)}")

    print("\nOpen http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
