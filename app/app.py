"""Tempest FWI Predictor - Flask web app (HMI).

    python app/app.py            -> http://localhost:5000
    POST /api/predict  (JSON)    -> {"fwi": 12.3, "danger_class": "High", ...}
    GET  /forecast?place=Kelowna -> 3-day fire danger forecast page
    GET  /api/forecast?place=... or ?lat=..&lon=.. -> forecast JSON

Replaces FWI_HMI.ipynb: one model file (the full sklearn pipeline) instead of
four pickles and hard-coded Windows paths.
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_FEATURES  # noqa: E402
from src.fwi_equations import fwi_from_isi_bui  # noqa: E402
import src.forecast as fc  # noqa: E402

MODEL_PATH = Path(os.environ.get("FWI_MODEL_PATH", ROOT / "models" / "fwi_model.joblib"))

app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# label, unit, min, max, example value
FIELDS = {
    "temperature": ("Temperature", "°C", -60, 50, 22.5),
    "dew_point_temperature": ("Dew point temperature", "°C", -70, 40, 8.0),
    "relative_humidity": ("Relative humidity", "%", 0, 100, 40),
    "wind_speed": ("Wind speed", "km/h", 0, 150, 15),
    "atmospheric_pressure": ("Atmospheric pressure", "hPa", 850, 1090, 1012),
    "precipitation": ("Precipitation (24 h)", "mm", 0, 300, 0),
    "rain_days": ("Rain days (CWFIS rndays)", "days", 0, 365, 3),
    "fine_fuel_moisture_code": ("FFMC", "", 0, 101, 88),
    "duff_moisture_code": ("DMC", "", 0, 1000, 35),
    "drought_code": ("DC", "", 0, 2000, 250),
    "build_up_index": ("BUI", "", 0, 1000, 50),
    "initial_spread_index": ("ISI", "", 0, 200, 6),
}
COUNTRIES = ["Canada", "United States"]

DANGER = [(5, "Low", "#2e7d32"), (10, "Moderate", "#1565c0"), (20, "High", "#f9a825"),
          (30, "Very high", "#ef6c00"), (float("inf"), "Extreme", "#c62828")]


def danger(fwi: float):
    for upper, label, color in DANGER:
        if fwi < upper:
            return label, color


def validate(data: dict) -> dict:
    values, errors = {}, []
    for key, (label, _, lo, hi, _) in FIELDS.items():
        try:
            v = float(data[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} is missing or not a number")
            continue
        if not lo <= v <= hi:
            errors.append(f"{label} must be between {lo} and {hi}")
        values[key] = v
    country = data.get("country", "Canada")
    if country not in COUNTRIES:
        errors.append(f"Country must be one of {COUNTRIES}")
    values["country"] = country
    if errors:
        raise ValueError("; ".join(errors))
    return values


def predict(values: dict) -> dict:
    X = pd.DataFrame([values])[MODEL_FEATURES]
    fwi = max(0.0, float(model.predict(X)[0]))
    label, color = danger(fwi)
    reference = float(fwi_from_isi_bui(values["initial_spread_index"], values["build_up_index"]))
    return {"fwi": round(fwi, 2), "danger_class": label, "color": color,
            "equation_fwi": round(reference, 2)}


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    form = {k: v[4] for k, v in FIELDS.items()} | {"country": "Canada"}
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            result = predict(validate(form))
        except ValueError as e:
            error = str(e)
    return render_template("index.html", fields=FIELDS, countries=COUNTRIES,
                           form=form, result=result, error=error)


@app.post("/api/predict")
def api_predict():
    try:
        return jsonify(predict(validate(request.get_json(force=True) or {})))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def run_forecast(args):
    place = (args.get("place") or "").strip()
    lat, lon = args.get("lat"), args.get("lon")
    if place:
        return fc.forecast_for_place(place)
    if lat and lon:
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Latitude/longitude out of range")
        return fc.forecast(lat, lon, args.get("country_code", ""))
    raise ValueError("Give a place name, or lat and lon")


@app.get("/forecast")
def forecast_page():
    result, error = None, None
    if request.args.get("place") or request.args.get("lat"):
        try:
            result = run_forecast(request.args)
        except (ValueError, RuntimeError) as e:
            error = str(e)
        except Exception as e:  # network problems etc.
            error = f"Couldn't get the weather right now ({type(e).__name__}). Try again shortly."
    return render_template("forecast.html", result=result, error=error,
                           place=request.args.get("place", ""))


@app.get("/api/forecast")
def api_forecast():
    try:
        return jsonify(run_forecast(request.args))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"upstream weather service failed: {e}"}), 502


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
