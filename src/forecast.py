"""Upgrade 3 - 3-day fire danger forecast for any location.

Pulls the last 92 days of weather plus the next days' forecast from the free
Open-Meteo API (no key needed), rebuilds the same rolling-history features the
weather-only model was trained on, and predicts FWI for each day.

    python -m src.forecast "Kelowna"
    python -m src.forecast --lat 53.55 --lon -113.49
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src.config import MODELS_DIR
from src.weather_features import WEATHER_ONLY_FEATURES, add_history

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOURLY = ["temperature_2m", "dew_point_2m", "relative_humidity_2m", "wind_speed_10m",
          "pressure_msl", "precipitation"]
RENAME = {"temperature_2m": "temperature", "dew_point_2m": "dew_point_temperature",
          "relative_humidity_2m": "relative_humidity", "wind_speed_10m": "wind_speed",
          "pressure_msl": "atmospheric_pressure"}
COUNTRY_CODES = {"CA": "Canada", "US": "United States"}
DANGER = [(5, "Low", "#2e7d32"), (10, "Moderate", "#1565c0"), (20, "High", "#f9a825"),
          (30, "Very high", "#ef6c00"), (float("inf"), "Extreme", "#c62828")]

_model = None


def model():
    global _model
    if _model is None:
        _model = joblib.load(Path(MODELS_DIR) / "fwi_weather_only.joblib")
    return _model


def danger(fwi):
    for upper, label, color in DANGER:
        if fwi < upper:
            return label, color


def geocode(name: str) -> dict:
    r = requests.get(GEOCODE_URL, params={"name": name, "count": 1}, timeout=15)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        raise ValueError(f"Couldn't find a place called '{name}'")
    p = results[0]
    label = ", ".join(x for x in [p.get("name"), p.get("admin1"), p.get("country")] if x)
    return {"latitude": p["latitude"], "longitude": p["longitude"], "label": label,
            "country_code": p.get("country_code", "")}


def fetch_hourly(lat: float, lon: float, forecast_days: int = 4) -> dict:
    r = requests.get(FORECAST_URL, params={
        "latitude": lat, "longitude": lon, "hourly": ",".join(HOURLY),
        "past_days": 92, "forecast_days": forecast_days, "timezone": "auto",
        "wind_speed_unit": "kmh"}, timeout=20)
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Weather service error: {data.get('reason')}")
    return data


def daily_noon(data: dict) -> pd.DataFrame:
    """CWFIS uses noon observations and 24 h rain ending at noon - rebuild that."""
    h = pd.DataFrame(data["hourly"]).rename(columns=RENAME)
    h["time"] = pd.to_datetime(h["time"])
    # precipitation is 'preceding hour' - hours 13:00 (prev day) .. 12:00 belong to that noon
    h["noon_day"] = (h["time"] - pd.Timedelta(hours=13)).dt.normalize() + pd.Timedelta(days=1)
    rain = h.groupby("noon_day")["precipitation"].sum(min_count=1)
    noon = h[h["time"].dt.hour == 12].copy()
    noon["report_date"] = noon["time"].dt.normalize()
    noon["precipitation"] = noon["report_date"].map(rain).fillna(0)
    cols = ["report_date", "temperature", "dew_point_temperature", "relative_humidity",
            "wind_speed", "atmospheric_pressure", "precipitation"]
    return noon[cols].dropna().reset_index(drop=True)


def forecast(lat: float, lon: float, country_code: str = "", label: str | None = None,
             past_days_shown: int = 7, data: dict | None = None) -> dict:
    data = data or fetch_hourly(lat, lon)
    days = daily_noon(data)
    days["wmo"] = 0
    days["latitude"], days["longitude"] = lat, lon
    days["elevation"] = data.get("elevation", 0)
    days["country"] = COUNTRY_CODES.get(country_code.upper(), "Other")
    feats = add_history(days)
    feats["fwi"] = np.clip(model().predict(feats[WEATHER_ONLY_FEATURES]), 0, None)

    today = pd.Timestamp.now(tz=data.get("timezone") or "UTC").tz_localize(None).normalize()
    feats = feats[feats["report_date"] >= today - pd.Timedelta(days=past_days_shown)]
    out = []
    for _, r in feats.iterrows():
        lab, col = danger(r["fwi"])
        out.append({"date": r["report_date"].strftime("%Y-%m-%d"),
                    "kind": "forecast" if r["report_date"] > today else
                            ("today" if r["report_date"] == today else "past"),
                    "fwi": round(float(r["fwi"]), 1), "danger_class": lab, "color": col,
                    "temperature": round(float(r["temperature"]), 1),
                    "relative_humidity": round(float(r["relative_humidity"])),
                    "wind_speed": round(float(r["wind_speed"]), 1),
                    "precipitation": round(float(r["precipitation"]), 1),
                    "days_since_rain": int(r["days_since_rain"])})
    note = None
    if country_code.upper() not in COUNTRY_CODES:
        note = ("The model was trained on Canadian and US stations; outside North America "
                "treat the numbers as indicative only.")
    return {"location": label or f"{lat:.3f}, {lon:.3f}", "latitude": lat, "longitude": lon,
            "timezone": data.get("timezone"), "days": out, "note": note}


def forecast_for_place(name: str) -> dict:
    p = geocode(name)
    return forecast(p["latitude"], p["longitude"], p["country_code"], p["label"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("place", nargs="?")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    a = ap.parse_args()
    res = forecast_for_place(a.place) if a.place else forecast(a.lat, a.lon)
    print(res["location"])
    for d in res["days"]:
        print(f"{d['date']} {d['kind']:8s} FWI {d['fwi']:5.1f} {d['danger_class']:9s} "
              f"T {d['temperature']}°C RH {d['relative_humidity']}% wind {d['wind_speed']} "
              f"rain {d['precipitation']} mm")
    if res["note"]:
        print("Note:", res["note"])
