"""Central configuration: paths, column names and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FWI_FILE = RAW_DIR / "cwfis_fwi2000s.csv"
RAW_STATIONS_FILE = RAW_DIR / "cwfis_allstn2025.csv"

RANDOM_STATE = 42

# CWFIS short names -> readable names (same mapping as data_cleaning.ipynb)
COLUMN_MAPPING = {
    "rep_date": "report_date",
    "temp": "temperature",
    "td": "dew_point_temperature",
    "rh": "relative_humidity",
    "ws": "wind_speed",
    "pres": "atmospheric_pressure",
    "precip": "precipitation",
    "rndays": "rain_days",
    "ffmc": "fine_fuel_moisture_code",
    "dmc": "duff_moisture_code",
    "dc": "drought_code",
    "bui": "build_up_index",
    "isi": "initial_spread_index",
    "fwi": "fire_weather_index",
}

TARGET = "fire_weather_index"

WEATHER_FEATURES = [
    "temperature", "dew_point_temperature", "relative_humidity", "wind_speed",
    "atmospheric_pressure", "precipitation", "rain_days",
]
FWI_SYSTEM_FEATURES = [
    "fine_fuel_moisture_code", "duff_moisture_code", "drought_code",
    "build_up_index", "initial_spread_index",
]
CATEGORICAL_FEATURES = ["country"]

# The 13 inputs the web form (HMI) asks for
MODEL_FEATURES = WEATHER_FEATURES + FWI_SYSTEM_FEATURES + CATEGORICAL_FEATURES

# Heavily skewed columns that get a Yeo-Johnson power transform (as in the notebooks)
SKEWED_FEATURES = FWI_SYSTEM_FEATURES + ["wind_speed"]

CANADIAN_PROVINCES = {"BC", "AB", "SA", "SK", "MB", "ON", "QC", "NB", "NS", "PE",
                      "NF", "NL", "YT", "NT", "NU"}

# Physically plausible ranges used to drop sensor / transcription errors
VALID_RANGES = {
    "temperature": (-60, 50),
    "dew_point_temperature": (-70, 40),
    "relative_humidity": (0, 100),
    "wind_speed": (0, 150),
    "atmospheric_pressure": (850, 1090),
    "precipitation": (0, 300),
    "rain_days": (0, 365),
    "fine_fuel_moisture_code": (0, 101),
    "duff_moisture_code": (0, 1000),
    "drought_code": (0, 2000),
    "build_up_index": (0, 1000),
    "initial_spread_index": (0, 200),
    "fire_weather_index": (0, 250),
}
