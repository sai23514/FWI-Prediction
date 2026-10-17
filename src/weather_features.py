"""Upgrade 1 - build a weather-only dataset (no FFMC/DMC/DC/BUI/ISI inputs).

The FWI codes are "memory" of past weather: FFMC ~ last few days of drying,
DMC ~ weeks, DC ~ months. Instead of feeding the codes, we give the model that
memory directly as per-station rolling weather statistics.

    python -m src.weather_features
Writes data/processed/weather_only.parquet
"""
import numpy as np
import pandas as pd

from src.config import (CANADIAN_PROVINCES, COLUMN_MAPPING, PROCESSED_DIR, RAW_FWI_FILE,
                        RAW_STATIONS_FILE, VALID_RANGES)

WEATHER = ["temperature", "dew_point_temperature", "relative_humidity", "wind_speed",
           "atmospheric_pressure", "precipitation"]

RAIN_WINDOWS = [3, 7, 14, 30, 60]
MEAN_WINDOWS = [3, 7, 30]


def load_weather() -> pd.DataFrame:
    cols = ["wmo", "rep_date", "temp", "td", "rh", "ws", "pres", "precip", "fwi"]
    df = pd.read_csv(RAW_FWI_FILE, usecols=cols,
                     dtype={c: "float32" for c in cols if c not in ("wmo", "rep_date")})
    df = df.rename(columns=COLUMN_MAPPING)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.normalize()
    # bad weather values -> NaN (kept as rows so the station history stays continuous)
    for c in WEATHER:
        lo, hi = VALID_RANGES[c]
        df.loc[~df[c].between(lo, hi), c] = np.nan
    df = df.drop_duplicates(["wmo", "report_date"]).sort_values(["wmo", "report_date"])
    return df.reset_index(drop=True)


def add_station_info(df):
    st = pd.read_csv(RAW_STATIONS_FILE, usecols=["wmo", "prov", "lat", "lon", "elev"])
    st = st.dropna(subset=["wmo"]).drop_duplicates("wmo")
    st["prov"] = st["prov"].str.strip()
    st["country"] = np.where(st["prov"].isin(CANADIAN_PROVINCES), "Canada",
                             np.where(st["prov"].str.fullmatch(r"[A-Z]{2}") & (st["prov"] != "XX"),
                                      "United States", None))
    st = st.rename(columns={"lat": "latitude", "lon": "longitude", "elev": "elevation"})
    return df.merge(st[["wmo", "country", "latitude", "longitude", "elevation"]], on="wmo",
                    how="inner").dropna(subset=["country"])


def add_history(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based rolling windows per station (gaps in dates are respected)."""
    out = []
    for _, g in df.groupby("wmo", sort=False):
        g = g.set_index("report_date")
        r = g["precipitation"].fillna(0)
        feats = {}
        for w in RAIN_WINDOWS:
            feats[f"rain_sum_{w}d"] = r.rolling(f"{w}D").sum()
        for w in MEAN_WINDOWS:
            feats[f"temp_mean_{w}d"] = g["temperature"].rolling(f"{w}D", min_periods=1).mean()
            feats[f"rh_mean_{w}d"] = g["relative_humidity"].rolling(f"{w}D", min_periods=1).mean()
        feats["temp_max_7d"] = g["temperature"].rolling("7D", min_periods=1).max()
        feats["rh_min_7d"] = g["relative_humidity"].rolling("7D", min_periods=1).min()
        feats["wind_mean_3d"] = g["wind_speed"].rolling("3D", min_periods=1).mean()
        # days since >= 2 mm of rain (capped at 120)
        wet = r >= 2.0
        last_wet = pd.Series(g.index.where(wet.values), index=g.index).ffill()
        feats["days_since_rain"] = ((g.index.to_series() - last_wet).dt.days
                                    .fillna(120).clip(upper=120))
        # yesterday's weather (only if yesterday exists)
        prev = g[["temperature", "relative_humidity", "wind_speed", "precipitation"]].shift(1)
        gap = g.index.to_series().diff().dt.days.ne(1)
        prev[gap.values] = np.nan
        prev.columns = [f"{c}_lag1" for c in prev.columns]
        h = pd.concat([g, pd.DataFrame(feats), prev], axis=1).reset_index()
        out.append(h)
    df = pd.concat(out, ignore_index=True)
    df["month"] = df["report_date"].dt.month
    df["day_of_year"] = df["report_date"].dt.dayofyear
    df["year"] = df["report_date"].dt.year
    return df


WEATHER_ONLY_FEATURES = (
    WEATHER + [f"rain_sum_{w}d" for w in RAIN_WINDOWS]
    + [f"temp_mean_{w}d" for w in MEAN_WINDOWS] + [f"rh_mean_{w}d" for w in MEAN_WINDOWS]
    + ["temp_max_7d", "rh_min_7d", "wind_mean_3d", "days_since_rain",
       "temperature_lag1", "relative_humidity_lag1", "wind_speed_lag1", "precipitation_lag1",
       "month", "day_of_year", "latitude", "longitude", "elevation", "country"]
)


def main():
    df = add_station_info(load_weather())
    print(f"weather rows: {len(df):,}")
    df = add_history(df)
    df = df[df["fire_weather_index"].between(0, 250)]  # rows with a target
    df = df.dropna(subset=WEATHER)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / "weather_only.parquet", index=False)
    print(f"weather-only modelling rows: {len(df):,}, stations: {df.wmo.nunique()}")


if __name__ == "__main__":
    main()
