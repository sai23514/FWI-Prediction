"""Synthetic Open-Meteo response so tests run offline."""
import numpy as np
import pandas as pd


def fake_response(days_past=92, days_fwd=4, dry=True):
    today = pd.Timestamp.now(tz="America/Edmonton").tz_localize(None).normalize()
    times = pd.date_range(today - pd.Timedelta(days=days_past),
                          today + pd.Timedelta(days=days_fwd) - pd.Timedelta(hours=1), freq="h")
    n = len(times)
    hour = times.hour.values
    rng = np.random.default_rng(0)
    temp = 18 + 8 * np.sin((hour - 9) / 24 * 2 * np.pi)
    rh = 60 - 25 * np.sin((hour - 9) / 24 * 2 * np.pi)
    rain = np.where(rng.random(n) < (0.005 if dry else 0.05), rng.uniform(1, 6, n), 0)
    return {"latitude": 53.5, "longitude": -113.5, "elevation": 668, "timezone": "America/Edmonton",
            "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                       "temperature_2m": temp.round(1).tolist(),
                       "dew_point_2m": (temp - 12).round(1).tolist(),
                       "relative_humidity_2m": rh.round().tolist(),
                       "wind_speed_10m": np.full(n, 18.0).tolist(),
                       "pressure_msl": np.full(n, 1013.0).tolist(),
                       "precipitation": rain.round(1).tolist()}}
