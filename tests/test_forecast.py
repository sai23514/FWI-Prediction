import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_openmeteo import fake_response  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "fwi_weather_only.joblib").exists(),
                                reason="weather-only model not trained")


def test_forecast_offline():
    from src.forecast import forecast
    res = forecast(53.5, -113.5, "CA", "Edmonton", data=fake_response())
    kinds = [d["kind"] for d in res["days"]]
    assert "today" in kinds and kinds.count("forecast") >= 2
    assert all(d["fwi"] >= 0 for d in res["days"])
    assert res["note"] is None


def test_dry_weather_is_more_dangerous_than_wet():
    from src.forecast import forecast
    dry = forecast(53.5, -113.5, "CA", data=fake_response(dry=True))["days"][-1]["fwi"]
    wet = forecast(53.5, -113.5, "CA", data=fake_response(dry=False))["days"][-1]["fwi"]
    assert dry > wet


def test_forecast_api(monkeypatch):
    import src.forecast as fc
    monkeypatch.setattr(fc, "fetch_hourly", lambda lat, lon, forecast_days=4: fake_response())
    monkeypatch.setattr(fc, "geocode", lambda name: {"latitude": 53.5, "longitude": -113.5,
                                                    "label": "Edmonton", "country_code": "CA"})
    sys.path.insert(0, str(ROOT / "app"))
    from app import app
    c = app.test_client()
    r = c.get("/api/forecast?place=Edmonton")
    assert r.status_code == 200 and len(r.get_json()["days"]) >= 8
    assert c.get("/forecast?place=Edmonton").status_code == 200
    assert c.get("/api/forecast").status_code == 400
