import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fwi_equations import fwi_from_isi_bui  # noqa: E402

SAMPLE = {
    "temperature": 22.5, "dew_point_temperature": 8.0, "relative_humidity": 40,
    "wind_speed": 15, "atmospheric_pressure": 1012, "precipitation": 0, "rain_days": 3,
    "fine_fuel_moisture_code": 88, "duff_moisture_code": 35, "drought_code": 250,
    "build_up_index": 50, "initial_spread_index": 6, "country": "Canada",
}


def test_fwi_equation_known_values():
    assert fwi_from_isi_bui(0, 0) == pytest.approx(0)
    # FWI grows with both ISI and BUI
    assert fwi_from_isi_bui(10, 80) > fwi_from_isi_bui(5, 80) > fwi_from_isi_bui(5, 20)


@pytest.fixture(scope="module")
def client():
    if not (ROOT / "models" / "fwi_model.joblib").exists():
        pytest.skip("model not trained yet - run python run_pipeline.py")
    sys.path.insert(0, str(ROOT / "app"))
    from app import app
    return app.test_client()


def test_api_predict(client):
    r = client.post("/api/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.get_json()
    assert body["fwi"] >= 0
    assert abs(body["fwi"] - body["equation_fwi"]) < 2
    assert body["danger_class"] in {"Low", "Moderate", "High", "Very high", "Extreme"}


def test_api_rejects_bad_input(client):
    r = client.post("/api/predict", json={**SAMPLE, "relative_humidity": 150})
    assert r.status_code == 400


def test_form_page(client):
    assert client.get("/").status_code == 200
    r = client.post("/", data=SAMPLE)
    assert r.status_code == 200 and b"danger" in r.data
