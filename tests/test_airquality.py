import numpy as np
import pandas as pd
import pytest

from airquality.analysis import association_rules, cpp_style_features
from airquality.classify import (
    CATEGORICAL,
    NUMERIC,
    _cross_validate,
    global_label,
    per_pollutant_label,
    run_experiments,
    station_id,
)
from airquality.data import clean_for_modelling, load_data


@pytest.fixture
def csv_file(tmp_path):
    rows = [
        "Country Code;City;Location;Coordinates;Pollutant;Source Name;Unit;Value;Last Updated;Country Label",
        "NA;;Windhoek;-22.5, 17.0;PM10;src;µg/m³;10.0;2024-01-01;Namibia",
        "ES;;Madrid;40.4, -3.7;NO2;src;µg/m³;-5.0;2024-01-01;Spain",  # negative value
        "ES;;Sevilla;;NO2;src;µg/m³;5.0;2024-01-01;Spain",  # no coordinates
        'ES;;"Calle ""X""";37.4, -6.0;O3;src;µg/m³;30.0;2024-01-01;Spain',  # quoted field
    ]
    path = tmp_path / "mini.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_load_data_filters_and_keeps_namibia(csv_file):
    df = load_data(csv_file)
    assert df["country"].tolist() == ["NA", "ES"]  # "NA" is not read as a missing value
    assert (df["value"] >= 0).all()
    assert df[["lat", "lon"]].notna().all().all()


def test_clean_for_modelling_drops_implausible_and_non_pollutants():
    df = pd.DataFrame(
        {
            "pollutant": ["CO", "CO", "NO2", "TEMPERATURE", "PM2.5"],
            "unit": ["µg/m³", "µg/m³", "ppm", "c", "ppm"],
            "value": [500.0, 3_198_676.0, 0.02, 25.0, 5.0],
        }
    )
    clean = clean_for_modelling(df)
    assert clean["value"].tolist() == [500.0, 0.02]


def test_value_is_not_a_feature():
    assert "value" not in NUMERIC + CATEGORICAL


@pytest.fixture
def synthetic():
    n = 400
    return pd.DataFrame(
        {
            "country": ["ES", "FR", "PT", "IT", "DE", "NL", "BE", "AT"] * (n // 8),
            "location": [f"loc{i}" for i in range(n)],
            "pollutant": ["NO2", "O3", "PM10", "CO"] * (n // 4),
            "unit": ["µg/m³"] * n,
            "lat": [40 + (i % 7) for i in range(n)],
            "lon": [-3 + (i % 5) for i in range(n)],
            "value": [float(i % 50) for i in range(n)],
        }
    )


def test_labels_are_roughly_35_percent_positive(synthetic):
    assert global_label(synthetic).mean() == pytest.approx(0.35, abs=0.05)
    assert per_pollutant_label(synthetic).mean() == pytest.approx(0.35, abs=0.05)


def test_leakage_is_detected_and_removed(synthetic):
    results = run_experiments(synthetic)
    leaky = results["cpp_replica_leaky"]["models"]["logreg_lbfgs"]["f1"]
    fixed = results["no_leakage_cpp_setup"]["models"]["logreg_lbfgs"]["f1"]
    assert "value" in results["cpp_replica_leaky"]["features"]
    assert "value" not in results["no_leakage_cpp_setup"]["features"]
    assert "value" not in results["no_leakage_clean"]["features"]
    assert leaky > 0.95 > fixed  # with the leak the model gets almost everything right; without it, not


def test_station_id_groups_by_coordinates():
    df = pd.DataFrame({"lat": [1.0, 1.0, 2.0], "lon": [5.0, 5.0, 5.0]})
    ids = station_id(df)
    assert ids[0] == ids[1] != ids[2]


@pytest.mark.parametrize("how", ["random", "by_station", "by_country"])
def test_cross_validation_returns_valid_metrics(synthetic, how):
    res = _cross_validate(synthetic, global_label(synthetic), seed=0, how=how)
    for metrics in res["models"].values():
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_association_rules_are_consistent(synthetic):
    rules = association_rules(cpp_style_features(synthetic), min_support=0.0)
    assert (rules["confidence"] >= rules["support"] - 1e-12).all()
    assert (rules["confidence"] <= 1 + 1e-12).all()
    # support of all level pairs of one column pair adds up to 1
    lat_lon = rules[rules["antecedent"].str.startswith("lat") & rules["consequent"].str.startswith("lon")]
    assert lat_lon["support"].sum() == pytest.approx(1.0)
    assert np.isfinite(rules["lift"]).all()


# --- Saved model and API ---

def test_model_roundtrip_and_predict(synthetic, tmp_path):
    from airquality import model as model_io

    bundle = model_io.train(synthetic)
    assert "value" not in bundle["features"]
    loaded = model_io.load(model_io.save(bundle, tmp_path / "m.joblib"))
    out = model_io.predict(loaded, [{"country": "ES", "pollutant": "NO2", "lat": 40.0, "lon": -3.0}])
    assert 0.0 <= out[0]["probability"] <= 1.0
    assert out[0]["high_pollution"] == (out[0]["probability"] > 0.5)


def test_api_health_and_predict(synthetic, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from airquality import model as model_io
    from airquality.api import app

    monkeypatch.setenv("MODEL_PATH", str(model_io.save(model_io.train(synthetic), tmp_path / "m.joblib")))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        ok = client.post("/predict", json={"country": "ES", "pollutant": "NO2", "lat": 40.0, "lon": -3.0})
        assert ok.status_code == 200 and set(ok.json()) == {"high_pollution", "probability"}
        assert client.post("/predict", json={"country": "ES", "pollutant": "NO2", "lat": 999, "lon": 0}).status_code == 422
