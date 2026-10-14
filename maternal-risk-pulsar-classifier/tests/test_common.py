"""Basic checks for the shared prediction logic. Run with:  pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_datasets, predict  # noqa: E402
from flask_app import app  # noqa: E402


def defaults(key):
    return {f["name"]: f["default"] for f in get_datasets()[key]["features"]}


def test_datasets_have_expected_shape():
    ds = get_datasets()
    assert set(ds) == {"maternal", "pulsar"}
    assert len(ds["maternal"]["features"]) == 6
    assert len(ds["pulsar"]["features"]) == 8
    assert all("grid" not in f for f in ds["pulsar"]["features"])


@pytest.mark.parametrize("key", ["maternal", "pulsar"])
def test_probabilities_sum_to_one(key):
    out = predict(key, defaults(key))
    assert abs(sum(p["p"] for p in out["probabilities"]) - 1) < 1e-3
    assert out["prediction"] in {p["label"] for p in out["probabilities"]}
    assert len(out["profile"]["you"]) == len(out["profile"]["axes"])


@pytest.mark.parametrize("key", ["maternal", "pulsar"])
def test_real_examples_are_classified_as_labelled(key):
    for ex in get_datasets()[key]["examples"]:
        assert predict(key, ex["values"])["prediction"] == ex["actual"]


def test_missing_and_bad_inputs_raise():
    vals = defaults("maternal")
    bad = dict(vals, Age="")
    with pytest.raises(ValueError):
        predict("maternal", bad)
    with pytest.raises(ValueError):
        predict("maternal", dict(vals, BS="abc"))
    with pytest.raises(ValueError):
        predict("maternal", dict(vals, SystolicBP=70, DiastolicBP=90))
    with pytest.raises(ValueError):
        predict("nope", vals)


def test_out_of_range_gives_warning_not_error():
    out = predict("maternal", dict(defaults("maternal"), Age=95))
    assert out["warnings"]


def test_pulsar_accepts_negative_values():
    out = predict("pulsar", dict(defaults("pulsar"), IP_Skewness=-0.5))
    assert "prediction" in out


def test_api_endpoints():
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/health").json["status"] == "ok"
    r = client.post("/api/predict/pulsar", json=defaults("pulsar"))
    assert r.status_code == 200 and r.json["dataset"] == "pulsar"
    assert client.post("/api/predict/maternal", json={"Age": 20}).status_code == 400
    assert client.post("/api/predict/unknown", json={}).status_code == 404
