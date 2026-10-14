"""Shared prediction logic for the Gradio and Flask apps.

Author: Ahmed Shahzad
Developer: Ahmed Shahzad

Reads the models and `models/model_info.json` produced by `03_model_training_pipeline.ipynb`,
so both web apps always match the trained models.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"

# Human-friendly text for each dataset (labels, units, wording)
META = {
    "maternal": {
        "title": "Maternal Risk",
        "tagline": "Six vital signs in, a pregnancy risk level out.",
        "source": "UCI Machine Learning Repository - Maternal Health Risk",
        "description": (
            "Enter a woman's vital signs and the model estimates whether her pregnancy risk "
            "level is low, mid or high."
        ),
        "target_label": "Predicted risk level",
        "positive_label": "high risk",
        "disclaimer": "Educational demo trained on a small dataset. It is not a medical device and must not be used for diagnosis.",
        "classes_display": {"low risk": "Low risk", "mid risk": "Mid risk", "high risk": "High risk"},
        "ref_classes": ["low risk", "high risk"],
        "labels": {
            "Age": {"label": "Age", "short": "Age", "unit": "years", "step": 1,
                    "help": "Age of the woman."},
            "SystolicBP": {"label": "Systolic blood pressure", "short": "Systolic", "unit": "mmHg", "step": 1,
                           "help": "The upper blood pressure value."},
            "DiastolicBP": {"label": "Diastolic blood pressure", "short": "Diastolic", "unit": "mmHg", "step": 1,
                            "help": "The lower blood pressure value."},
            "BS": {"label": "Blood glucose", "short": "Glucose", "unit": "mmol/L", "step": 0.1,
                   "help": "Blood sugar level."},
            "BodyTemp": {"label": "Body temperature", "short": "Temp", "unit": "\u00b0F", "step": 0.1,
                         "help": "Body temperature in degrees Fahrenheit."},
            "HeartRate": {"label": "Resting heart rate", "short": "Heart rate", "unit": "bpm", "step": 1,
                          "help": "Beats per minute."},
        },
        "positive_only": True,
    },
    "pulsar": {
        "title": "Pulsar Spotter",
        "tagline": "Eight signal statistics in, real pulsar or radio noise out.",
        "source": "UCI Machine Learning Repository - HTRU2 (High Time Resolution Universe survey)",
        "description": (
            "A pulsar is a spinning neutron star that sweeps a radio beam across the sky. "
            "Describe a telescope candidate with eight statistics and the model tells you whether "
            "it looks like a real pulsar or just noise."
        ),
        "target_label": "Candidate verdict",
        "positive_label": "pulsar",
        "disclaimer": "Trained on HTRU2 candidates only. Treat the output as a triage hint for an astronomer, not a discovery.",
        "classes_display": {"not pulsar": "Radio noise", "pulsar": "Pulsar"},
        "ref_classes": ["not pulsar", "pulsar"],
        "labels": {
            "IP_Mean": {"label": "Profile mean", "short": "Prof. mean", "unit": "", "step": 0.1,
                        "group": "Integrated pulse profile", "help": "Mean of the integrated pulse profile."},
            "IP_SD": {"label": "Profile spread", "short": "Prof. SD", "unit": "", "step": 0.1,
                      "group": "Integrated pulse profile", "help": "Standard deviation of the integrated profile."},
            "IP_Kurtosis": {"label": "Profile kurtosis", "short": "Prof. kurt.", "unit": "", "step": 0.01,
                            "group": "Integrated pulse profile", "help": "Excess kurtosis: how sharp and peaked the pulse is."},
            "IP_Skewness": {"label": "Profile skewness", "short": "Prof. skew", "unit": "", "step": 0.1,
                            "group": "Integrated pulse profile", "help": "Skewness: how lopsided the pulse is."},
            "DM_Mean": {"label": "DM-SNR mean", "short": "DM mean", "unit": "", "step": 0.1,
                        "group": "DM-SNR curve", "help": "Mean of the DM-SNR curve."},
            "DM_SD": {"label": "DM-SNR spread", "short": "DM SD", "unit": "", "step": 0.1,
                      "group": "DM-SNR curve", "help": "Standard deviation of the DM-SNR curve."},
            "DM_Kurtosis": {"label": "DM-SNR kurtosis", "short": "DM kurt.", "unit": "", "step": 0.01,
                            "group": "DM-SNR curve", "help": "Excess kurtosis of the DM-SNR curve."},
            "DM_Skewness": {"label": "DM-SNR skewness", "short": "DM skew", "unit": "", "step": 0.1,
                            "group": "DM-SNR curve", "help": "Skewness of the DM-SNR curve."},
        },
        "positive_only": False,
    },
}


@lru_cache(maxsize=1)
def _info() -> dict:
    return json.loads((MODEL_DIR / "model_info.json").read_text())


@lru_cache(maxsize=None)
def _model(key: str):
    return joblib.load(MODEL_DIR / _info()[key]["model_file"])


def _display(key: str, cls: str) -> str:
    return META[key]["classes_display"].get(cls, cls)


def warm_up() -> None:
    """Load both models once at start-up so the first prediction is instant."""
    for key in META:
        _model(key)


def get_datasets() -> dict:
    """Everything a UI needs: titles, labelled input fields, model scores and chart data."""
    out = {}
    for key, meta in META.items():
        info = _info()[key]
        labels = meta["labels"]
        features = []
        for f in info["features"]:
            extra = labels[f["name"]]
            slim = {k: v for k, v in f.items() if k != "grid"}  # the grid stays on the server
            features.append({**slim, **extra})
        out[key] = {
            "key": key,
            "title": meta["title"],
            "tagline": meta["tagline"],
            "source": meta["source"],
            "description": meta["description"],
            "target_label": meta["target_label"],
            "disclaimer": meta["disclaimer"],
            "task": info["task"],
            "classes": [{"label": c, "display": _display(key, c)} for c in info["classes"]],
            "positive_label": meta["positive_label"],
            "ref_classes": meta["ref_classes"],
            "threshold": info.get("threshold"),
            "features": features,
            "metrics": info["metrics"],
            "confusion_matrix": info["confusion_matrix"],
            "importance": [
                {**i, "label": labels[i["name"]]["label"]} for i in info["importance"]
            ],
            "examples": [
                {**e, "display": _display(key, e["name"])} for e in info["examples"]
            ],
        }
    return out


def _percentile(grid, value: float) -> float:
    """Percentile rank (0-100) of a value against a stored 101-point quantile grid."""
    g = np.asarray(grid)
    left = np.searchsorted(g, value, side="left")
    right = np.searchsorted(g, value, side="right")
    return float(round(100 * ((left + right) / 2) / len(g), 1))


def _validate(key: str, values: dict):
    """Turn raw input into a clean {feature: float} dict, or raise ValueError with a readable message."""
    info = _info()[key]
    labels = META[key]["labels"]
    row, warnings = {}, []

    for f in info["features"]:
        name, label = f["name"], labels[f["name"]]["label"]
        raw = values.get(name)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            raise ValueError(f"Please fill in: {label}")
        try:
            num = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number")
        if not math.isfinite(num):
            raise ValueError(f"{label} must be a finite number")
        if META[key]["positive_only"] and num <= 0:
            raise ValueError(f"{label} must be a positive number")
        if num < f["min"] or num > f["max"]:
            warnings.append(
                f"{label} = {num:g} is outside the range seen in training "
                f"({f['min']:g} to {f['max']:g}), so this prediction may be unreliable."
            )
        row[name] = num

    if key == "maternal" and row["SystolicBP"] <= row["DiastolicBP"]:
        raise ValueError("Systolic pressure must be higher than diastolic pressure")
    return row, warnings


def predict(key: str, values: dict) -> dict:
    """Validate the inputs and predict.

    Returns a dict with the verdict, class probabilities, per-feature drivers (what pushes
    the result) and a percentile profile for the radar chart.
    Raises ValueError with a readable message when an input is missing or invalid.
    """
    if key not in META:
        raise ValueError(f"Unknown dataset '{key}'")

    info = _info()[key]
    labels = META[key]["labels"]
    names = [f["name"] for f in info["features"]]
    row, warnings = _validate(key, values)
    model = _model(key)
    classes = info["classes"]

    # Row 0 is the real input. Rows 1..n each swap ONE feature for its training median,
    # which tells us how much that feature is pushing the result.
    rows = [row]
    for f in info["features"]:
        alt = dict(row)
        alt[f["name"]] = f["p50"]
        rows.append(alt)
    probs = model.predict_proba(pd.DataFrame(rows, columns=names))
    p0 = probs[0]

    if info["task"] == "binary":
        threshold = float(info["threshold"])
        focus_idx = 1
        verdict_idx = int(p0[1] >= threshold)
        confidence = float(p0[1] if verdict_idx == 1 else 1 - p0[1])
    else:
        threshold = None
        verdict_idx = int(np.argmax(p0))
        focus_idx = verdict_idx
        confidence = float(p0[verdict_idx])

    verdict = classes[verdict_idx]
    focus = classes[focus_idx]

    drivers = []
    for i, f in enumerate(info["features"]):
        delta = float(p0[focus_idx] - probs[i + 1][focus_idx])
        meta = labels[f["name"]]
        drivers.append({
            "name": f["name"], "label": meta["label"], "value": row[f["name"]],
            "unit": meta["unit"], "delta": round(delta, 4),
        })
    drivers.sort(key=lambda d: abs(d["delta"]), reverse=True)

    grids = {f["name"]: f["grid"] for f in info["features"]}
    profile = {
        "axes": [labels[n]["short"] for n in names],
        "you": [_percentile(grids[n], row[n]) for n in names],
        "refs": {
            c: info["class_profile"][c] for c in META[key]["ref_classes"]
        },
    }

    return {
        "prediction": verdict,
        "display": _display(key, verdict),
        "confidence": round(confidence, 4),
        "focus": focus,
        "focus_display": _display(key, focus),
        "threshold": threshold,
        "probabilities": [
            {"label": c, "display": _display(key, c), "p": round(float(p0[i]), 4)}
            for i, c in enumerate(classes)
        ],
        "drivers": drivers,
        "profile": profile,
        "warnings": warnings,
    }
