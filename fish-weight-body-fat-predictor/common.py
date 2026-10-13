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
    "fish": {
        "title": "Fish Weight",
        "source": "Kaggle - Fish Market dataset",
        "description": "Predict the weight of a fish from its species and size measurements.",
        "target_label": "Predicted weight",
        "unit": "g",
        "labels": {
            "Species": "Species",
            "Length1": "Vertical length (cm)",
            "Length2": "Diagonal length (cm)",
            "Length3": "Cross length (cm)",
            "Height": "Height (cm)",
            "Width": "Diagonal width (cm)",
        },
    },
    "bodyfat": {
        "title": "Body Fat",
        "source": "Kaggle - Body Fat Prediction dataset",
        "description": "Predict a man's body fat percentage from body measurements.",
        "target_label": "Predicted body fat",
        "unit": "%",
        "labels": {
            "Age": "Age (years)",
            "Weight": "Weight (pounds)",
            "Height": "Height (inches)",
            "Neck": "Neck circumference (cm)",
            "Chest": "Chest circumference (cm)",
            "Abdomen": "Abdomen circumference (cm)",
            "Hip": "Hip circumference (cm)",
            "Thigh": "Thigh circumference (cm)",
            "Knee": "Knee circumference (cm)",
            "Ankle": "Ankle circumference (cm)",
            "Biceps": "Biceps circumference (cm)",
            "Forearm": "Forearm circumference (cm)",
            "Wrist": "Wrist circumference (cm)",
        },
    },
}


@lru_cache(maxsize=1)
def _info() -> dict:
    return json.loads((MODEL_DIR / "model_info.json").read_text())


@lru_cache(maxsize=None)
def _model(key: str):
    return joblib.load(MODEL_DIR / _info()[key]["model_file"])


def get_datasets() -> dict:
    """Everything a UI needs: titles, labelled input fields, and model scores."""
    out = {}
    for key, meta in META.items():
        info = _info()[key]
        features = []
        for f in info["features"]:
            features.append({**f, "label": meta["labels"][f["name"]]})
        out[key] = {**meta, "features": features, "metrics": info["metrics"]}
    return out


def predict(key: str, values: dict) -> dict:
    """Validate the inputs, predict, and return {"prediction": float, "warnings": [str, ...]}.

    Raises ValueError with a readable message when an input is missing or invalid.
    """
    if key not in META:
        raise ValueError(f"Unknown dataset '{key}'")

    info = _info()[key]
    labels = META[key]["labels"]
    row, warnings = {}, []

    for f in info["features"]:
        name, label = f["name"], labels[f["name"]]
        raw = values.get(name)
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            raise ValueError(f"Please fill in: {label}")

        if f["kind"] == "category":
            if raw not in f["choices"]:
                raise ValueError(f"{label} must be one of: {', '.join(f['choices'])}")
            row[name] = raw
        else:
            try:
                num = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{label} must be a number")
            if not math.isfinite(num) or num <= 0:
                raise ValueError(f"{label} must be a positive number")
            if num < f["min"] or num > f["max"]:
                warnings.append(
                    f"{label} = {num:g} is outside the range seen in training "
                    f"({f['min']:g} to {f['max']:g}), so this prediction may be unreliable."
                )
            row[name] = num

    frame = pd.DataFrame([row], columns=[f["name"] for f in info["features"]])
    pred = float(_model(key).predict(frame)[0])
    pred = float(np.clip(pred, 0, None))  # weight and body fat cannot be negative
    return {"prediction": round(pred, 1), "warnings": warnings}
