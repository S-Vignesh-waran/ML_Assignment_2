"""Smoke tests for the ML Assignment 2 project artifacts.

Run from the project root after training:
    python tests/smoke_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"


def main() -> None:
    metadata_path = MODEL_DIR / "metadata.json"
    metrics_path = MODEL_DIR / "metrics.json"
    test_data_path = ROOT_DIR / "test_data.csv"

    assert metadata_path.exists(), "metadata.json is missing"
    assert metrics_path.exists(), "metrics.json is missing"
    assert test_data_path.exists(), "test_data.csv is missing"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_data = pd.read_csv(test_data_path)

    feature_names = metadata["feature_names"]
    assert len(feature_names) >= 12, "dataset must have at least 12 features"
    assert metadata["instances"] >= 500, "dataset must have at least 500 instances"
    assert len(metrics) >= 5, "at least five models must be evaluated"
    assert all(feature in test_data.columns for feature in feature_names), "test_data.csv is missing features"
    assert metadata["target_column"] in test_data.columns, "test_data.csv is missing target column"

    X = test_data[feature_names]
    for model_key in metadata["model_names"]:
        model_path = MODEL_DIR / f"{model_key}.joblib"
        assert model_path.exists(), f"{model_path.name} is missing"
        model = joblib.load(model_path)
        predictions = model.predict(X.head(10))
        assert len(predictions) == min(10, len(X)), f"{model_key} prediction count mismatch"

    print("Smoke test passed: artifacts, test data, and model predictions are valid.")


if __name__ == "__main__":
    main()

