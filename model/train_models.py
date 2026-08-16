"""Train classification models for ML Assignment 2.

Dataset: Kaggle Breast Cancer Wisconsin Diagnostic dataset.
Kaggle slug: uciml/breast-cancer-wisconsin-data

The dataset satisfies the assignment constraints: 569 instances and 30 numeric
features for a binary classification problem.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
KAGGLE_DATASET = "uciml/breast-cancer-wisconsin-data"
DATASET_CSV_PATH = DATA_DIR / "data.csv"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "diagnosis"
TARGET_NAMES = ["malignant", "benign"]
LABEL_TO_INDEX = {"M": 0, "B": 1}

MODEL_SPECS: dict[str, tuple[str, Pipeline | DecisionTreeClassifier | RandomForestClassifier]] = {
    "logistic_regression": (
        "Logistic Regression",
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    ),
    "decision_tree": (
        "Decision Tree",
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
    "knn": (
        "kNN",
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
    ),
    "naive_bayes": (
        "Naive Bayes",
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
    ),
    "random_forest": (
        "Random Forest (Ensemble)",
        RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
}


def download_kaggle_dataset() -> None:
    """Download and unzip the Kaggle dataset if data/data.csv is missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(DATA_DIR),
        "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI is not installed. Run `pip install kaggle` and configure "
            "your Kaggle API token before training."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle dataset download failed. Configure credentials using either "
            "~/.kaggle/kaggle.json or KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        ) from exc


def ensure_dataset_exists() -> Path:
    """Return the Kaggle CSV path, downloading it when necessary."""
    if not DATASET_CSV_PATH.exists():
        print(f"{DATASET_CSV_PATH} not found. Downloading {KAGGLE_DATASET} from Kaggle...")
        download_kaggle_dataset()
    if not DATASET_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected Kaggle CSV not found at {DATASET_CSV_PATH}. "
            "Check the dataset download output."
        )
    return DATASET_CSV_PATH


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load the Kaggle CSV and return features, encoded target, and class names."""
    csv_path = ensure_dataset_exists()
    # noinspection PyArgumentList
    dataset = pd.read_csv(csv_path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target column `{TARGET_COLUMN}` was not found in {csv_path}")

    drop_columns = [column for column in ["id", "Unnamed: 32", TARGET_COLUMN] if column in dataset.columns]
    X = dataset.drop(columns=drop_columns).copy()
    X = X.apply(pd.to_numeric, errors="raise")
    y = dataset[TARGET_COLUMN].map(LABEL_TO_INDEX)
    if y.isna().any():
        invalid_labels = sorted(dataset.loc[y.isna(), TARGET_COLUMN].astype(str).unique())
        raise ValueError(f"Unexpected diagnosis labels found: {invalid_labels}")

    feature_names = list(X.columns)
    return X, y.astype(int), feature_names, TARGET_NAMES


def predict_scores(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return positive-class probabilities or decision scores for AUC."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X)
        if np.ndim(decision_scores) == 1:
            return decision_scores
    return None


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, target_names: list[str]) -> dict[str, Any]:
    """Compute all required classification metrics for one model."""
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)

    auc_score = None
    if y_score is not None:
        auc_score = float(roc_auc_score(y_test, y_score))

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": auc_score,
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_test_data(X_test: pd.DataFrame, y_test: pd.Series, target_names: list[str]) -> None:
    """Save held-out test data with human-readable labels for app upload."""
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.map(lambda idx: target_names[int(idx)]).values
    test_df.to_csv(TEST_DATA_PATH, index=False)


def save_metadata(feature_names: list[str], target_names: list[str], model_names: dict[str, str]) -> None:
    """Save metadata consumed by the Streamlit application."""
    metadata = {
        "dataset_name": "Breast Cancer Wisconsin Diagnostic Dataset",
        "dataset_source": "Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data",
        "kaggle_dataset_slug": KAGGLE_DATASET,
        "problem_type": "Binary classification",
        "instances": 569,
        "features": len(feature_names),
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "target_names": target_names,
        "label_to_index": {name: index for index, name in enumerate(target_names)},
        "index_to_label": {str(index): name for index, name in enumerate(target_names)},
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "model_names": model_names,
        "note": "Five classification models are included as listed in the assignment comparison table.",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_and_save() -> dict[str, Any]:
    """Train every model, save artifacts, and return metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y, feature_names, target_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    save_test_data(X_test, y_test, target_names)

    metrics: dict[str, Any] = {}
    model_names: dict[str, str] = {}
    for model_key, (display_name, model) in MODEL_SPECS.items():
        print(f"Training {display_name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{model_key}.joblib")
        model_names[model_key] = display_name
        metrics[model_key] = {
            "display_name": display_name,
            **evaluate_model(model, X_test, y_test, target_names),
        }

    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metadata(feature_names, target_names, model_names)
    print(f"Saved {len(MODEL_SPECS)} trained models to {MODEL_DIR}")
    print(f"Saved test data to {TEST_DATA_PATH}")
    return metrics


def main() -> None:
    metrics = train_and_save()
    summary = pd.DataFrame(
        [
            {
                "Model": value["display_name"],
                "Accuracy": value["accuracy"],
                "AUC": value["auc"],
                "Precision": value["precision"],
                "Recall": value["recall"],
                "F1": value["f1"],
                "MCC": value["mcc"],
            }
            for value in metrics.values()
        ]
    )
    print("\nEvaluation summary:")
    print(summary.to_string(index=False, float_format=lambda number: f"{number:.4f}"))


if __name__ == "__main__":
    main()


