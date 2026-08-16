"""Streamlit application for classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
DEFAULT_TEST_DATA = ROOT_DIR / "test_data.csv"

st.set_page_config(
    page_title="ML Assignment 2 - Classification Models",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load metadata, saved metrics, and trained models."""
    metadata_path = MODEL_DIR / "metadata.json"
    metrics_path = MODEL_DIR / "metrics.json"

    if not metadata_path.exists() or not metrics_path.exists():
        try:
            with st.spinner("Model artifacts are missing. Training models once from data/data.csv..."):
                from model.train_models import train_and_save

                train_and_save()
        except Exception as exc:
            st.error(
                "Model artifacts are missing and automatic training failed. "
                "Run `python download_kaggle_dataset.py` and `python model/train_models.py` first."
            )
            st.exception(exc)
            st.stop()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = {
        model_key: joblib.load(MODEL_DIR / f"{model_key}.joblib")
        for model_key in metadata["model_names"]
        if (MODEL_DIR / f"{model_key}.joblib").exists()
    }

    if not models:
        st.error("No trained model files were found in the model directory.")
        st.stop()

    return metadata, saved_metrics, models


def read_csv_dataframe(source: Any) -> pd.DataFrame:
    """Read CSV input and return a dataframe."""
    # noinspection PyArgumentList
    return cast(pd.DataFrame, pd.read_csv(source, iterator=False))


def normalize_target(y_raw: pd.Series, metadata: dict[str, Any]) -> pd.Series:
    """Convert string or numeric target values to encoded numeric labels."""
    label_to_index = metadata["label_to_index"]

    if pd.api.types.is_numeric_dtype(y_raw):
        return y_raw.astype(int)

    normalized_mapping = {label.lower(): index for label, index in label_to_index.items()}
    normalized_mapping.update({str(index): index for index in label_to_index.values()})
    normalized_mapping.update({label[0].lower(): index for label, index in label_to_index.items()})

    cleaned_target = y_raw.astype(str).str.strip().str.lower()
    mapped_target = cleaned_target.map(normalized_mapping)

    if mapped_target.isna().any():
        invalid_values = sorted(str(value) for value in y_raw[mapped_target.isna()].astype(str).unique())
        expected_values = ", ".join(label_to_index.keys())
        raise ValueError(
            f"Invalid target labels found: {', '.join(invalid_values)}. "
            f"Expected labels: {expected_values}."
        )

    return mapped_target.astype(int)


def get_score_values(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return scores suitable for binary ROC AUC calculation."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X)
        if np.ndim(decision_scores) == 1:
            return decision_scores

    return None


def evaluate_predictions(model: Any, X: pd.DataFrame, y_true: pd.Series) -> dict[str, Any]:
    """Calculate assignment metrics for one model."""
    y_pred = model.predict(X)
    y_scores = get_score_values(model, X)
    auc = float(roc_auc_score(y_true, y_scores)) if y_scores is not None and y_true.nunique() == 2 else None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "predictions": y_pred,
    }


def format_metric(value: Optional[float]) -> str:
    """Format metric values for display."""
    return "N/A" if value is None or pd.isna(value) else f"{value:.4f}"


def metrics_dataframe(metrics_by_model: dict[str, dict[str, Any]], model_names: dict[str, str]) -> pd.DataFrame:
    """Build a metrics comparison dataframe."""
    return pd.DataFrame(
        [
            {
                "ML Model Name": model_names.get(model_key, model_key),
                "Accuracy": metrics["accuracy"],
                "AUC": metrics["auc"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "MCC": metrics["mcc"],
            }
            for model_key, metrics in metrics_by_model.items()
        ]
    )


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, target_names: list[str]) -> plt.Figure:
    """Create a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    fig.tight_layout()
    return fig


def validate_input_data(data: pd.DataFrame, feature_names: list[str]) -> list[str]:
    """Return missing feature columns."""
    return [feature for feature in feature_names if feature not in data.columns]


def load_input_data(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    """Load uploaded CSV or bundled test data."""
    if uploaded_file is not None:
        return read_csv_dataframe(uploaded_file), "Uploaded CSV"
    return read_csv_dataframe(DEFAULT_TEST_DATA), "Bundled test_data.csv"


def display_dataset_preview(data: pd.DataFrame, data_source: str) -> None:
    """Display dataset preview."""
    st.subheader("Dataset Preview")
    st.write(f"**Source:** {data_source}")
    st.write(f"Rows: **{data.shape[0]}**, Columns: **{data.shape[1]}**")
    st.dataframe(data.head(20), use_container_width=True)


def display_metric_cards(metrics: dict[str, Any]) -> None:
    """Display the required evaluation metrics."""
    metric_columns = st.columns(6)
    metric_columns[0].metric("Accuracy", format_metric(metrics["accuracy"]))
    metric_columns[1].metric("AUC", format_metric(metrics["auc"]))
    metric_columns[2].metric("Precision", format_metric(metrics["precision"]))
    metric_columns[3].metric("Recall", format_metric(metrics["recall"]))
    metric_columns[4].metric("F1 Score", format_metric(metrics["f1"]))
    metric_columns[5].metric("MCC", format_metric(metrics["mcc"]))


def display_model_comparison(current_metrics: dict[str, dict[str, Any]], model_names: dict[str, str]) -> None:
    """Display model comparison table."""
    st.subheader("Model Comparison on Current Test Data")
    comparison_df = metrics_dataframe(current_metrics, model_names)
    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "AUC": lambda value: "N/A" if pd.isna(value) else f"{value:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ),
        use_container_width=True,
    )


def display_classification_outputs(y_true: pd.Series, y_pred: np.ndarray, target_names: list[str]) -> None:
    """Display confusion matrix and classification report."""
    matrix_column, report_column = st.columns([1, 1.2])

    with matrix_column:
        st.pyplot(plot_confusion_matrix(y_true, y_pred, target_names))

    with report_column:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


def display_predictions(data: pd.DataFrame, predicted_labels: list[str]) -> None:
    """Display predictions and provide a CSV download."""
    prediction_output = data.copy()
    prediction_output["predicted_diagnosis"] = predicted_labels

    st.subheader("Predictions")
    st.dataframe(prediction_output.head(100), use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        prediction_output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def display_dataset_details(metadata: dict[str, Any], model_names: dict[str, str]) -> None:
    """Display dataset and model metadata."""
    with st.expander("Dataset and model details"):
        st.json(
            {
                "dataset": metadata["dataset_name"],
                "dataset_source": metadata["dataset_source"],
                "problem_type": metadata["problem_type"],
                "instances": metadata["instances"],
                "features": metadata["features"],
                "target_column": metadata["target_column"],
                "target_names": metadata["target_names"],
                "models": model_names,
            }
        )


def main() -> None:
    metadata, saved_metrics, models = load_artifacts()
    feature_names = metadata["feature_names"]
    target_column = metadata["target_column"]
    target_names = metadata["target_names"]
    model_names = metadata["model_names"]

    st.title("📊 ML Assignment 2: Classification Model Demo")
    st.markdown(
        "Interactive Streamlit app for comparing classification models trained on the "
        "**Kaggle Breast Cancer Wisconsin Diagnostic Dataset**."
    )

    with st.sidebar:
        st.header("Dataset Input")
        uploaded_file = st.file_uploader(
            "Upload test CSV",
            type=["csv"],
            help="Upload the provided test_data.csv or another CSV with the same feature columns.",
        )
        st.caption("If no file is uploaded, the bundled `test_data.csv` is used.")
        selected_model_key = st.selectbox(
            "Select model",
            list(models.keys()),
            format_func=lambda key: model_names.get(key, key),
        )

    data, data_source = load_input_data(uploaded_file)
    display_dataset_preview(data, data_source)

    missing_features = validate_input_data(data, feature_names)
    if missing_features:
        st.error("The CSV is missing required feature columns: " + ", ".join(missing_features))
        st.stop()

    X = data[feature_names].copy()
    selected_model = models[selected_model_key]
    selected_display_name = model_names[selected_model_key]
    predictions = selected_model.predict(X)
    predicted_labels = [metadata["index_to_label"][str(int(prediction))] for prediction in predictions]

    st.subheader(f"Selected Model: {selected_display_name}")

    if target_column in data.columns:
        try:
            y_true = normalize_target(data[target_column], metadata)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        current_metrics = {
            model_key: evaluate_predictions(model, X, y_true)
            for model_key, model in models.items()
        }
        selected_metrics = current_metrics[selected_model_key]

        display_metric_cards(selected_metrics)
        display_model_comparison(current_metrics, model_names)
        display_classification_outputs(y_true, selected_metrics["predictions"], target_names)
    else:
        st.warning(
            f"Column `{target_column}` was not found, so evaluation metrics cannot be calculated. "
            "Predictions are still shown below."
        )
        st.subheader("Training-Time Metrics from Saved Test Split")
        st.dataframe(metrics_dataframe(saved_metrics, model_names), use_container_width=True)

    display_predictions(data, predicted_labels)
    display_dataset_details(metadata, model_names)


if __name__ == "__main__":
    main()

