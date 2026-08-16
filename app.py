import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


TARGET_COLUMN = "diagnosis"


MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl"
}


SCALED_MODELS = [
    "Logistic Regression",
    "kNN"
]


st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🧬",
    layout="wide"
)


@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])


@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")


@st.cache_data
def load_saved_metrics():
    return pd.read_csv("outputs/comparison_metrics.csv")


def calculate_auc(model, X_eval, y_eval):
    try:
        probabilities = model.predict_proba(X_eval)

        return roc_auc_score(
            y_eval,
            probabilities[:, 1]
        )

    except Exception:
        return None


def calculate_metrics(model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    auc = calculate_auc(model, X_eval, y_eval)

    metrics = {
        "Accuracy": accuracy_score(y_eval, predictions),
        "AUC": auc,
        "Precision": precision_score(
            y_eval,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_eval,
            predictions,
            zero_division=0
        ),
        "F1": f1_score(
            y_eval,
            predictions,
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_eval, predictions)
    }

    return metrics, predictions


st.title("Breast Cancer Classification Dashboard")
st.caption("Machine Learning Assignment 2")

st.markdown(
    """
    This Streamlit application evaluates multiple classification models using the
    Breast Cancer Wisconsin Diagnostic Dataset.

    Upload the provided `test_data.csv`, select a model, and view the model's
    evaluation metrics, confusion matrix, and classification report.
    """
)

with st.sidebar:
    st.header("Input Controls")

    selected_model_name = st.selectbox(
        "Select Classification Model",
        list(MODEL_PATHS.keys())
    )

    uploaded_file = st.file_uploader(
        "Upload test_data.csv",
        type=["csv"]
    )


st.subheader("Saved Model Comparison Table")

try:
    saved_metrics = load_saved_metrics()
    st.dataframe(saved_metrics, use_container_width=True)
except Exception as error:
    st.warning("Unable to load saved comparison metrics.")
    st.write(error)


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    if TARGET_COLUMN not in data.columns:
        st.error(
            f"The uploaded file must contain the target column '{TARGET_COLUMN}'."
        )
    else:
        X = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]

        model = load_model(selected_model_name)

        if selected_model_name in SCALED_MODELS:
            scaler = load_scaler()
            X_eval = scaler.transform(X)
        else:
            X_eval = X

        metrics, predictions = calculate_metrics(
            model,
            X_eval,
            y
        )

        st.subheader(f"Evaluation Metrics: {selected_model_name}")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")

        if metrics["AUC"] is None:
            col2.metric("AUC", "NA")
        else:
            col2.metric("AUC", f"{metrics['AUC']:.4f}")

        col3.metric("Precision", f"{metrics['Precision']:.4f}")
        col4.metric("Recall", f"{metrics['Recall']:.4f}")
        col5.metric("F1 Score", f"{metrics['F1']:.4f}")
        col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title(f"Confusion Matrix - {selected_model_name}")

        st.pyplot(fig)

        st.subheader("Classification Report")

        report = classification_report(
            y,
            predictions,
            target_names=["Benign", "Malignant"],
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

else:
    st.info("Please upload test_data.csv to evaluate the selected model.")