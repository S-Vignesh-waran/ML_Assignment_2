import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/data.csv"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42

MODEL_DIR = "model"
OUTPUT_DIR = "outputs"


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print("Original dataset shape:", df.shape)

    return df


def clean_dataset(df):
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "Unnamed: 32" in df.columns:
        df = df.drop(columns=["Unnamed: 32"])

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        {
            "M": 1,
            "B": 0
        }
    )

    print("Dataset cleaning completed")
    print("Cleaned dataset shape:", df.shape)

    return df


def split_dataset(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv("test_data.csv", index=False)

    print("Train-test split completed")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("test_data.csv exported successfully")

    return X_train, X_test, y_train, y_test


def calculate_auc(model, X_test, y_test):
    try:
        probabilities = model.predict_proba(X_test)

        return roc_auc_score(
            y_test,
            probabilities[:, 1]
        )

    except Exception:
        return None


def evaluate_model(model_name, model, X_test, y_test):
    predictions = model.predict(X_test)
    auc = calculate_auc(model, X_test, y_test)

    return {
        "ML Model Name": model_name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "AUC": round(auc, 4) if auc is not None else "NA",
        "Precision": round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        ),
        "Recall": round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        ),
        "F1": round(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        ),
        "MCC": round(matthews_corrcoef(y_test, predictions), 4)
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)

    decision_tree_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    decision_tree_model.fit(X_train, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, y_train)

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )
    random_forest_model.fit(X_train, y_train)

    joblib.dump(logistic_model, f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(decision_tree_model, f"{MODEL_DIR}/decision_tree.pkl")
    joblib.dump(knn_model, f"{MODEL_DIR}/knn.pkl")
    joblib.dump(naive_bayes_model, f"{MODEL_DIR}/naive_bayes.pkl")
    joblib.dump(random_forest_model, f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    metadata = {
        "target_column": TARGET_COLUMN,
        "feature_columns": X_train.columns.tolist(),
        "scaled_models": [
            "Logistic Regression",
            "kNN"
        ],
        "class_mapping": {
            "Benign": 0,
            "Malignant": 1
        }
    }

    joblib.dump(metadata, f"{MODEL_DIR}/metadata.pkl")

    results = [
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_test_scaled,
            y_test
        ),
        evaluate_model(
            "Decision Tree",
            decision_tree_model,
            X_test,
            y_test
        ),
        evaluate_model(
            "kNN",
            knn_model,
            X_test_scaled,
            y_test
        ),
        evaluate_model(
            "Naive Bayes",
            naive_bayes_model,
            X_test,
            y_test
        ),
        evaluate_model(
            "Random Forest (Ensemble)",
            random_forest_model,
            X_test,
            y_test
        )
    ]

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(
        f"{OUTPUT_DIR}/comparison_metrics.csv",
        index=False
    )

    print("\nModel comparison table:")
    print(comparison_df)

    print("\nAll trained models saved successfully.")
    print("comparison_metrics.csv generated successfully.")


if __name__ == "__main__":
    dataset = load_dataset()
    cleaned_dataset = clean_dataset(dataset)

    X_train, X_test, y_train, y_test = split_dataset(cleaned_dataset)

    train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )