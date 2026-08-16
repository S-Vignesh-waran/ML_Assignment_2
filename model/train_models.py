import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_PATH = "data/data.csv"
TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42
MODEL_DIR = "model"


def load_dataset():
    return pd.read_csv(DATA_PATH)


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

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, f"{MODEL_DIR}/logistic_regression.pkl")

    print("Logistic Regression Accuracy:", round(accuracy, 4))
    print("Logistic Regression model saved")

    return model


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = load_dataset()
    cleaned_dataset = clean_dataset(dataset)

    X_train, X_test, y_train, y_test = split_dataset(cleaned_dataset)

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    logistic_model = train_logistic_regression(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )