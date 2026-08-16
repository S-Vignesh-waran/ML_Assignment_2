# ML Assignment 2 - Classification Models with Streamlit Deployment

##  Problem Statement

The objective of this assignment is to build, evaluate, and deploy multiple machine learning classification models on one public classification dataset. The project demonstrates an end-to-end ML workflow:

1. Download a public dataset from Kaggle.
2. Train multiple classification models on the same dataset.
3. Evaluate every model using the required classification metrics.
4. Build an interactive Streamlit frontend.
5. Deploy the app on Streamlit Community Cloud.

The Streamlit app supports CSV upload, model selection, metrics display, confusion matrix, classification report, and downloadable predictions.

---

##  Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset  
**Dataset Source:** Kaggle - <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>  
**Kaggle Dataset Slug:** `uciml/breast-cancer-wisconsin-data`  
**Problem Type:** Binary Classification  
**Target Column:** `diagnosis`  
**Target Classes:** `M` / `malignant`, `B` / `benign`

**Number of Instances:** 569  
**Number of Features:** 30 numeric diagnostic features

This dataset satisfies the assignment constraints because it has more than 500 instances and more than 12 features.

---

##  GitHub Repository Link

GitHub Repository: <https://github.com/S-Vignesh-waran/ML_Assignment_2>

---

## Project Structure

```text
project-folder/
│-- app.py
│-- download_kaggle_dataset.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- data/
│   │-- data.csv                  # downloaded locally from Kaggle;
│-- model/
│   │-- __init__.py
│   │-- train_models.py
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- random_forest.joblib
│   │-- metadata.json
│   │-- metrics.json
│-- tests/
│   │-- smoke_test.py
```

---

## Installation

Use Python 3.11 for this project.

```zsh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Train Models and Generate Artifacts

```zsh
python download_kaggle_dataset.py
python model/train_models.py
```

The training script creates saved model files under `model/*.joblib`, `model/metadata.json`, `model/metrics.json`, and `test_data.csv`.

---

## Run Streamlit App Locally

```zsh
streamlit run app.py
```

---

##  Models Used and Evaluation Metrics

The project implements the five required models 
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier - Ensemble Model

Run `python model/train_models.py` to generate the exact metrics in `model/metrics.json`. The Streamlit app also displays these metrics on `test_data.csv`.

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9561 | 0.9954 | 0.9581 | 0.9561 | 0.9564 | 0.9085 |
| Decision Tree | 0.9123 | 0.9329 | 0.9137 | 0.9123 | 0.9127 | 0.8139 |
| kNN | 0.9737 | 0.9884 | 0.9747 | 0.9737 | 0.9735 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Random Forest (Ensemble) | 0.9386 | 0.9940 | 0.9390 | 0.9386 | 0.9387 | 0.8689 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Usually performs strongly on this dataset because the standardized numeric features are close to linearly separable. |
| Decision Tree | Easy to interpret but may overfit compared with ensemble and margin-based models. Depth and leaf-size limits are used to reduce overfitting. |
| kNN | Performs well when features are standardized, because distance-based classification is sensitive to scale. |
| Naive Bayes | Fast baseline model. It may perform slightly lower if feature independence assumptions are violated. |
| Random Forest (Ensemble) | Typically robust and high-performing because it combines many decision trees and reduces variance. |
| Overall Winner for this dataset? | kNN is the overall winner on the generated test split because it has the highest Accuracy, Precision, F1, and MCC. Logistic Regression has the highest AUC, so it is also a strong model for probability ranking. |

---

## Streamlit App Features

The deployed app includes the required features:

- Dataset upload option for CSV test data.
- Model selection dropdown.
- Evaluation metrics display: Accuracy, AUC, Precision, Recall, F1, and MCC.
- Model comparison table.
- Confusion matrix.
- Classification report.
- Prediction table and prediction CSV download.

---