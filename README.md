# ML Assignment 2 - Classification Models with Streamlit Deployment

## a. Problem Statement

The objective of this project is to build, evaluate, compare, and deploy multiple machine learning classification models using a public classification dataset.

This project uses the Breast Cancer Wisconsin Diagnostic Dataset to classify whether a tumor is malignant or benign based on numeric diagnostic measurements. The workflow includes dataset preparation, model training, evaluation using multiple classification metrics, comparison of model performance, and deployment through a Streamlit application.

The application allows users to upload test data, select a trained machine learning model, view predictions, and compare model performance using standard evaluation metrics.

---

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset  
**Dataset Source:** Kaggle  
**Kaggle Dataset Link:** <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>  
**Kaggle Dataset Slug:** `uciml/breast-cancer-wisconsin-data`
**Problem Type:** Binary Classification  
**Target Column:** `diagnosis`  
**Target Classes:** `M` / `malignant`, `B` / `benign`

**Number of Instances:** 569  
**Number of Features:** 30 numeric diagnostic features

The dataset contains:

- **569 instances**
- **30 numeric diagnostic features**
- **1 target column**

The features describe measurements computed from digitized images of breast mass cell nuclei, such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

This dataset satisfies the assignment requirement because it contains more than 500 instances and more than 12 features.

---

## c. GitHub Repository Link

GitHub Repository: <https://github.com/S-Vignesh-waran/ML_Assignment_2>

---

## d. Models Used

The following classification models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier - Ensemble Model

Run `python model/train_models.py` to generate the exact metrics in `model/metrics.json`. The Streamlit app also displays these metrics on `test_data.csv`.

The models were evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC, Matthews Correlation Coefficient

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9561 | 0.9954 | 0.9581 | 0.9561 | 0.9564 | 0.9085 |
| Decision Tree | 0.9123 | 0.9329 | 0.9137 | 0.9123 | 0.9127 | 0.8139 |
| kNN | 0.9737 | 0.9884 | 0.9747 | 0.9737 | 0.9735 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9298 | 0.9298 | 0.9298 | 0.8492 |
| Random Forest (Ensemble) | 0.9386 | 0.9940 | 0.9390 | 0.9386 | 0.9387 | 0.8689 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed very well on this dataset with an accuracy of 0.9561 and the highest AUC score of 0.9954. This shows that the model separates the benign and malignant classes very effectively. Since the dataset contains standardized numeric medical features, Logistic Regression is suitable because the classes are close to linearly separable. |
| Decision Tree | Decision Tree achieved the lowest performance among the evaluated models, with an accuracy of 0.9123 and MCC of 0.8139. Although it is easy to interpret, it may overfit or underperform compared with more stable models. Its AUC of 0.9329 is also the lowest among the models. |
| kNN | kNN achieved the best overall performance, with the highest accuracy of 0.9737, highest precision of 0.9747, highest F1 score of 0.9735, and highest MCC of 0.9442. This indicates that the feature space is well-suited for distance-based classification after scaling. |
| Naive Bayes | Naive Bayes performed reasonably well with an accuracy of 0.9298 and AUC of 0.9868. It is a fast and simple probabilistic model, but its performance is slightly lower because the model assumes feature independence, which may not fully hold for this medical dataset. |
| Random Forest (Ensemble) | Random Forest produced strong and stable results with an accuracy of 0.9386 and AUC of 0.9940. As an ensemble model, it reduces variance by combining multiple decision trees. However, in this test split, it did not outperform kNN or Logistic Regression. |
| Overall Winner | kNN is the overall winner for this dataset because it achieved the highest Accuracy, Precision, F1 Score, and MCC. Logistic Regression is also a strong model because it achieved the highest AUC, making it very effective for probability ranking and class separation. |

---

## Conclusion

Among the evaluated models, kNN performed the best overall on the chosen dataset. It achieved the strongest classification metrics across most categories, especially Accuracy, Precision, F1 Score, and MCC.

Logistic Regression was the second strongest model and achieved the highest AUC score, indicating excellent ability to separate malignant and benign cases. Random Forest also performed well as an ensemble model, while Naive Bayes provided a fast and reasonable baseline. Decision Tree was the most interpretable model but had the lowest overall performance.

Therefore, for this dataset, kNN is selected as the best-performing model based on the majority of evaluation metrics.


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