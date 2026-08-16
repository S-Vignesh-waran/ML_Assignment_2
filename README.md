# ML_Assignment_2

##  Problem Statement

The objective of this assignment is to build an end-to-end machine learning classification solution.
Multiple classification models are implemented on the same dataset, evaluated using standard classification metrics, and demonstrated through an interactive Streamlit web application.

The application allows users to upload test data, select a trained machine learning model, and view model performance using metrics, confusion matrix, and classification report.

##  GitHub Repository Link

GitHub Repository Link:

https://github.com/S-Vignesh-waran/ML_Assignment_2.git

## Dataset Download

The Breast Cancer Wisconsin dataset is downloaded from Kaggle.

Kaggle dataset identifier:

```text
uciml/breast-cancer-wisconsin-data

##  Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Random Forest Classifier

##  Streamlit Application

Live Streamlit App Link:

PASTE_YOUR_STREAMLIT_APP_LINK_HERE

##  Project Structure

```text
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
│   │-- train_models.py
│   │-- logistic_regression
│   │-- decision_tree
│   │-- knn
│   │-- naive_bayes
│   │-- random_forest
```

##  Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE |
| Decision Tree | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE |
| kNN | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE |
| Naive Bayes | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE |
| Random Forest (Ensemble) | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE | UPDATE |

##  Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed well as a baseline model. It is simple, interpretable, and works well when the class boundaries are close to linear. |
| Decision Tree | Decision Tree provided interpretable classification rules. However, it may be more sensitive to overfitting compared to ensemble methods. |
| kNN | kNN performed well after feature scaling. Since it depends on distance calculations, scaling was important for improving its performance. |
| Naive Bayes | Naive Bayes was computationally efficient and fast. Its performance depends on the independence assumption between features. |
| Random Forest (Ensemble) | Random Forest provided strong and stable performance because it combines multiple decision trees and reduces overfitting. |
| Overall Winner for your dataset? | UPDATE_AFTER_RUNNING_CODE. Usually Random Forest is expected to perform strongly due to ensemble learning. |
