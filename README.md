# Wine Quality Prediction

Predicting the quality of red wine using machine learning models trained on physicochemical properties.

---

## 📌 Overview

This project develops regression and classification models to predict red wine quality based on chemical characteristics. The regression task predicts wine quality scores , while the classification task categorizes wines as "Good" (7-8) or "Bad" (3-6).

- **Dataset**: [UCI Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- **Problem Types**:
  - Regression: Predict quality score 
  - Classification: Binary classification (Good vs Bad)
- **Metrics**:
  - Regression: RMSE
  - Classification: Accuracy, F1-score, Precision, Recall, AUC

---

# 🔬 Exploratory Data Analysis (EDA)

- **Key Findings**:
  - **Alcohol** has the strongest positive correlation with quality.
  - **Volatile acidity** negatively impacts quality.
  - High alcohol and low volatile acidity correlate with higher quality scores.
  - Most features have outliers; data is not normally distributed except for **density**.
  - Classes are imbalanced, with quality scores 3, 4, 7, and 8 being less frequent.
  - PCA shows no clear clustering of quality classes.

- **Visualizations**:
  - Correlation matrix: ![image](https://github.com/user-attachments/assets/e1f4032f-1432-47ae-9e51-3113659bef72)
  - Quality distribution: ![image](https://github.com/user-attachments/assets/51aa9981-392a-4c5f-ab54-ba0fcebbca40)![image](https://github.com/user-attachments/assets/12509ed9-48ab-4d4a-bf7e-44dfe46c26af)

---
## 🚀 Models & Results

### Regression Models

| Model                  | RMSE   |
|------------------------|--------|
| Linear Regression      | 0.6572 |
| Ridge Regression       | 0.6602 |
| Lasso Regression       | 0.6598 |
| **Random Forest**      | **0.6322** |

- **Best Model**: RandomForestRegressor
- **Key Features**: Alcohol, volatile acidity, sulphates (based on feature importance).

### Classification Models (Macro-Averaged Metrics)
> **Note:** All metrics except AUC are macro-averaged across classes.

| Model                          | Accuracy | F1-score (macro) | Precision (macro) | Recall (macro) | AUC  |
|--------------------------------|----------|----------|-----------|--------|------|
| Random Forest (Baseline)       | 0.88   | 0.70   | 0.76    | 0.67 | -    |
| Random Forest (Optimized)      | 0.88   | 0.75   | 0.73    | 0.76 | 0.887 |
| Logistic Regression (Baseline) | 0.88   | 0.68   | 0.73    | 0.65 | -    |
| Logistic Regression (Optimized)| 0.82 | 0.73 | 0.70 | 0.84 | 0.888 |

- **Best Model**: Optimized Random Forest and Optimized Logistic Regression (slightly better balance of metrics).
- **Key Features**: Alcohol, sulphates, volatile acidity (based on coefficients and feature importance).
- **Visualizations**:
  - ROC Curves: ![image](https://github.com/user-attachments/assets/8e544a69-b00f-45de-a42c-7e9007ac8ebd)
  - Confusion Matrices: ![image](https://github.com/user-attachments/assets/4ad00f34-9b00-4f97-8ba6-1793a879b5da)
  - Feature Importance: ![image](https://github.com/user-attachments/assets/54b8a525-3d8d-441d-8e03-c3f5fb29c472)

---

## ⚙️ Tech Stack

- **Python**: Core programming language
- **pandas**, **numpy**: Data manipulation
- **scikit-learn**: Machine learning models and metrics
- **feature-engine**: Outlier handling
- **imblearn**: Oversampling (SMOTE)
- **matplotlib**, **seaborn**: Data visualization
- **kagglehub**: Dataset downloading
- **Jupyter Notebook**: Development environment
