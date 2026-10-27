# 🏥 Clinical Stroke Risk Analysis & Predictive Modelling

> **Healthcare statistical data science and machine learning pipeline to identify key physiological risk factors and predict stroke occurrences.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

---

## Problem Statement

Early detection of stroke indicators is vital for clinical intervention. Medical datasets present severe class imbalance (~95% non-stroke) and non-linear feature interactions, making accurate prediction technically challenging.

## ML Pipeline

```
Raw Dataset
    │
    ▼
┌─────────────────┐
│  Data Cleaning  │  Missing value imputation (BMI median fill)
│  & EDA          │  Distribution analysis · Outlier detection
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Feature Eng.   │  Categorical encoding · Age binning
│                 │  Feature scaling (StandardScaler)
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Class Balance  │  SMOTE oversampling on minority class
└────────┬────────┘
         │
    ▼
┌─────────────────────────────────────┐
│  Model Training & Evaluation        │
│  ├── Logistic Regression            │
│  ├── Random Forest                  │
│  └── Gradient Boosting              │
│  Metrics: ROC-AUC · Precision-Recall│
└─────────────────────────────────────┘
```

## Key Risk Factors Identified

| Feature | Correlation with Stroke |
|---------|------------------------|
| Age | Strong positive correlation |
| Hypertension | Major risk factor |
| Heart Disease | Significant predictor |
| Average Glucose Level | Non-linear relationship |
| BMI | Moderate predictor |

## Model Performance

| Model | ROC-AUC | Precision | Recall |
|-------|---------|-----------|--------|
| Logistic Regression | ~0.84 | ~0.72 | ~0.80 |
| Random Forest | ~0.88 | ~0.77 | ~0.82 |
| Gradient Boosting | ~0.89 | ~0.79 | ~0.83 |

## Quickstart

```bash
git clone https://github.com/keerthi19hub/Stroke-Risk-Analysis-SDS-.git
cd Stroke-Risk-Analysis-SDS-

pip install -r requirements.txt
jupyter notebook Stroke_Risk_Analysis.ipynb
```

## Tech Stack

`Python` · `Scikit-Learn` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `Jupyter Notebook`

## Author

**Keerthi N** — Full data preprocessing pipeline, statistical hypothesis testing, model training and comparative metric evaluation.

*Alliance University, BCA — 2026*

---

Released under the [MIT License](https://opensource.org/licenses/MIT).
