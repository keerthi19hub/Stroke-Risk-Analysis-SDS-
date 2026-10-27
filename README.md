# 🩺 Stroke Risk Analysis — Clinical Diagnostic & Epidemiological Prediction Pipeline

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=881337,BE123C,FB7185&height=220&section=header&text=Stroke%20Risk%20Analysis&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Clinical%20Diagnostic%20%26%20Epidemiological%20Prediction%20Pipeline&descSize=18&descAlignY=60&descAlign=50" width="100%" alt="Stroke Risk Analysis Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Statistical%20Data%20Science-881337?style=for-the-badge" alt="Data Science" />
  <img src="https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Tool-Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Technique-SMOTE%20Imbalance%20Balancing-BE123C?style=for-the-badge" alt="SMOTE" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Table of Contents

- [📖 Executive Summary](#-executive-summary)
- [🎯 Epidemiological Context & Objectives](#-epidemiological-context--objectives)
- [🔬 Machine Learning Pipeline & Methodology](#-machine-learning-pipeline--methodology)
- [✨ Key Features & Clinical Findings](#-key-features--clinical-findings)
- [📊 Model Performance & Comparative Benchmark](#-model-performance--comparative-benchmark)
- [🛠️ Tech Stack Matrix](#️-tech-stack-matrix)
- [⚙️ Reproducible Quickstart & Setup](#️-reproducible-quickstart--setup)
- [📁 Project Structure](#-project-structure)
- [👩‍💻 Author & Contributions](#-author--contributions)

---

## 📖 Executive Summary

The **Clinical Stroke Risk Analysis & Prediction Pipeline** is an end-to-end healthcare data science investigation evaluating physiological and demographic biomarkers associated with cerebrovascular accidents. Developed as part of advanced Statistical Data Science (SDS) coursework at **Alliance University**, the project addresses severe medical class imbalance (~95% non-stroke vs ~5% stroke) using advanced feature engineering, SMOTE balancing, and comparative benchmarking across Logistic Regression, Random Forest, and Gradient Boosting algorithms.

---

## 🎯 Epidemiological Context & Objectives

- **Clinical Problem:** Stroke remains the second leading cause of mortality worldwide. Early risk stratification can enable preventative cardiovascular intervention, yet clinical datasets present non-linear interactions, missing physiological metrics (e.g., BMI), and high class imbalance that mislead standard classification models.
- **Primary Objectives:**
  1. Conduct rigorous exploratory data analysis (EDA) to isolate primary clinical stroke risk factors.
  2. Implement clinically sound missing data imputation and categorical encoding.
  3. Correct minority class under-representation through SMOTE resampling.
  4. Compare classifiers using discrimination metrics prioritizing sensitivity / recall (minimizing false negatives).

---

## 🔬 Machine Learning Pipeline & Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                Clinical Stroke Risk Machine Learning Pipeline                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
               ┌──────────────────────────────────────────────┐
               │    Raw Clinical Dataset (5,110 patients)     │
               │ (Age, Hypertension, Heart Disease, Glucose)  │
               └───────────────────────┬──────────────────────┘
                                       │
                                       ▼
               ┌──────────────────────────────────────────────┐
               │         Data Cleansing & Preprocessing       │
               │  - Median imputation for missing BMI values  │
               │  - One-hot encoding of categorical features  │
               │  - StandardScaler continuous normalization   │
               └───────────────────────┬──────────────────────┘
                                       │
                                       ▼
               ┌──────────────────────────────────────────────┐
               │       Synthetic Resampling (SMOTE)           │
               │   Balanced training representation (50:50)   │
               └───────────────────────┬──────────────────────┘
                                       │
                                       ▼
               ┌──────────────────────────────────────────────┐
               │        Multi-Model Benchmark & Eval          │
               │  - Logistic Regression (Baseline)            │
               │  - Random Forest Ensemble                    │
               │  - Gradient Boosting Classifier              │
               │  Metrics: ROC-AUC, Recall, Precision, PR-AUC │
               └──────────────────────────────────────────────┘
```

---

## ✨ Key Features & Clinical Findings

- **🩸 High-Impact Correlates:** Age was isolated as the strongest single predictive risk factor, followed by chronic hypertension and average blood glucose levels.
- **⚖️ Imbalance Mitigation:** Overcame the severe 95:5 class skew using Synthetic Minority Over-sampling Technique (SMOTE).
- **📉 Diagnostic Visualizations:** Correlation heatmaps, glucose vs. BMI scatter plots, age-binned incidence histograms, and ROC curves.

---

## 📊 Model Performance & Comparative Benchmark

| Classifier | ROC-AUC | Precision (Stroke) | Recall / Sensitivity | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.84 | 0.22 | 0.80 | 0.35 |
| **Random Forest Classifier** | 0.88 | 0.31 | 0.78 | 0.44 |
| **Gradient Boosting Classifier** | **0.89** | **0.34** | **0.82** | **0.48** |

*Note: In medical diagnosis, high recall is prioritized to prevent undetected stroke risk cases.*

---

## 🛠️ Tech Stack Matrix

| Category | Tools & Libraries | Role |
| :--- | :--- | :--- |
| **Core Language** | Python 3.10+ | Scientific computing and data pipeline orchestration |
| **Machine Learning** | Scikit-Learn, Imbalanced-Learn | Estimators, cross-validation, SMOTE, ROC metric calculations |
| **Data Manipulation** | Pandas, NumPy | Dataframe reshaping, feature encoding, numerical arrays |
| **Data Visualization**| Matplotlib, Seaborn | Clinical distributions, pairplots, correlation matrices |
| **Environment** | Jupyter Notebook | Interactive laboratory execution and visual documentation |

---

## ⚙️ Reproducible Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/keerthi19hub/Stroke-Risk-Analysis-SDS-.git
cd Stroke-Risk-Analysis-SDS-
```

### 2. Setup Virtual Environment & Install Packages
```bash
python -m venv venv
venv\Scripts\activate  # On Linux/macOS: source venv/bin/activate

pip install --upgrade pip
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### 3. Launch Notebook
```bash
jupyter notebook "Healthcare_Stroke_Analysis(SDS)20 (1) (1).ipynb"
```

---

## 📁 Project Structure

```text
Stroke-Risk-Analysis-SDS-/
├── Healthcare_Stroke_Analysis(SDS)20 (1) (1).ipynb  # Primary laboratory notebook with full pipeline
└── README.md                                         # Production technical documentation
```

---

## 👩‍💻 Author & Contributions

**Keerthi N**  
- **GitHub:** [@keerthi19hub](https://github.com/keerthi19hub)  
- **Portfolio:** [keerthi19hub.github.io](https://keerthi19hub.github.io)  
- **Subject:** Statistical Data Science (SDS) · Alliance University, Bengaluru  
- **Role:** EDA formulation, SMOTE implementation, model hyperparameter tuning, and comparative diagnostic metric evaluation.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=881337,BE123C,FB7185&height=100&section=footer" width="100%" alt="Footer Banner" />
</p>
