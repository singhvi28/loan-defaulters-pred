# 🏦 LendingClub Loan Defaulters Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter)](https://jupyter.org/)

An end-to-end risk analytics and machine learning solution for predicting peer-to-peer loan defaults using historical data from **LendingClub**. The project implements exploratory data analysis (EDA), domain-driven feature engineering, and a benchmark comparison across **Artificial Neural Networks (ANN)**, **XGBoost**, and **Random Forest** classifiers.

---

## 📑 Table of Contents
- [Business Understanding](#-business-understanding)
- [Dataset Overview](#-dataset-overview)
- [Project Architecture & Pipeline](#-project-architecture--pipeline)
- [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
- [Model Architecture & Training](#-model-architecture--training)
- [Model Evaluation & Benchmark](#-model-evaluation--benchmark)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Key Business Takeaways](#-key-business-takeaways)

---

## 📝 Business Understanding

LendingClub is a peer-to-peer lending platform that connects borrowers with individual or institutional investors. When an applicant applies for a loan, the lending institution faces two primary risks:

1. **Loss of Business (Opportunity Cost):** Rejecting an applicant who is creditworthy and likely to fully repay.
2. **Financial Loss (Default / Write-off):** Approving an applicant who defaults (`Charged Off`), leading to substantial capital loss.

**Objective:** Build a robust predictive classification engine that ranks applicant default risk to optimize the lending cutoff threshold and protect capital while preserving loan origination volume.

---

## 📊 Dataset Overview

The dataset (`lending_club_loan_two.csv`) consists of **396,030** historical loan records with **27** original features covering borrower financial health, loan characteristics, and repayment history.

- **Target Variable (`loan_status`):**
  - `Fully Paid` (Class `1`): **80.38%** (~318,357 loans)
  - `Charged Off` (Class `0`): **19.62%** (~77,673 loans)
- **Class Imbalance:** Roughly an **80 : 20** ratio of good vs. defaulting loans.
- **Data Split:** 67% Training set (264,796 samples) and 33% Test set (130,423 samples), resulting in **78** engineered numerical features.

---

## 🔄 Project Architecture & Pipeline

```
Raw Loan Data (396k records, 27 features)
   │
   ├──► 1. Exploratory Data Analysis (EDA)
   │        • Target distribution analysis
   │        • Correlation analysis (installment vs loan_amnt, dti, mort_acc)
   │        • Loan grade & subgrade risk segmentation
   │
   ├──► 2. Data Cleaning & Imputation
   │        • Drop high-cardinality & repetitive columns (emp_title, title, issue_d)
   │        • Correlated group imputation (mort_acc imputed via total_acc)
   │        • Outlier detection & clipping
   │
   ├──► 3. Feature Engineering & Encoding
   │        • Extract ZIP code from address
   │        • Parse earliest_cr_line to credit history origination year
   │        • One-hot encoding for sub_grade, home_ownership, verification_status, purpose
   │        • Normalization via MinMaxScaler
   │
   └──► 4. Model Benchmarking & Evaluation
            • Artificial Neural Network (ANN / Deep MLP with BatchNorm & Dropout)
            • XGBoost Classifier
            • Random Forest Classifier
```

---

## ⚙️ Data Preprocessing & Feature Engineering

1. **Preventing Data Leakage:**
   - The `issue_d` (month the loan was funded) was removed to avoid look-ahead bias, as this information is unavailable prior to loan origination.
2. **Handling High Cardinality & Redundancy:**
   - `emp_title` (thousands of individual titles) and `emp_length` (low correlation to outcome) were dropped.
   - `title` was dropped in favor of the standardized `purpose` category.
   - `grade` was dropped in favor of `sub_grade` to retain finer-grained credit risk signals without collinearity.
3. **Correlated Missing Value Imputation:**
   - `mort_acc` (mortgage accounts) had ~9.5% missing values. Rather than dropping or using a simple mean, it was imputed using the mean `mort_acc` grouped by `total_acc` ($r = 0.38$, the strongest financial correlation).
4. **Geographical & Temporal Feature Extraction:**
   - **Address:** Extracted the 5-digit US ZIP code to create localized risk dummy variables.
   - **Credit History:** Extracted the credit line creation year from `earliest_cr_line`.
5. **Encoding & Normalization:**
   - Nominal variables were one-hot encoded using `pd.get_dummies`.
   - Continuous features were scaled using `MinMaxScaler` for gradient-based convergence in the neural network.

---

## 🤖 Model Architecture & Training

Three model families were trained and evaluated on identical test splits:

### 1. Artificial Neural Network (Deep MLP)
- **Framework:** TensorFlow / Keras
- **Architecture:**
  - Input Layer: 78 features
  - Batch Normalization + Dropout (0.1)
  - Dense Layer 1: 150 units (`ReLU`) + Batch Normalization + Dropout (0.0)
  - Dense Layer 2: 150 units (`ReLU`) + Batch Normalization + Dropout (0.1)
  - Dense Layer 3: 150 units (`ReLU`) + Batch Normalization + Dropout (0.0)
  - Output Layer: 1 unit (`Sigmoid`)
- **Optimizer & Loss:** Adam (`lr=0.001`), Binary Cross-Entropy loss, monitored via `AUC`.
- **Batch Size & Epochs:** 32 batch size across 20 epochs.

### 2. XGBoost Classifier
- Extreme Gradient Boosting configured for binary classification (`binary:logistic`) to capture non-linear feature interactions and high-order decision splits.

### 3. Random Forest Classifier
- Ensemble of 100 bagging decision trees with bootstrapping to evaluate feature importance and baseline non-linear performance.

---

## 📈 Model Evaluation & Benchmark

### Performance Summary on Test Set (130,423 samples)

| Model | Test Accuracy | Precision (Default / 0) | Recall (Default / 0) | F1-Score (Default / 0) | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Artificial Neural Network (ANN)** | **88.87%** | 0.94 | 0.46 | 0.62 | **0.905** |
| **XGBoost Classifier** | **88.94%** | 0.91 | 0.48 | 0.63 | 0.734 |
| **Random Forest Classifier** | **88.94%** | 0.96 | 0.46 | 0.62 | 0.725 |

### Insights:
- **ROC-AUC Champion:** The **ANN** achieved the strongest overall ranking capability with a **0.905 ROC-AUC**, substantially outperforming traditional tree ensembles on this feature space.
- **High Precision on High-Risk Borrowers:** All models exhibited **91%–96% precision** when flagging a defaulter, ensuring that applicants flagged as high-risk have a very high probability of defaulting.
- **Threshold Adjustment Recommendation:** In real-world loan underwriting, the classification threshold should be shifted down from the default `0.5` toward `0.20–0.30` to increase default recall (currently ~46–48%), balancing the bank's risk tolerance against portfolio yield.

---

## 📁 Repository Structure

```
loan-defaulters-pred/
├── lending-club-loan-defaulters-prediction.ipynb  # Full Jupyter notebook with EDA, preprocessing & models
└── README.md                                     # Project documentation & benchmark overview
```

---

## 🚀 Getting Started

### Prerequisites
Make sure Python 3.8+ is installed on your machine.

### Installation
Clone or navigate into the repository:
```bash
git clone https://github.com/<your-username>/loan-defaulters-pred.git
cd loan-defaulters-pred
```

Install the required scientific, machine learning, and visualization dependencies:
```bash
pip install numpy pandas scipy scikit-learn xgboost tensorflow matplotlib seaborn hvplot jupyter
```

### Running the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook lending-club-loan-defaulters-prediction.ipynb
```

> **Note on Data Source:** The notebook expects the LendingClub dataset (`lending_club_loan_two.csv`), which is publicly available on [Kaggle's Lending Club Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Ensure the dataset path matches your local environment.

---

## 💡 Key Business Takeaways

1. **Credit Grade / Subgrade is Paramount:** Subgrades (`A1` through `G5`) show an almost monotonic relationship with default rates; loans in grade `G` have default rates exceeding 45%, compared to < 7% in grade `A`.
2. **Debt Burden Matters More Than Income Alone:** Debt-to-Income (`dti`) and installment amount relative to income are far more predictive of default than nominal `annual_inc`.
3. **Mortgage Accounts Proxy Stability:** Borrowers with open mortgage accounts (`mort_acc`) consistently show higher repayment rates, reflecting asset backing and higher credit vetting standards.

---

## 📜 Acknowledgments
- Dataset provided by **LendingClub** public records.
- Modeling pipeline inspired by standard financial credit scoring architectures and community research on Kaggle.
