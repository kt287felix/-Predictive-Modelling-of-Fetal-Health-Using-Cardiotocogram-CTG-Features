# 🩺 Predictive Modelling of Fetal Health Using CTG Features

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning project that predicts fetal health outcomes — **Normal**, **Suspect**, or **Pathological** — from Cardiotocogram (CTG) exam features, using Logistic Regression and Support Vector Machine (SVM) models enhanced with Principal Component Analysis (PCA).

---

## 📌 Project Overview

Cardiotocography (CTG) is a technique used in obstetrics to monitor fetal heart rate and uterine contractions. Manual interpretation of CTG data is time-consuming and prone to human error. This project addresses that gap by building and comparing two supervised machine learning classifiers to automate fetal health classification.

**Key results:**
| Model | Accuracy | AUC (Class 1) | AUC (Class 2) | AUC (Class 3) |
|---|---|---|---|---|
| Logistic Regression | 82.8% | ~High | ~Moderate | ~Low |
| SVM (Polynomial kernel) | **87.5%** | **~High** | **~High** | **~High** |

The **SVM model** outperformed Logistic Regression across most evaluation metrics.

---

## 📂 Project Structure

```
fetal-health-prediction/
│
├── fetal_health_prediction.py   # Full pipeline: EDA → PCA → LR → SVM → Evaluation
├── fetal_health.csv             # Dataset (CTG features + fetal health labels)
├── README.md
└── requirements.txt
```

---

## 🧠 Methodology

### 1. Data Preprocessing
- Loaded CTG dataset (1,489 records, 22 features)
- Handled missing values using `dropna()`
- Detected and removed outliers using **Z-scores** (threshold < 3)
- Dropped low-value features: `fetal_movement`, `uterine_contractions`, `severe_decelerations`

### 2. Exploratory Data Analysis (EDA)
- Box plots to visualise feature distributions and outliers
- Correlation heatmap to assess multicollinearity between CTG features
- Count plot showing class imbalance (majority: Normal, minority: Suspect & Pathological)

### 3. Dimensionality Reduction — PCA
- Standardised all features (zero mean, unit variance)
- Selected **6 Principal Components** using the Variance Information method (explains ~81.6% of variance)
- Validated selection using Kaiser's Criterion and Scree Plot
- Visualised PCA clusters across the three fetal health classes

### 4. Models
- **Logistic Regression** — fitted on PCA components, evaluated with confusion matrix, classification report, and ROC-AUC
- **SVM** — tested across four kernels (`linear`, `rbf`, `sigmoid`, `poly`); Polynomial kernel performed best
- **Hyperparameter Tuning** — `GridSearchCV` with 5-fold cross-validation over `C`, `kernel`, and `degree`

### 5. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Multi-class ROC-AUC Curve (one-vs-rest)

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `baseline_value` | Baseline fetal heart rate (bpm) |
| `accelerations` | Number of accelerations per second |
| `abnormal_short_term_variability` | % time with abnormal short-term variability |
| `histogram_width`, `histogram_mean`, etc. | Histogram features of FHR |
| `fetal_health` | **Target**: 1 = Normal, 2 = Suspect, 3 = Pathological |

**Dataset source:** [Kaggle – Reproductive & Child Healthcare CTG Dataset](https://www.kaggle.com/datasets/omegasaransh12/reproductivechildhealthcare)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
scikitplot
```

### Run
```bash
python fetal_health_prediction.py
```

Make sure `fetal_health.csv` is in the same directory as the script.

---

## 📈 Results Summary

- Both models struggled with minority classes (Suspect & Pathological) due to class imbalance.
- SVM with a Polynomial kernel achieved the best overall performance (87.5% accuracy).
- PCA successfully reduced dimensionality from 22 features to 6 components while retaining 81.6% of the variance.
- High AUC-ROC values confirmed both models' ability to discriminate between fetal health classes.

---

## 🔮 Future Work

- Address class imbalance using SMOTE or class-weighting strategies
- Explore ensemble methods (Random Forest, XGBoost, LightGBM)
- Apply deep learning approaches for improved pattern recognition
- Integrate t-SNE for better dimensionality visualisation
- Collaborate with medical professionals to validate predictions in a clinical setting

---

## 👤 Author

**[Your Name]**
- 📧 [your.email@example.com]
- 💼 [LinkedIn Profile URL]
- 🐙 [GitHub Profile URL]

*This project was completed as an academic final-year project in Statistics / Data Science.*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
