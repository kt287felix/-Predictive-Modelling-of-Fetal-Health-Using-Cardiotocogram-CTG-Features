# 🩺 Predictive Modelling of Fetal Health Using Cardiotocogram (CTG) Features

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📄 Description

Fetal distress during pregnancy is one of the leading causes of infant mortality and complications during childbirth. Early and accurate detection of abnormal fetal conditions is critical for timely medical intervention.

This project builds a machine learning pipeline to **automatically classify fetal health** into three categories — **Normal (1)**, **Suspect (2)**, and **Pathological (3)** — using features extracted from **Cardiotocogram (CTG) exams**. CTG is a widely used clinical technique that records fetal heart rate (FHR) and uterine contractions, but its manual interpretation is subjective and error-prone.

To solve this, two supervised machine learning models are developed, compared, and evaluated:

- **Logistic Regression (LR)** — a statistical baseline classifier
- **Support Vector Machine (SVM)** — tested across multiple kernels and hypertuned using GridSearchCV

Before modelling, **Principal Component Analysis (PCA)** is applied to reduce the high-dimensional CTG feature space from 22 features down to 6 principal components, retaining over 81% of the variance. This improves model generalisation and removes multicollinearity.

The SVM model (Polynomial kernel) achieved **87.5% accuracy**, outperforming Logistic Regression (82.8%) across most evaluation metrics including Precision, Recall, F1-Score, and AUC-ROC.

> 📘 **For the full project report** including literature review, mathematical formulations, model assumptions, result tables, figures, and academic references — see the Word document included in this repository:
> **`PREDICTIVE_MODELLING_OF_FETAL_HEALTH_FINAL.docx`**

---

## 📂 Repository Structure

```
fetal-health-prediction/
│
├── fetal_health_prediction.py                       # Full ML pipeline (Python code)
├── PREDICTIVE_MODELLING_OF_FETAL_HEALTH_FINAL.docx  # Full academic project report
├── fetal_health.csv                                 # CTG dataset
├── requirements.txt                                 # Python dependencies
└── README.md
```

---

## 🧠 Methodology Summary

| Step | Details |
|---|---|
| Data Cleaning | Removed missing values & outliers (Z-score < 3) |
| Feature Selection | Dropped `fetal_movement`, `uterine_contractions`, `severe_decelerations` |
| Dimensionality Reduction | PCA → 6 components (81.6% variance explained) |
| Models | Logistic Regression, SVM (linear, rbf, sigmoid, poly kernels) |
| Hypertuning | GridSearchCV with 5-fold cross-validation |
| Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC |

---

## 📊 Results

| Model | Accuracy | F1-Score (Normal) | F1-Score (Suspect) | F1-Score (Pathological) |
|---|---|---|---|---|
| Logistic Regression | 82.8% | 92% | 28% | 18% |
| **SVM (Polynomial)** | **87.5%** | **93%** | **51%** | **20%** |

The SVM model consistently outperformed Logistic Regression, especially in detecting Suspect and Pathological cases — the clinically critical minority classes.

---

## 🚀 Getting Started

### Install dependencies
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

### Run the pipeline
```bash
python fetal_health_prediction.py
```

Ensure `fetal_health.csv` is in the same directory.

### Dataset source
[Kaggle – Reproductive & Child Healthcare CTG Dataset](https://www.kaggle.com/datasets/omegasaransh12/reproductivechildhealthcare)

---

## 🔮 Future Work

- Handle class imbalance with SMOTE or class-weighting
- Explore ensemble models (Random Forest, XGBoost, LightGBM)
- Apply deep learning for improved pattern recognition
- Use t-SNE for advanced dimensionality visualisation
- Clinical validation with medical professionals

---

## 👤 Author

**[Your Name]**
- 📧 [your.email@example.com]
- 💼 [LinkedIn Profile URL]
- 🐙 [GitHub Profile URL]

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
