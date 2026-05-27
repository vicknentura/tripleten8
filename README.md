# Supervised & Unsupervised Learning — Class Imbalance and SMOTE

![Sprint](https://img.shields.io/badge/TripleTen-Sprint_9-FF9800?style=flat-square)
![Type](https://img.shields.io/badge/Type-Classification_%7C_SMOTE_%7C_Imbalanced_Learning-FF9800?style=flat-square)
![Stack](https://img.shields.io/badge/Stack-scikit--learn_%7C_imbalanced--learn_%7C_Pandas-FF9800?style=flat-square)
![Data](https://img.shields.io/badge/Dataset-Churn.csv_%7C_Credit_Card_Fraud-FF9800?style=flat-square)

---

## Overview

A focused study of **class imbalance** in supervised learning — one of the most common and consequential problems in applied ML. When one class is rare (fraud = 0.17%, disease = 1%), standard classifiers optimize toward the majority class and fail on the cases that matter most.

This sprint covers two datasets:
1. **Customer churn prediction** (`Churn.csv`) — telecom subscriber churn with moderate imbalance
2. **Credit card fraud detection** — severe imbalance (1:100 class ratio), addressed with SMOTE

---

## The Class Imbalance Problem

```
Naive classifier that predicts "no fraud" on all examples:
  → Accuracy: 99.83%  ← looks great
  → Recall:    0.0%   ← catches zero fraud cases
  → AUC-ROC:   0.5    ← equivalent to random guessing

Why accuracy is the wrong metric for imbalanced data.
```

---

## SMOTE — Synthetic Minority Oversampling Technique

Rather than duplicating existing minority samples (which adds no new information), SMOTE **synthesizes new samples** by interpolating between real minority-class examples in feature space:

```
Algorithm:
1. Select a random minority-class example x
2. Find k nearest neighbors of x in feature space (typically k=5)
3. Randomly choose one neighbor x_neighbor
4. Generate synthetic example:
       x_new = x + λ × (x_neighbor - x)
       where λ ~ Uniform(0, 1)

Result: new examples along line segments between real minority samples
```

```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=123)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Before SMOTE:  {0: 199020, 1: 345}     ← 577:1 ratio
# After SMOTE:   {0: 199020, 1: 199020}  ← balanced
```

---

## Full Modeling Pipeline

```
1. Data Loading
   └── Churn.csv / creditcard.csv → pandas

2. Feature Engineering
   ├── StandardScaler on 'Amount' column
   └── Drop 'Time' (non-informative)

3. Train/Test Split
   └── 70% train / 30% test (stratified)

4. SMOTE Oversampling
   └── Applied to TRAINING set only (never to test set)

5. Model Training
   └── Logistic Regression (C=4 regularization)

6. Evaluation
   ├── ROC-AUC score
   ├── Confusion matrix (seaborn heatmap)
   ├── Precision-Recall at multiple thresholds
   │       threshold in np.arange(0, 1, 0.02)
   └── ROC curve (FPR vs. TPR)
```

---

## Key Evaluation Metrics

| Metric | Formula | When to use |
|---|---|---|
| Accuracy | TP+TN / total | Only for balanced classes |
| Precision | TP / (TP+FP) | Minimize false alarms |
| Recall (Sensitivity) | TP / (TP+FN) | Minimize missed positives |
| ROC-AUC | Area under ROC curve | Overall discrimination ability |
| F1 Score | 2 × (P×R)/(P+R) | Balance precision and recall |

---

## Files

```
Supervised-Unsupervised-Learning/
├── Churn.csv                   # Telecom customer churn data (693KB)
└── SMOTE in Supervised Learning  # SMOTE theory + full credit card fraud pipeline
```

---

## Computational Biology Connection

Class imbalance is ubiquitous in biomedical data and the same techniques apply directly:

| Supervised/SMOTE concept | Biomedical equivalent |
|---|---|
| Fraud vs. non-fraud (1:577) | Pathogenic variant vs. benign (1:1000+) |
| SMOTE synthetic oversampling | Data augmentation in rare disease cohorts |
| Logistic regression + SMOTE | Variant pathogenicity classifiers (ClinVar, CADD) |
| Precision-recall threshold sweep | Clinical sensitivity / specificity tradeoff |
| ROC-AUC | Standard metric in diagnostic model validation |
| Confusion matrix | Clinical trial outcome categorization (TP/FP/TN/FN) |

In GWAS and genomic ML, rare variants and rare disease phenotypes create extreme class imbalance. SMOTE and cost-sensitive learning are active research areas in:
- **Rare variant association testing**
- **Cell type classification** in scRNA-seq (rare cell types)
- **Pathogenic variant prediction** (ClinVar positive:negative ratio ~1:50)
- **Clinical outcome prediction** in small-N rare disease trials
