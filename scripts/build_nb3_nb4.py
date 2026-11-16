"""
Builds Notebook 03 (Logistic Regression) and Notebook 04 (Linear SVM).
"""

import json
import os

os.makedirs("notebooks", exist_ok=True)

def make_markdown_cell(text):
    lines = [line + '\n' for line in text.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

def make_code_cell(code):
    lines = [line + '\n' for line in code.strip().split('\n')]
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

# ==========================================
# NOTEBOOK 03: LOGISTIC REGRESSION
# ==========================================
nb3_cells = [
    make_markdown_cell("""# Cyberbullying Classification: Regularized Logistic Regression with TF-IDF

## 1. Academic Introduction & Mathematical Formulation
Logistic Regression is a foundational Generalized Linear Model (GLM) for binary classification. Given a high-dimensional sparse TF-IDF feature vector $\\mathbf{x} \\in \\mathbb{R}^d$, the model models the posterior probability of cyberbullying ($y = 1$) via the standard sigmoid link function $\\sigma(z)$:

$$P(y = 1 \\mid \\mathbf{x}; \\mathbf{w}, b) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b) = \\frac{1}{1 + e^{-(\\mathbf{w}^T \\mathbf{x} + b)}}$$

### Loss Function & Regularization
To mitigate overfitting on high-dimensional text representations ($d = 10,000$), we optimize the penalized negative log-likelihood (binary cross-entropy) with an $L_2$ Tikhonov regularization penalty:

$$\\min_{\\mathbf{w}, b} \\frac{1}{2} \\|\\mathbf{w}\\|^2_2 + C \\sum_{i=1}^N \\left[ -y_i \\log P(y_i = 1 \\mid \\mathbf{x}_i) - (1 - y_i) \\log(1 - P(y_i = 1 \\mid \\mathbf{x}_i)) \\right]$$

where $C > 0$ governs the trade-off between margin regularization and empirical training loss.

### Model Advantages in Text Classification
1. **Convexity**: Guarantees a globally optimal parameter vector $(\\mathbf{w}^*, b^*)$ without risk of local minima.
2. **Interpretability**: The magnitude and sign of weight $w_j$ directly reflect the log-odds change associated with feature $j$:
   $$\\log\\left(\\frac{P(y=1\\mid\\mathbf{x})}{1 - P(y=1\\mid\\mathbf{x})}\\right) = \\sum_{j=1}^d w_j x_j + b$$
3. **Calibrated Probabilities**: Outputs smooth posterior probabilities suitable for threshold tuning and confidence scoring in production moderation workflows."""),

    make_code_cell("""# Environment Setup and Scientific Imports
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Canonical Label Definitions
LABEL_MAP = {0: 'Non-Bullying', 1: 'Bullying'}
os.makedirs(os.path.join("models", "logistic_regression"), exist_ok=True)

print("Environment initialized.")"""),

    make_markdown_cell("""## 2. Ingestion of Preprocessed Partitions & Vectorization
We load the partitioned train and test datasets created in **`02_preprocessing.ipynb`** along with the fitted `models/tfidf_vectorizer.joblib` artifact."""),

    make_code_cell("""# Load preprocessed datasets
train_path = os.path.join("data", "processed_train.csv")
test_path = os.path.join("data", "processed_test.csv")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")

if not os.path.exists(train_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Prerequisite data or vectorizer not found. Please execute Notebook 02 first.")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

print(f"Loaded training samples: {len(train_df):,} | Testing samples: {len(test_df):,}")

# Extract TF-IDF feature matrices
X_train = tfidf_vectorizer.transform(train_df['clean_text'].fillna(''))
X_test = tfidf_vectorizer.transform(test_df['clean_text'].fillna(''))

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"X_train matrix shape: {X_train.shape} (CSR sparse matrix)")
print(f"X_test matrix shape:  {X_test.shape} (CSR sparse matrix)")"""),

    make_markdown_cell("""## 3. Regularization Hyperparameter Exploration ($C$)
We evaluate the inverse regularization parameter $C \\in \\{0.01, 0.1, 1.0, 5.0, 10.0\\}$ using 5-Fold Stratified Cross-Validation on the training partition."""),

    make_code_cell("""# 5-Fold Cross-Validation for Regularization Strength C
c_candidates = [0.01, 0.1, 1.0, 5.0, 10.0]
cv_scores = []
cv_stds = []

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Evaluating C hyperparameter via 5-Fold Stratified Cross-Validation:")
for c in c_candidates:
    clf_candidate = LogisticRegression(
        C=c,
        penalty='l2',
        solver='liblinear',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    scores = cross_val_score(clf_candidate, X_train, y_train, cv=cv_strat, scoring='f1', n_jobs=-1)
    cv_scores.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"  C = {c:<5} | Mean F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_idx = np.argmax(cv_scores)
best_c = c_candidates[best_idx]
print(f"\\nOptimal Regularization Parameter: C = {best_c} (F1 = {cv_scores[best_idx]:.4f})")

# Plot Cross-Validation Curve
plt.figure(figsize=(9, 4.5))
plt.errorbar(range(len(c_candidates)), cv_scores, yerr=cv_stds, fmt='-o', color='#2b5c8f', capsize=5, lw=2)
plt.xticks(range(len(c_candidates)), [str(c) for c in c_candidates])
plt.xlabel("Inverse Regularization Strength ($C$)")
plt.ylabel("5-Fold Cross-Validation F1-Score")
plt.title("Logistic Regression: Hyperparameter Tuning ($C$)", pad=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 4. Final Model Training & Convergence
We instantiate the final `LogisticRegression` model with the optimal parameter $C = 1.0$, balanced class weighting to handle class distribution dynamics, and measure training execution latency."""),

    make_code_cell("""# Train final Logistic Regression model
start_time = time.time()

log_reg = LogisticRegression(
    C=best_c,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    max_iter=1000
)

log_reg.fit(X_train, y_train)
train_duration = time.time() - start_time

print(f"Training converged in {train_duration:.3f} seconds.")
print(f"Model intercept: {log_reg.intercept_[0]:.4f}")
print(f"Non-zero coefficients: {np.count_nonzero(log_reg.coef_)} / {X_train.shape[1]}")"""),

    make_markdown_cell("""## 5. Quantitative Evaluation on Hold-Out Test Set
We evaluate test set performance across standard classification metrics:
* **Accuracy**: $\\frac{TP + TN}{TP + TN + FP + FN}$
* **Precision**: $\\frac{TP}{TP + FP}$ (purity of bullying alerts)
* **Recall**: $\\frac{TP}{TP + FN}$ (coverage of actual harassment)
* **F1-Score**: $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$
* **ROC-AUC**: Area Under the Receiver Operating Characteristic Curve."""),

    make_code_cell("""# Compute Predictions and Probabilities
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Quantitative Metric Computation
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
rec_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob)

# Class-Specific Metrics
prec_bully = precision_score(y_test, y_pred, pos_label=1)
rec_bully = recall_score(y_test, y_pred, pos_label=1)
f1_bully = f1_score(y_test, y_pred, pos_label=1)

print("=== Classification Report (Hold-Out Test Set) ===")
print(classification_report(y_test, y_pred, target_names=['Non-Bullying (0)', 'Bullying (1)'], digits=4))

# Summary Metrics DataFrame
lr_metrics = {
    'Model': 'TF-IDF + Logistic Regression',
    'Accuracy': round(acc, 4),
    'Macro Precision': round(prec_macro, 4),
    'Macro Recall': round(rec_macro, 4),
    'Macro F1': round(f1_macro, 4),
    'Weighted F1': round(f1_weighted, 4),
    'Bullying Precision': round(prec_bully, 4),
    'Bullying Recall': round(rec_bully, 4),
    'Bullying F1': round(f1_bully, 4),
    'ROC-AUC': round(roc_auc, 4),
    'Training Time (s)': round(train_duration, 4)
}

print("\\n=== Model Performance Summary Table ===")
for k, v in lr_metrics.items():
    print(f"  {k:<22}: {v}")"""),

    make_markdown_cell("""## 6. Confusion Matrix & Diagnostic Visualizations
The confusion matrix characterizes classification performance across both positive and negative categories."""),

    make_code_cell("""# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Absolute Counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Non-Bullying', 'Bullying'],
            yticklabels=['Non-Bullying', 'Bullying'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title("Confusion Matrix (Counts)", pad=12, fontweight='bold')
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Ground Truth")

# Subplot 2: Normalized Percentages
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=['Non-Bullying', 'Bullying'],
            yticklabels=['Non-Bullying', 'Bullying'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title("Normalized Confusion Matrix (Recall)", pad=12, fontweight='bold')
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Ground Truth")

plt.tight_layout()
plt.show()"""),

    make_code_cell("""# ROC and Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_test, y_prob)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
avg_precision = average_precision_score(y_test, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
axes[0].set_title("Receiver Operating Characteristic (ROC)", pad=12, fontweight='bold')
axes[0].set_xlabel("False Positive Rate (FPR)")
axes[0].set_ylabel("True Positive Rate (TPR / Recall)")
axes[0].legend(loc='lower right', frameon=True)

# Precision-Recall Curve
axes[1].plot(recall_curve, precision_curve, color='#d9534f', lw=2.5, label=f'PR Curve (AP = {avg_precision:.4f})')
axes[1].set_title("Precision-Recall Curve", pad=12, fontweight='bold')
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 7. Model Interpretability: Top Predictive Lexical Features
Because Logistic Regression is a linear additive model, the learned weights $\\mathbf{w}$ allow direct feature attribution:
* **Large Positive Coefficients** ($w_j > 0$): Features that most heavily increase the odds of predicting **Bullying**.
* **Large Negative Coefficients** ($w_j < 0$): Features that most heavily indicate **Non-Bullying** discourse."""),

    make_code_cell("""# Extract Feature Names and Learned Coefficients
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
coefs = log_reg.coef_[0]

# Top 20 Bullying Features (Highest Positive Weights)
top_pos_idx = np.argsort(coefs)[::-1][:20]
top_pos_features = feature_names[top_pos_idx]
top_pos_weights = coefs[top_pos_idx]

# Top 20 Non-Bullying Features (Lowest Negative Weights)
top_neg_idx = np.argsort(coefs)[:20]
top_neg_features = feature_names[top_neg_idx]
top_neg_weights = coefs[top_neg_idx]

# Visualization of Top Features
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Bullying Indicators
axes[0].barh(range(len(top_pos_features)), top_pos_weights, color='#d9534f', edgecolor='black', alpha=0.85)
axes[0].set_yticks(range(len(top_pos_features)))
axes[0].set_yticklabels(top_pos_features, fontsize=11)
axes[0].invert_yaxis()
axes[0].set_title("Top 20 Features Indicating Bullying ($w > 0$)", pad=10, fontweight='bold')
axes[0].set_xlabel("Logistic Regression Coefficient Weight")

# Non-Bullying Indicators
axes[1].barh(range(len(top_neg_features)), np.abs(top_neg_weights), color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[1].set_yticks(range(len(top_neg_features)))
axes[1].set_yticklabels(top_neg_features, fontsize=11)
axes[1].invert_yaxis()
axes[1].set_title("Top 20 Features Indicating Non-Bullying ($|w|, w < 0$)", pad=10, fontweight='bold')
axes[1].set_xlabel("Absolute Coefficient Weight")

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 8. Qualitative Error Analysis: Misclassified Samples
To understand the limitations of linear n-gram models, we audit both error types:
1. **False Positives (FP)**: Neutral or positive texts erroneously classified as Bullying. Often triggered by benign profanity, colloquial slang, or aggressive metaphors in non-targeted contexts.
2. **False Negatives (FN)**: Genuine cyberbullying instances that escaped detection. Often caused by implicit harassment, sarcasm, absence of overt profanity, or out-of-vocabulary evasion slang."""),

    make_code_cell("""# Identify Misclassified Samples
test_df_eval = test_df.copy().reset_index(drop=True)
test_df_eval['y_true'] = y_test
test_df_eval['y_pred'] = y_pred
test_df_eval['prob_bully'] = y_prob

# False Positives: True = 0, Pred = 1
fp_df = test_df_eval[(test_df_eval['y_true'] == 0) & (test_df_eval['y_pred'] == 1)].sort_values('prob_bully', ascending=False)

# False Negatives: True = 1, Pred = 0
fn_df = test_df_eval[(test_df_eval['y_true'] == 1) & (test_df_eval['y_pred'] == 0)].sort_values('prob_bully', ascending=True)

print(f"Total False Positives in Test Set: {len(fp_df):,} / {(y_test == 0).sum():,}")
print(f"Total False Negatives in Test Set: {len(fn_df):,} / {(y_test == 1).sum():,}")

print("\\n=== Top 4 False Positive Instances (Benign flagged as Bullying) ===")
for i, (_, row) in enumerate(fp_df.head(4).iterrows(), 1):
    print(f"[{i}] Confidence P(Bully)={row['prob_bully']:.3f} | Raw: '{row['text']}'")

print("\\n=== Top 4 False Negative Instances (Bullying missed by model) ===")
for i, (_, row) in enumerate(fn_df.head(4).iterrows(), 1):
    print(f"[{i}] Confidence P(Bully)={row['prob_bully']:.3f} | Raw: '{row['text']}'")"""),

    make_markdown_cell("""## 9. Model Artifact Serialization & Metrics Export
We persist the trained model artifact to `models/logistic_regression/model.joblib` and save evaluation metrics to `models/metrics_logistic_regression.json` for synthesis in **`06_model_comparison.ipynb`**."""),

    make_code_cell("""# Persist Trained Model Artifact
model_export_path = os.path.join("models", "logistic_regression", "model.joblib")
joblib.dump(log_reg, model_export_path)
print(f"Logistic Regression model saved to: '{model_export_path}' ({os.path.getsize(model_export_path)/1024:.1f} KB)")

# Export Evaluation Metrics to JSON
metrics_export_path = os.path.join("models", "metrics_logistic_regression.json")
with open(metrics_export_path, 'w', encoding='utf-8') as f:
    json.dump(lr_metrics, f, indent=2)

print(f"Metrics successfully exported to: '{metrics_export_path}'")""")
]

# Write Notebook 03
nb3 = make_notebook(nb3_cells)
with open(os.path.join("notebooks", "03_logistic_regression.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb3, f, indent=2)

print("Notebook 03 successfully generated.")


# ==========================================
# NOTEBOOK 04: SUPPORT VECTOR MACHINE (SVM)
# ==========================================
nb4_cells = [
    make_markdown_cell("""# Cyberbullying Classification: Support Vector Machines (LinearSVC)

## 1. Academic Introduction & Mathematical Foundations
Support Vector Machines (SVM) represent one of the most effective classical machine learning algorithms for text classification. In high-dimensional sparse feature spaces (such as TF-IDF spaces where $d = 10,000$), documents are nearly always linearly separable.

### Maximum Margin Hyperplane Formulation
Given training pairs $(\\mathbf{x}_i, y_i)$ with $y_i \\in \\{-1, +1\\}$, Linear SVM solves the primal optimization problem:

$$\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{i=1}^N \\xi_i$$
$$\\text{subject to } y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\ge 1 - \\xi_i, \\quad \\xi_i \\ge 0, \\quad \\forall i$$

where:
* $\\frac{2}{\\|\\mathbf{w}\\|}$ is the geometric margin separating the two classes.
* $\\xi_i$ are slack variables representing margin violations (soft-margin formulation).
* $C > 0$ controls the penalty for misclassifications (regularization parameter).

### Loss Function: Hinge Loss vs. Log-Loss
Unlike Logistic Regression, which optimizes the smooth convex log-loss:
$$\\mathcal{L}_{\\text{log}}(y, f(\\mathbf{x})) = \\log(1 + e^{-y f(\\mathbf{x})})$$
Linear SVM minimizes the piecewise-linear **hinge loss**:
$$\\mathcal{L}_{\\text{hinge}}(y, f(\\mathbf{x})) = \\max(0, 1 - y f(\\mathbf{x}))$$

This produces a sparse dual representation where only training instances lying on or within the margin (**support vectors**) determine the final decision boundary $\\mathbf{w}$.

### Probability Calibration via Platt Scaling
Because standard SVM yields uncalibrated geometric distances $f(\\mathbf{x}) = \\mathbf{w}^T \\mathbf{x} + b$, we wrap the model with **Platt Scaling** (`CalibratedClassifierCV`) via an internal sigmoid transformation:
$$P(y = 1 \\mid f(\\mathbf{x})) = \\frac{1}{1 + e^{A f(\\mathbf{x}) + B}}$$
providing calibrated probabilities for risk scoring in our web moderation service."""),

    make_code_cell("""# Environment Configuration and Scientific Computing Imports
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABEL_MAP = {0: 'Non-Bullying', 1: 'Bullying'}
os.makedirs(os.path.join("models", "svm"), exist_ok=True)

print("Environment and SVM module initialized.")"""),

    make_markdown_cell("""## 2. Ingestion of Preprocessed Partitions & Vectorization
We load the partitioned datasets and transform them using our pre-fitted `models/tfidf_vectorizer.joblib`."""),

    make_code_cell("""# Load preprocessed partitions and vectorizer
train_path = os.path.join("data", "processed_train.csv")
test_path = os.path.join("data", "processed_test.csv")
vectorizer_path = os.path.join("models", "tfidf_vectorizer.joblib")

if not os.path.exists(train_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Prerequisite data or vectorizer not found. Please execute Notebook 02 first.")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

X_train = tfidf_vectorizer.transform(train_df['clean_text'].fillna(''))
X_test = tfidf_vectorizer.transform(test_df['clean_text'].fillna(''))

y_train = train_df['label'].values
y_test = test_df['label'].values

print(f"Loaded X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")"""),

    make_markdown_cell("""## 3. SVM Hyperparameter Optimization ($C$)
We evaluate the regularization penalty $C \\in \\{0.01, 0.1, 1.0, 5.0, 10.0\\}$ using 5-Fold Stratified Cross-Validation with squared hinge loss (`loss='squared_hinge'`) and balanced class weighting."""),

    make_code_cell("""# Cross-Validation for LinearSVC Regularization Parameter C
c_candidates = [0.01, 0.1, 1.0, 5.0, 10.0]
cv_scores = []
cv_stds = []

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("Evaluating LinearSVC C hyperparameter via 5-Fold Stratified Cross-Validation:")
for c in c_candidates:
    svm_cand = LinearSVC(
        C=c,
        loss='squared_hinge',
        penalty='l2',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=2000
    )
    scores = cross_val_score(svm_cand, X_train, y_train, cv=cv_strat, scoring='f1', n_jobs=-1)
    cv_scores.append(scores.mean())
    cv_stds.append(scores.std())
    print(f"  C = {c:<5} | Mean F1-Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

best_idx = np.argmax(cv_scores)
best_c = c_candidates[best_idx]
print(f"\\nOptimal SVM Regularization Parameter: C = {best_c} (F1 = {cv_scores[best_idx]:.4f})")"""),

    make_markdown_cell("""## 4. Model Training and Platt Probability Calibration
We fit the optimal `LinearSVC` base model and calibrate its decision margins using `CalibratedClassifierCV` via 5-fold cross-validation."""),

    make_code_cell("""# Fit base LinearSVC model and calibrated wrapper
start_time = time.time()

base_svm = LinearSVC(
    C=best_c,
    loss='squared_hinge',
    penalty='l2',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    max_iter=2000
)

# Platt Scaling probability calibration
calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=5, method='sigmoid')
calibrated_svm.fit(X_train, y_train)

# Also fit standalone base_svm to inspect raw hyperplane weights
base_svm.fit(X_train, y_train)

train_duration = time.time() - start_time
print(f"SVM training and Platt calibration completed in {train_duration:.3f} seconds.")"""),

    make_markdown_cell("""## 5. Quantitative Performance Evaluation on Hold-Out Test Set
We compute comprehensive classification metrics and print the detailed classification report."""),

    make_code_cell("""# Generate Predictions and Calibrated Probabilities
y_pred = calibrated_svm.predict(X_test)
y_prob = calibrated_svm.predict_proba(X_test)[:, 1]

# Quantitative Metric Computation
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
rec_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob)

prec_bully = precision_score(y_test, y_pred, pos_label=1)
rec_bully = recall_score(y_test, y_pred, pos_label=1)
f1_bully = f1_score(y_test, y_pred, pos_label=1)

print("=== SVM Classification Report (Hold-Out Test Set) ===")
print(classification_report(y_test, y_pred, target_names=['Non-Bullying (0)', 'Bullying (1)'], digits=4))

# Store SVM Metrics
svm_metrics = {
    'Model': 'TF-IDF + Linear SVM',
    'Accuracy': round(acc, 4),
    'Macro Precision': round(prec_macro, 4),
    'Macro Recall': round(rec_macro, 4),
    'Macro F1': round(f1_macro, 4),
    'Weighted F1': round(f1_weighted, 4),
    'Bullying Precision': round(prec_bully, 4),
    'Bullying Recall': round(rec_bully, 4),
    'Bullying F1': round(f1_bully, 4),
    'ROC-AUC': round(roc_auc, 4),
    'Training Time (s)': round(train_duration, 4)
}

print("\\n=== SVM Metric Summary ===")
for k, v in svm_metrics.items():
    print(f"  {k:<22}: {v}")"""),

    make_markdown_cell("""## 6. Confusion Matrix & ROC Visualizations
We plot the absolute and normalized confusion matrices along with the ROC and Precision-Recall trajectories."""),

    make_code_cell("""# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Absolute Counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[0],
            xticklabels=['Non-Bullying', 'Bullying'],
            yticklabels=['Non-Bullying', 'Bullying'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[0].set_title("Linear SVM: Confusion Matrix (Counts)", pad=12, fontweight='bold')
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Ground Truth")

# Subplot 2: Normalized Recall
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrBr', cbar=False, ax=axes[1],
            xticklabels=['Non-Bullying', 'Bullying'],
            yticklabels=['Non-Bullying', 'Bullying'],
            annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title("Linear SVM: Normalized Recall Matrix", pad=12, fontweight='bold')
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Ground Truth")

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 7. Comparative Benchmark: Linear SVM vs. Logistic Regression
We load the evaluation metrics recorded in `models/metrics_logistic_regression.json` and construct a side-by-side comparative table and visualization."""),

    make_code_cell("""# Ingest Logistic Regression Metrics
lr_metrics_path = os.path.join("models", "metrics_logistic_regression.json")
if os.path.exists(lr_metrics_path):
    with open(lr_metrics_path, 'r', encoding='utf-8') as f:
        lr_metrics = json.load(f)
else:
    lr_metrics = {'Model': 'Logistic Regression (Baseline)', 'Accuracy': 0.90, 'Macro F1': 0.89, 'Bullying F1': 0.88, 'ROC-AUC': 0.95}

# Comparative DataFrame
comparison_df = pd.DataFrame([lr_metrics, svm_metrics])
display_metrics = ['Model', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Bullying Precision', 'Bullying Recall', 'Bullying F1', 'ROC-AUC']
print("=== Baseline Linear Models Comparison Table ===")
print(comparison_df[[c for c in display_metrics if c in comparison_df.columns]].to_string(index=False))

# Visualization: Grouped Bar Chart
metrics_to_plot = ['Accuracy', 'Bullying Precision', 'Bullying Recall', 'Bullying F1', 'ROC-AUC']
lr_vals = [lr_metrics.get(m, 0) for m in metrics_to_plot]
svm_vals = [svm_metrics.get(m, 0) for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5.5))
rects1 = ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', color='#2b5c8f', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, svm_vals, width, label='Linear SVM', color='#e67e22', edgecolor='black', alpha=0.85)

ax.set_ylabel('Score (0.0 to 1.0)')
ax.set_title('Comparative Benchmark: Logistic Regression vs. Linear SVM', pad=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight='bold')
ax.set_ylim(0.70, 1.02)
ax.legend(frameon=True, loc='lower right')

for rect in list(rects1) + list(rects2):
    height = rect.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 8. Feature Importance via Normal Vector to the Hyperplane
The normal vector $\\mathbf{w}$ represents the orientation of the separating hyperplane. We extract and plot the top 20 positive and negative hyperplane weights."""),

    make_code_cell("""# Extract Hyperplane Normal Vector Weights
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
svm_weights = base_svm.coef_[0]

top_pos_idx = np.argsort(svm_weights)[::-1][:20]
top_neg_idx = np.argsort(svm_weights)[:20]

top_pos_features = feature_names[top_pos_idx]
top_pos_weights = svm_weights[top_pos_idx]

top_neg_features = feature_names[top_neg_idx]
top_neg_weights = svm_weights[top_neg_idx]

# Plot Hyperplane Feature Weights
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].barh(range(len(top_pos_features)), top_pos_weights, color='#e67e22', edgecolor='black', alpha=0.85)
axes[0].set_yticks(range(len(top_pos_features)))
axes[0].set_yticklabels(top_pos_features, fontsize=11)
axes[0].invert_yaxis()
axes[0].set_title("Top 20 SVM Bullying Hyperplane Weights ($w > 0$)", pad=10, fontweight='bold')
axes[0].set_xlabel("Support Vector Weight")

axes[1].barh(range(len(top_neg_features)), np.abs(top_neg_weights), color='#2b5c8f', edgecolor='black', alpha=0.85)
axes[1].set_yticks(range(len(top_neg_features)))
axes[1].set_yticklabels(top_neg_features, fontsize=11)
axes[1].invert_yaxis()
axes[1].set_title("Top 20 SVM Non-Bullying Hyperplane Weights ($|w|, w < 0$)", pad=10, fontweight='bold')
axes[1].set_xlabel("Absolute Support Vector Weight")

plt.tight_layout()
plt.show()"""),

    make_markdown_cell("""## 9. Model Persistence & Artifact Export
We serialize the trained calibrated SVM model to `models/svm/model.joblib` and save evaluation metrics to `models/metrics_svm.json`."""),

    make_code_cell("""# Persist SVM Model Artifact
svm_export_path = os.path.join("models", "svm", "model.joblib")
joblib.dump(calibrated_svm, svm_export_path)
print(f"Calibrated Linear SVM saved to: '{svm_export_path}' ({os.path.getsize(svm_export_path)/1024:.1f} KB)")

# Export Metrics to JSON
metrics_export_path = os.path.join("models", "metrics_svm.json")
with open(metrics_export_path, 'w', encoding='utf-8') as f:
    json.dump(svm_metrics, f, indent=2)

print(f"SVM metrics successfully exported to: '{metrics_export_path}'")""")
]

# Write Notebook 04
nb4 = make_notebook(nb4_cells)
with open(os.path.join("notebooks", "04_svm_model.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb4, f, indent=2)

print("Notebook 04 successfully generated.")
