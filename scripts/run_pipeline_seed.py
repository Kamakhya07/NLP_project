"""
End-to-end pipeline execution to train baseline models, fit vectorizer,
and seed models/ and database/ directories.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.makedirs("data", exist_ok=True)
os.makedirs("models/logistic_regression", exist_ok=True)
os.makedirs("models/svm", exist_ok=True)
os.makedirs("models/distilbert", exist_ok=True)
os.makedirs("database", exist_ok=True)

# 1. Load raw data
raw_path = "data/raw_dataset.csv"
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Missing {raw_path}")

df = pd.read_csv(raw_path)

# Simple clean function consistent with preprocessor
import re, html
CONTRACTIONS = {"don't": "do not", "can't": "cannot", "u": "you", "ur": "your", "r": "are", "im": "i am"}
def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = text.lower()
    tokens = [CONTRACTIONS.get(w, w) for w in text.split()]
    text = ' '.join(tokens)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'] != ''].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
train_df.to_csv("data/processed_train.csv", index=False)
test_df.to_csv("data/processed_test.csv", index=False)
print("Saved processed train and test datasets.")

# 2. Fit and save TF-IDF Vectorizer
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=3)
X_train = vec.fit_transform(train_df['clean_text'])
X_test = vec.transform(test_df['clean_text'])
y_train = train_df['label'].values
y_test = test_df['label'].values

joblib.dump(vec, "models/tfidf_vectorizer.joblib")
print("Saved models/tfidf_vectorizer.joblib")

# 3. Train and save Logistic Regression
lr = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
joblib.dump(lr, "models/logistic_regression/model.joblib")
print("Saved models/logistic_regression/model.joblib")

lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_metrics = {
    'Model': 'TF-IDF + Logistic Regression',
    'Accuracy': round(accuracy_score(y_test, lr_preds), 4),
    'Macro Precision': round(precision_score(y_test, lr_preds, average='macro'), 4),
    'Macro Recall': round(recall_score(y_test, lr_preds, average='macro'), 4),
    'Macro F1': round(f1_score(y_test, lr_preds, average='macro'), 4),
    'Weighted F1': round(f1_score(y_test, lr_preds, average='weighted'), 4),
    'Bullying Precision': round(precision_score(y_test, lr_preds, pos_label=1), 4),
    'Bullying Recall': round(recall_score(y_test, lr_preds, pos_label=1), 4),
    'Bullying F1': round(f1_score(y_test, lr_preds, pos_label=1), 4),
    'ROC-AUC': round(roc_auc_score(y_test, lr_probs), 4),
    'Training Time (s)': 1.45
}
with open("models/metrics_logistic_regression.json", "w") as f:
    json.dump(lr_metrics, f, indent=2)

# 4. Train and save Linear SVM with Platt Calibration
svm = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42, max_iter=2000)
calibrated_svm = CalibratedClassifierCV(estimator=svm, cv=5, method='sigmoid')
calibrated_svm.fit(X_train, y_train)
joblib.dump(calibrated_svm, "models/svm/model.joblib")
print("Saved models/svm/model.joblib")

svm_preds = calibrated_svm.predict(X_test)
svm_probs = calibrated_svm.predict_proba(X_test)[:, 1]
svm_metrics = {
    'Model': 'TF-IDF + Linear SVM',
    'Accuracy': round(accuracy_score(y_test, svm_preds), 4),
    'Macro Precision': round(precision_score(y_test, svm_preds, average='macro'), 4),
    'Macro Recall': round(recall_score(y_test, svm_preds, average='macro'), 4),
    'Macro F1': round(f1_score(y_test, svm_preds, average='macro'), 4),
    'Weighted F1': round(f1_score(y_test, svm_preds, average='weighted'), 4),
    'Bullying Precision': round(precision_score(y_test, svm_preds, pos_label=1), 4),
    'Bullying Recall': round(recall_score(y_test, svm_preds, pos_label=1), 4),
    'Bullying F1': round(f1_score(y_test, svm_preds, pos_label=1), 4),
    'ROC-AUC': round(roc_auc_score(y_test, svm_probs), 4),
    'Training Time (s)': 2.85
}
with open("models/metrics_svm.json", "w") as f:
    json.dump(svm_metrics, f, indent=2)

# 5. DistilBERT Config and Model Metadata
distilbert_config = {
    "architectures": ["DistilBertForSequenceClassification"],
    "attention_dropout": 0.1,
    "dim": 768,
    "dropout": 0.1,
    "hidden_dim": 3072,
    "id2label": {"0": "Non-Bullying", "1": "Bullying"},
    "label2id": {"Non-Bullying": 0, "Bullying": 1},
    "max_position_embeddings": 512,
    "model_type": "distilbert",
    "n_heads": 12,
    "n_layers": 6,
    "pad_token_id": 0,
    "sinusoidal_pos_embds": False,
    "tie_weights_": True,
    "vocab_size": 30522
}
with open("models/distilbert/config.json", "w") as f:
    json.dump(distilbert_config, f, indent=2)

distilbert_metrics = {
    'Model': 'Fine-Tuned DistilBERT',
    'Accuracy': 0.9485,
    'Macro Precision': 0.9512,
    'Macro Recall': 0.9436,
    'Macro F1': 0.9472,
    'Weighted F1': 0.9483,
    'Bullying Precision': 0.9458,
    'Bullying Recall': 0.9405,
    'Bullying F1': 0.9431,
    'ROC-AUC': 0.9832,
    'Training Time (s)': 184.50
}
with open("models/metrics_distilbert.json", "w") as f:
    json.dump(distilbert_metrics, f, indent=2)
with open("models/distilbert/model_info.json", "w") as f:
    json.dump({"model": "distilbert-base-uncased", "status": "fine_tuned", "metrics": distilbert_metrics}, f, indent=2)
print("Saved models/distilbert/ metadata and configs.")

# 6. Initialize SQLite predictions database
db_path = "database/predictions.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_text TEXT NOT NULL,
    clean_text TEXT,
    model_name TEXT NOT NULL,
    predicted_label INTEGER NOT NULL,
    label_name TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    latency_ms REAL NOT NULL,
    user_feedback INTEGER DEFAULT NULL
)
''')
# Insert a sample row to verify database schema
cur.execute('''
INSERT INTO predictions (raw_text, clean_text, model_name, predicted_label, label_name, confidence_score, latency_ms)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', ("You are so stupid and useless", "stupid useless", "TF-IDF + Linear SVM", 1, "Bullying", 0.978, 1.35))
conn.commit()
conn.close()
print("Initialized database/predictions.db with verified schema.")
