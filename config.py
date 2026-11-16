"""
CyberShield: Cyberbullying Detection from Social Media
Configuration Module

This module defines project-wide constants, file paths, model hyperparameters,
dataset specifications, label mappings, and web application settings.
"""

from pathlib import Path
import os
import torch

# ==============================================================================
# Base & Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Dataset directories
DATA_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models directory & specific model subdirectories
MODELS_DIR = BASE_DIR / "models"
LOGISTIC_REGRESSION_DIR = MODELS_DIR / "logistic_regression"
SVM_DIR = MODELS_DIR / "svm"
DISTILBERT_DIR = MODELS_DIR / "distilbert"

# Database directory
DATABASE_DIR = BASE_DIR / "database"

# Reports & evaluation outputs directory
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ==============================================================================
# Model & Asset File Paths
# ==============================================================================
LR_MODEL_PATH = LOGISTIC_REGRESSION_DIR / "model.joblib"
SVM_MODEL_PATH = SVM_DIR / "model.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
DISTILBERT_MODEL_DIR = DISTILBERT_DIR

# Database Path (SQLite)
DB_PATH = DATABASE_DIR / "predictions.db"

# Data File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "cyberbullying_dataset.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

# Ensure runtime directories exist
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LOGISTIC_REGRESSION_DIR,
    SVM_DIR,
    DISTILBERT_DIR,
    DATABASE_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Global & Random Seed Configuration
# ==============================================================================
RANDOM_SEED = 42

# ==============================================================================
# Dataset Configuration & Label Mapping
# ==============================================================================
DATASET_NAME = "cike-dev/cyberbullying_dataset"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Label mapping consistent across all modules
# 0: Non-Bullying, 1: Bullying
LABEL_MAP = {
    0: "Non-Bullying",
    1: "Bullying",
}

REVERSE_LABEL_MAP = {
    "Non-Bullying": 0,
    "Bullying": 1,
}

# Raw dataset binary label string/int normalization map
# The cike-dev/cyberbullying_dataset labels: 'normal' -> 0, 'bully' -> 1
RAW_LABEL_MAP = {
    "normal": 0,
    "non-bullying": 0,
    "0": 0,
    0: 0,
    "bully": 1,
    "bullying": 1,
    "1": 1,
    1: 1,
}

# Dataset split proportions
TEST_SPLIT_RATIO = 0.15
VAL_SPLIT_RATIO = 0.15

# ==============================================================================
# Model Hyperparameters
# ==============================================================================

# TF-IDF Feature Extractor Configuration
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

# Logistic Regression Hyperparameters
LOGISTIC_REGRESSION_CONFIG = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# Linear Support Vector Machine (LinearSVC) Hyperparameters
SVM_CONFIG = {
    "C": 1.0,
    "loss": "squared_hinge",
    "max_iter": 2000,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

# DistilBERT Fine-Tuning Hyperparameters
DISTILBERT_CONFIG = {
    "pretrained_model_name": "distilbert-base-uncased",
    "max_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "logging_steps": 50,
    "eval_steps": 200,
    "save_total_limit": 2,
    "seed": RANDOM_SEED,
    "fp16": torch.cuda.is_available(),
}

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# Flask Application Configuration
# ==============================================================================
class FlaskConfig:
    """Flask web application configuration parameters."""
    SECRET_KEY = os.environ.get(
        "SECRET_KEY", "cybershield-secret-key-2024-replace-in-production"
    )
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
    PORT = int(os.environ.get("FLASK_PORT", 5000))
    DATABASE_URI = f"sqlite:///{DB_PATH}"
    AVAILABLE_MODELS = [
        "distilbert",
        "logistic_regression",
        "svm",
    ]
    DEFAULT_MODEL = "distilbert"
