"""Cyberbullying Detection from Social Media.

This package provides unified modules for data loading, preprocessing,
model evaluation, baseline modeling, and inference for cyberbullying detection.
"""

from pathlib import Path

# Paths & Project Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP = {0: "Non-Bullying", 1: "Bullying"}
INVERSE_LABEL_MAP = {"Non-Bullying": 0, "Bullying": 1}

# Preprocessing module exports
from src.preprocessing import (
    clean_text,
    preprocess_text,
    preprocess_dataframe,
    build_tfidf_vectorizer,
    fit_tfidf,
    transform_tfidf,
    save_vectorizer,
    load_vectorizer,
)

# Data loader module exports
from src.data_loader import (
    download_dataset,
    inspect_dataset,
    identify_columns,
    map_labels_to_binary,
    create_stratified_split,
    save_processed_splits,
    load_processed_splits,
    load_and_prepare_dataset,
    load_data,
)

# Evaluation module exports
from src.evaluate import (
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    compare_models,
    save_evaluation_results,
    load_evaluation_results,
)

# Optional predictor import (if predict.py is present)
try:
    from src.predict import CyberbullyingPredictor
except ImportError:
    CyberbullyingPredictor = None

__version__ = "1.0.0"

__all__ = [
    "PROJECT_ROOT",
    "LABEL_MAP",
    "INVERSE_LABEL_MAP",
    "clean_text",
    "preprocess_text",
    "preprocess_dataframe",
    "build_tfidf_vectorizer",
    "fit_tfidf",
    "transform_tfidf",
    "save_vectorizer",
    "load_vectorizer",
    "download_dataset",
    "inspect_dataset",
    "identify_columns",
    "map_labels_to_binary",
    "create_stratified_split",
    "save_processed_splits",
    "load_processed_splits",
    "load_and_prepare_dataset",
    "load_data",
    "evaluate_model",
    "print_classification_report",
    "plot_confusion_matrix",
    "compare_models",
    "save_evaluation_results",
    "load_evaluation_results",
    "CyberbullyingPredictor",
]
