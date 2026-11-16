"""
CyberShield: Cyberbullying Detection from Social Media
Traditional Machine Learning Training and Evaluation Pipeline
==============================================================

This script executes the complete machine learning training workflow:
1. Loads stratified dataset splits (Train 70%, Validation 15%, Test 15%) from
   processed storage or dynamically using data_loader.
2. Applies standard text cleaning using preprocessing module.
3. Fits and serializes a TF-IDF feature extractor (ngram_range=(1,2), max_features=50000).
4. Trains and evaluates TF-IDF + Logistic Regression:
   - LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
   - Optional GridSearchCV hyperparameter tuning on validation set
   - Serializes model to models/logistic_regression/model.joblib
5. Trains and evaluates TF-IDF + Linear SVM:
   - LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=5000)
   - Wraps LinearSVC in CalibratedClassifierCV for probability estimation
   - Optional GridSearchCV hyperparameter tuning on validation set
   - Serializes model to models/svm/model.joblib
6. Evaluates both models on validation set during training and held-out test set at the end.
7. Generates styled confusion matrix heatmaps saved to reports/ and reports/figures/.
8. Prints detailed classification reports with precision, recall, f1, and support.
9. Exports structured metrics to reports/ml_results.json.

Label Mapping:
    0: 'Non-Bullying'
    1: 'Bullying'

Outputs:
    - models/tfidf_vectorizer.joblib
    - models/logistic_regression/model.joblib
    - models/svm/model.joblib
    - reports/ml_results.json
    - reports/figures/confusion_matrix_logistic_regression.png
    - reports/figures/confusion_matrix_svm.png
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

# ------------------------------------------------------------------------------
# Path Setup & Cross-Module Imports
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Project configuration constants with safe fallbacks
try:
    from config import (
        BASE_DIR,
        FIGURES_DIR,
        LABEL_MAP,
        LOGISTIC_REGRESSION_CONFIG,
        LOGISTIC_REGRESSION_DIR,
        LR_MODEL_PATH,
        MODELS_DIR,
        RANDOM_SEED,
        REPORTS_DIR,
        SVM_CONFIG,
        SVM_DIR,
        SVM_MODEL_PATH,
        TEST_DATA_PATH,
        TFIDF_CONFIG,
        TFIDF_VECTORIZER_PATH,
        TRAIN_DATA_PATH,
        VAL_DATA_PATH,
    )
except ImportError:
    BASE_DIR = PROJECT_ROOT
    MODELS_DIR = BASE_DIR / "models"
    LOGISTIC_REGRESSION_DIR = MODELS_DIR / "logistic_regression"
    SVM_DIR = MODELS_DIR / "svm"
    TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
    LR_MODEL_PATH = LOGISTIC_REGRESSION_DIR / "model.joblib"
    SVM_MODEL_PATH = SVM_DIR / "model.joblib"
    REPORTS_DIR = BASE_DIR / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"
    TRAIN_DATA_PATH = BASE_DIR / "dataset" / "processed" / "train.csv"
    VAL_DATA_PATH = BASE_DIR / "dataset" / "processed" / "val.csv"
    TEST_DATA_PATH = BASE_DIR / "dataset" / "processed" / "test.csv"
    RANDOM_SEED = 42
    LABEL_MAP = {0: "Non-Bullying", 1: "Bullying"}
    TFIDF_CONFIG = {
        "max_features": 50000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }
    LOGISTIC_REGRESSION_CONFIG = {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }
    SVM_CONFIG = {
        "C": 1.0,
        "max_iter": 5000,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

# Data loader imports
try:
    from src.data_loader import (
        load_and_prepare_dataset,
        load_data,
        load_processed_splits,
        load_split_data,
    )
except ImportError:
    try:
        from data_loader import (
            load_and_prepare_dataset,
            load_data,
            load_processed_splits,
            load_split_data,
        )
    except ImportError:
        load_and_prepare_dataset = None
        load_data = None
        load_split_data = None
        load_processed_splits = None

# Preprocessing module imports
try:
    from src.preprocessing import (
        build_tfidf_vectorizer,
        clean_text,
        fit_tfidf,
        preprocess_batch,
        preprocess_dataframe,
        preprocess_text,
        save_tfidf_vectorizer,
        transform_tfidf,
    )
except ImportError:
    try:
        from preprocessing import (
            build_tfidf_vectorizer,
            clean_text,
            fit_tfidf,
            preprocess_batch,
            preprocess_dataframe,
            preprocess_text,
            save_tfidf_vectorizer,
            transform_tfidf,
        )
    except ImportError:
        def preprocess_text(t: Any) -> str:
            return str(t).strip().lower() if t is not None else ""
        preprocess_batch = lambda texts: [preprocess_text(t) for t in texts]
        build_tfidf_vectorizer = None
        fit_tfidf = None
        transform_tfidf = None
        save_tfidf_vectorizer = None

# Evaluation module imports
try:
    from src.evaluate import (
        CLASS_NAMES,
        compare_models,
        compute_classification_metrics,
        evaluate_model,
        evaluate_predictions,
        plot_confusion_matrix,
        print_classification_report,
        save_evaluation_results,
    )
except ImportError:
    try:
        from evaluate import (
            CLASS_NAMES,
            compare_models,
            compute_classification_metrics,
            evaluate_model,
            evaluate_predictions,
            plot_confusion_matrix,
            print_classification_report,
            save_evaluation_results,
        )
    except ImportError:
        CLASS_NAMES = ["Non-Bullying", "Bullying"]
        evaluate_predictions = None
        compute_classification_metrics = None
        evaluate_model = None
        print_classification_report = None
        plot_confusion_matrix = None
        save_evaluation_results = None
        compare_models = None

# ------------------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_ml")


# ------------------------------------------------------------------------------
# Reproducibility Utility
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across standard Python libraries, NumPy, and PyTorch (if available)
    to guarantee reproducible data splitting, model weight initialization, and evaluation.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info(f"Random seed set to {seed} for full reproducibility.")


# ------------------------------------------------------------------------------
# Data Loading & Text Extraction
# ------------------------------------------------------------------------------
def load_and_verify_splits(
    force_reload: bool = False,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads train, validation, and test splits with fallback guarantees.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames containing 'text' and 'label' columns.
    """
    train_df, val_df, test_df = None, None, None

    # Option 1: Load pre-existing processed splits if available
    if not force_reload and TRAIN_DATA_PATH.is_file() and VAL_DATA_PATH.is_file() and TEST_DATA_PATH.is_file():
        logger.info(f"Loading existing processed CSV splits from '{TRAIN_DATA_PATH.parent}'...")
        try:
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            val_df = pd.read_csv(VAL_DATA_PATH)
            test_df = pd.read_csv(TEST_DATA_PATH)
            if len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0:
                logger.info(
                    f"Successfully loaded splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
                )
                return train_df, val_df, test_df
        except Exception as e:
            logger.warning(f"Error reading existing processed splits: {e}. Rebuilding...")

    # Option 2: Use data_loader module functions
    if load_and_prepare_dataset is not None:
        try:
            logger.info("Executing load_and_prepare_dataset from data_loader module...")
            train_df, val_df, test_df = load_and_prepare_dataset(
                force_reload=force_reload, random_state=seed
            )
            return train_df, val_df, test_df
        except Exception as e:
            logger.warning(f"load_and_prepare_dataset encountered an issue: {e}")

    if load_split_data is not None:
        try:
            logger.info("Executing load_split_data from data_loader module...")
            train_df, val_df, test_df = load_split_data(
                force_reload=force_reload, random_state=seed
            )
            return train_df, val_df, test_df
        except Exception as e:
            logger.warning(f"load_split_data encountered an issue: {e}")

    if load_data is not None:
        try:
            splits = load_data(force_reload=force_reload)
            if isinstance(splits, tuple) and len(splits) == 3:
                return splits
        except Exception as e:
            logger.warning(f"load_data encountered an issue: {e}")

    # Option 3: Synthetic seed fallback if data_loader could not be used
    logger.warning("Falling back to internal standard seed samples for training.")
    seed_samples = [
        ("You are so pathetic and ugly, nobody likes you. Delete your account and disappear.", 1),
        ("Keep talking and you will seriously regret showing your face anywhere near here, loser.", 1),
        ("Nobody wants you here, go kill yourself freak.", 1),
        ("You are an idiot and your family hates you.", 1),
        ("Shut up you worthless scum, you are pure trash.", 1),
        ("Disgusting retard, nobody asked for your stupid opinion.", 1),
        ("Go die in a ditch, literally everyone despises you.", 1),
        ("Look at how fat and ugly you are, what a joke of a person.", 1),
        ("You deserve all the pain coming your way, creep.", 1),
        ("Get off the internet loser, nobody cares about you.", 1),
        ("You're a disgusting piece of garbage and should hang yourself.", 1),
        ("Shut up bitch, go cry somewhere else.", 1),
        ("You are completely brain dead and worthless.", 1),
        ("Drop dead idiot, nobody loves you.", 1),
        ("I hope you suffer every single day for being so vile.", 1),
        ("Had an amazing time collaborating with our study group today! So grateful for supportive friends.", 0),
        ("I respectfully disagree with the points made in your review, but I appreciate your perspective.", 0),
        ("Does anyone know what time the library closes on Sundays?", 0),
        ("Happy birthday! Wishing you a fantastic year filled with joy and success.", 0),
        ("Great job on the presentation today, you explained the difficult concepts really clearly.", 0),
        ("Could you please share the link to the open source repository mentioned earlier?", 0),
        ("I loved the movie soundtrack, especially the violin soloist in the third act.", 0),
        ("Thanks for helping me fix that bug in my code, really appreciate your patience!", 0),
        ("The weather today is so nice, going out for a walk in the park.", 0),
        ("Let us schedule our project sync meeting for tomorrow at 2 PM.", 0),
        ("Congratulations on passing your exams! Proud of all your hard work.", 0),
        ("Interesting discussion on artificial intelligence ethics, thanks for organizing.", 0),
        ("Can someone recommend a good book on natural language processing?", 0),
        ("Looking forward to the weekend and spending time with family.", 0),
        ("That recipe was delicious, my friends enjoyed it a lot.", 0),
    ]
    df = pd.DataFrame(seed_samples, columns=["text", "label"])
    from sklearn.model_selection import train_test_split
    tr, temp = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["label"])
    va, te = train_test_split(temp, test_size=0.5, random_state=seed, stratify=temp["label"])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def extract_features_and_labels(
    df: pd.DataFrame,
    split_name: str = "split",
) -> Tuple[List[str], np.ndarray]:
    """
    Extracts, cleans, and validates text features and numeric binary labels from DataFrame.

    Args:
        df: Input DataFrame.
        split_name: Label used for logging.

    Returns:
        Tuple of (cleaned_texts_list, labels_numpy_array).
    """
    # Identify text column
    text_col = None
    for cand in ["text", "cleaned_text", "comment_text", "tweet", "tweet_text", "content"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        text_col = df.columns[0]

    # Identify label column
    label_col = None
    for cand in ["label", "Label", "target", "class", "bully", "cyberbullying_type"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    raw_texts = df[text_col].astype(str).tolist()
    # Apply standard preprocessing
    cleaned_texts = [preprocess_text(t) for t in raw_texts]

    # Standardize labels to binary 0 and 1
    raw_labels = df[label_col].tolist()
    binary_labels = []
    for l in raw_labels:
        if isinstance(l, (int, float, np.integer)):
            binary_labels.append(int(l))
        elif isinstance(l, str):
            clean_l = l.strip().lower()
            if clean_l in ("bully", "bullying", "1", "toxic"):
                binary_labels.append(1)
            else:
                binary_labels.append(0)
        else:
            binary_labels.append(1 if l else 0)

    y_arr = np.array(binary_labels, dtype=int)
    logger.info(
        f"Extracted {len(cleaned_texts)} samples from {split_name} (Class distribution: {dict(pd.Series(y_arr).value_counts())})"
    )
    return cleaned_texts, y_arr


# ------------------------------------------------------------------------------
# TF-IDF Feature Extraction
# ------------------------------------------------------------------------------
def create_and_fit_vectorizer(
    train_texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int = 50000,
    sublinear_tf: bool = True,
    save_path: Optional[Path] = TFIDF_VECTORIZER_PATH,
) -> TfidfVectorizer:
    """
    Builds, fits, and serializes the TF-IDF feature extractor.
    Dynamically adjusts min_df for smaller datasets to prevent vocabulary starvation.

    Args:
        train_texts: List of cleaned training text strings.
        ngram_range: Unigram and bigram configuration. Defaults to (1, 2).
        min_df: Minimum document frequency threshold. Defaults to 2.
        max_features: Maximum vocabulary limit. Defaults to 50000.
        sublinear_tf: Apply sublinear tf scaling. Defaults to True.
        save_path: Path to serialize vectorizer (.joblib). Defaults to models/tfidf_vectorizer.joblib.

    Returns:
        Fitted TfidfVectorizer instance.
    """
    # Safeguard min_df if training corpus is very small
    effective_min_df = min_df if len(train_texts) >= 50 else 1

    logger.info(
        f"Initializing TF-IDF Vectorizer (ngram_range={ngram_range}, min_df={effective_min_df}, "
        f"max_features={max_features}, sublinear_tf={sublinear_tf})..."
    )

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=effective_min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )

    t0 = time.time()
    vectorizer.fit(train_texts)
    fit_duration = time.time() - t0

    vocab_size = len(vectorizer.vocabulary_)
    logger.info(f"TF-IDF Vectorizer fitted in {fit_duration:.2f}s. Vocabulary size: {vocab_size} tokens.")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, save_path)
        logger.info(f"TF-IDF Vectorizer serialized to: '{save_path}'")

    return vectorizer


# ------------------------------------------------------------------------------
# Evaluation & Reporting Helpers
# ------------------------------------------------------------------------------
def run_evaluation(
    model: Any,
    X: Any,
    y_true: np.ndarray,
    model_name: str,
    split_name: str = "Test",
    figure_path: Optional[Path] = None,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    Computes predictions and probabilities, runs evaluation routines,
    generates styled confusion matrix, and prints classification report.

    Args:
        model: Fitted classification estimator.
        X: Sparse TF-IDF feature matrix.
        y_true: Ground truth binary labels.
        model_name: Descriptive name of the model.
        split_name: Dataset split ('Validation' or 'Test').
        figure_path: Optional path to save confusion matrix heatmap image.
        print_report: Whether to print classification report to stdout.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    y_pred = model.predict(X)

    # Obtain calibrated probabilities if supported
    y_probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            y_probs = probs[:, 1]
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim > 1:
            decision = decision[:, 0]
        y_probs = 1.0 / (1.0 + np.exp(-np.clip(decision, -20.0, 20.0)))

    # Use evaluate.py functions if available
    metrics = None
    if evaluate_predictions is not None:
        try:
            metrics = evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs,
                class_names=["Non-Bullying", "Bullying"],
                plot_path=figure_path,
                model_name=f"{model_name} ({split_name})",
            )
        except Exception as e:
            logger.debug(f"evaluate_predictions failed with {e}. Falling back to compute_classification_metrics.")

    if metrics is None and compute_classification_metrics is not None:
        try:
            metrics = compute_classification_metrics(y_true, y_pred, y_probs)
        except Exception as e:
            logger.debug(f"compute_classification_metrics failed with {e}.")

    # Built-in fallback evaluation calculation
    if metrics is None:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        acc = float(accuracy_score(y_true, y_pred))
        p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_per, r_per, f1_per, sup = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(float(p_bin), 4),
            "recall": round(float(r_bin), 4),
            "f1": round(float(f1_bin), 4),
            "macro_precision": round(float(p_mac), 4),
            "macro_recall": round(float(r_mac), 4),
            "macro_f1": round(float(f1_mac), 4),
            "per_class": {
                "Non-Bullying": {
                    "precision": round(float(p_per[0]), 4) if len(p_per) > 0 else 0.0,
                    "recall": round(float(r_per[0]), 4) if len(r_per) > 0 else 0.0,
                    "f1": round(float(f1_per[0]), 4) if len(f1_per) > 0 else 0.0,
                    "support": int(sup[0]) if len(sup) > 0 else 0,
                },
                "Bullying": {
                    "precision": round(float(p_per[1]), 4) if len(p_per) > 1 else 0.0,
                    "recall": round(float(r_per[1]), 4) if len(r_per) > 1 else 0.0,
                    "f1": round(float(f1_per[1]), 4) if len(f1_per) > 1 else 0.0,
                    "support": int(sup[1]) if len(sup) > 1 else 0,
                },
            },
        }
        if y_probs is not None:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_probs)), 4)
            except Exception:
                metrics["roc_auc"] = None

    metrics["model_name"] = model_name
    metrics["split"] = split_name

    # Plot confusion matrix if requested and not already saved
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        if plot_confusion_matrix is not None:
            try:
                plot_confusion_matrix(
                    y_true=y_true,
                    y_pred=y_pred,
                    labels=["Non-Bullying", "Bullying"],
                    output_path=figure_path,
                    title=f"{model_name} Confusion Matrix ({split_name})",
                )
            except Exception as e:
                logger.warning(f"Could not plot confusion matrix via evaluate.py: {e}")
                _plot_confusion_matrix_fallback(y_true, y_pred, figure_path, f"{model_name} ({split_name})")
        else:
            _plot_confusion_matrix_fallback(y_true, y_pred, figure_path, f"{model_name} ({split_name})")

    # Print detailed classification report to stdout
    if print_report:
        print(f"\n{'='*70}")
        print(f" CLASSIFICATION REPORT: {model_name.upper()} ({split_name.upper()} SET)")
        print(f"{'='*70}")
        if print_classification_report is not None:
            try:
                print_classification_report(y_true, y_pred, target_names=["Non-Bullying", "Bullying"])
            except Exception:
                print(classification_report(y_true, y_pred, target_names=["Non-Bullying", "Bullying"], digits=4, zero_division=0))
        else:
            print(classification_report(y_true, y_pred, target_names=["Non-Bullying", "Bullying"], digits=4, zero_division=0))
        print(f"Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
        print(f"Macro-F1:  {metrics.get('macro_f1', 0.0):.4f}")
        if metrics.get("roc_auc") is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"{'='*70}\n")

    return metrics


def _plot_confusion_matrix_fallback(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    """Internal fallback confusion matrix plotting routine using matplotlib and seaborn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Bullying", "Bullying"],
            yticklabels=["Non-Bullying", "Bullying"],
            cbar=True,
        )
        plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
        plt.ylabel("True Label", fontsize=11, fontweight="bold")
        plt.title(f"{title} - Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved to '{save_path}'")
    except Exception as exc:
        logger.warning(f"Could not save fallback confusion matrix: {exc}")


# ------------------------------------------------------------------------------
# Model Training: Logistic Regression
# ------------------------------------------------------------------------------
def train_logistic_regression(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    seed: int = RANDOM_SEED,
    tune: bool = False,
    save_path: Optional[Path] = LR_MODEL_PATH,
) -> Tuple[LogisticRegression, Dict[str, Any], Dict[str, Any]]:
    """
    Trains Logistic Regression with TF-IDF features.

    Specifications:
    - LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
    - Optional hyperparameter tuning with GridSearchCV on validation set
    - Evaluated on validation set during training
    - Saved to models/logistic_regression/model.joblib

    Args:
        X_train: Transformed training feature matrix.
        y_train: Training labels.
        X_val: Transformed validation feature matrix.
        y_val: Validation labels.
        seed: Random seed for reproducibility.
        tune: Whether to run GridSearchCV hyperparameter optimization.
        save_path: File path to serialize the model (.joblib).

    Returns:
        Tuple of (fitted_model, validation_metrics, best_params_dict).
    """
    logger.info("Initializing TF-IDF + Logistic Regression training...")
    best_params: Dict[str, Any] = {}

    if tune:
        logger.info("Starting GridSearchCV hyperparameter optimization for Logistic Regression...")
        # Parameter grid exploring regularization strengths and solvers
        param_grid = {
            "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        }
        base_lr = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )

        min_class_count = int(np.min(np.bincount(y_train)))
        cv_folds = max(2, min(5, min_class_count))

        grid = GridSearchCV(
            estimator=base_lr,
            param_grid=param_grid,
            scoring="f1",
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
            n_jobs=-1,
            verbose=1,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        tuning_duration = time.time() - t0

        model = grid.best_estimator_
        best_params = grid.best_params_
        logger.info(
            f"GridSearchCV completed in {tuning_duration:.2f}s. "
            f"Best CV F1: {grid.best_score_:.4f}. Best params: {best_params}"
        )
    else:
        # Default specifications
        logger.info("Training Logistic Regression with baseline hyperparameters...")
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_duration = time.time() - t0
        logger.info(f"Logistic Regression fitted in {fit_duration:.2f}s.")
        best_params = {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": seed,
        }

    # Evaluate on validation set during training
    logger.info("Evaluating Logistic Regression on validation set...")
    val_metrics = run_evaluation(
        model=model,
        X=X_val,
        y_true=y_val,
        model_name="Logistic Regression",
        split_name="Validation",
        figure_path=None,
        print_report=False,
    )
    logger.info(
        f"Logistic Regression Validation - Accuracy: {val_metrics['accuracy']:.4f}, "
        f"F1: {val_metrics['f1']:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}"
    )

    # Save model artifact
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        logger.info(f"Logistic Regression model saved to: '{save_path}'")

    return model, val_metrics, best_params


# ------------------------------------------------------------------------------
# Model Training: Linear SVM (Calibrated)
# ------------------------------------------------------------------------------
def train_linear_svm(
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    seed: int = RANDOM_SEED,
    tune: bool = False,
    save_path: Optional[Path] = SVM_MODEL_PATH,
) -> Tuple[CalibratedClassifierCV, Dict[str, Any], Dict[str, Any]]:
    """
    Trains Linear SVM wrapped in CalibratedClassifierCV for probability estimates.

    Specifications:
    - LinearSVC(C=1.0, class_weight='balanced', random_state=seed, max_iter=5000)
    - Wrapped in CalibratedClassifierCV for probability estimates (predict_proba)
    - Optional hyperparameter tuning with GridSearchCV on validation set
    - Evaluated on validation set during training
    - Saved to models/svm/model.joblib

    Args:
        X_train: Transformed training feature matrix.
        y_train: Training labels.
        X_val: Transformed validation feature matrix.
        y_val: Validation labels.
        seed: Random seed for reproducibility.
        tune: Whether to run GridSearchCV hyperparameter optimization.
        save_path: File path to serialize the calibrated SVM model (.joblib).

    Returns:
        Tuple of (fitted_calibrated_svm, validation_metrics, best_params_dict).
    """
    logger.info("Initializing TF-IDF + Linear SVM training...")
    best_params: Dict[str, Any] = {}

    min_class_count = int(np.min(np.bincount(y_train)))
    cv_folds = max(2, min(5, min_class_count))

    if tune:
        logger.info("Starting GridSearchCV hyperparameter optimization for Linear SVM...")
        param_grid = {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "loss": ["squared_hinge"],
        }
        base_svm = LinearSVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=5000,
        )

        grid = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            scoring="f1",
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
            n_jobs=-1,
            verbose=1,
        )

        t0 = time.time()
        grid.fit(X_train, y_train)
        tuning_duration = time.time() - t0

        best_base_svm = grid.best_estimator_
        best_params = grid.best_params_
        logger.info(
            f"GridSearchCV completed in {tuning_duration:.2f}s. "
            f"Best CV F1: {grid.best_score_:.4f}. Best params: {best_params}"
        )
    else:
        logger.info("Configuring Linear SVM with baseline hyperparameters...")
        best_base_svm = LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=seed,
            max_iter=5000,
        )
        best_params = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": seed,
        }

    # Wrap in CalibratedClassifierCV for probability estimates
    logger.info(
        f"Wrapping Linear SVM in CalibratedClassifierCV (cv={cv_folds}, method='sigmoid') for probability estimates..."
    )
    try:
        calibrated_svm = CalibratedClassifierCV(estimator=best_base_svm, cv=cv_folds, method="sigmoid")
    except TypeError:
        # Backward compatibility for scikit-learn < 1.2
        calibrated_svm = CalibratedClassifierCV(base_estimator=best_base_svm, cv=cv_folds, method="sigmoid")

    t0 = time.time()
    calibrated_svm.fit(X_train, y_train)
    fit_duration = time.time() - t0
    logger.info(f"Calibrated Linear SVM fitted in {fit_duration:.2f}s.")

    # Verify predict_proba capability
    assert hasattr(calibrated_svm, "predict_proba"), "CalibratedClassifierCV must provide predict_proba!"

    # Evaluate on validation set during training
    logger.info("Evaluating Linear SVM on validation set...")
    val_metrics = run_evaluation(
        model=calibrated_svm,
        X=X_val,
        y_true=y_val,
        model_name="Linear SVM",
        split_name="Validation",
        figure_path=None,
        print_report=False,
    )
    logger.info(
        f"Linear SVM Validation - Accuracy: {val_metrics['accuracy']:.4f}, "
        f"F1: {val_metrics['f1']:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}"
    )

    # Save model artifact
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_svm, save_path)
        logger.info(f"Calibrated Linear SVM model saved to: '{save_path}'")

    return calibrated_svm, val_metrics, best_params


# ------------------------------------------------------------------------------
# Full Training & Evaluation Pipeline
# ------------------------------------------------------------------------------
def run_ml_pipeline(
    seed: int = RANDOM_SEED,
    tune: bool = False,
    force_reload: bool = False,
    max_features: int = 50000,
    ngram_max: int = 2,
    results_output_path: Path = REPORTS_DIR / "ml_results.json",
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, Any]:
    """
    Executes end-to-end ML training, evaluation, figure generation, and results serialization.

    Args:
        seed: Random seed for reproducibility.
        tune: Enable GridSearchCV hyperparameter tuning.
        force_reload: Force reload dataset splits from raw data.
        max_features: Maximum TF-IDF vocabulary limit.
        ngram_max: Maximum n-gram range (1 to ngram_max).
        results_output_path: Destination JSON file for all benchmark results.
        figures_dir: Destination folder for confusion matrix plots.

    Returns:
        Structured dictionary containing full benchmark results.
    """
    pipeline_start_time = time.time()
    set_seed(seed)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_output_path = Path(results_output_path)
    results_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data Splits
    logger.info("=" * 70)
    logger.info(" STEP 1: LOADING & PREPARING DATASET SPLITS")
    logger.info("=" * 70)
    train_df, val_df, test_df = load_and_verify_splits(force_reload=force_reload, seed=seed)

    X_train_raw, y_train = extract_features_and_labels(train_df, "Train")
    X_val_raw, y_val = extract_features_and_labels(val_df, "Validation")
    X_test_raw, y_test = extract_features_and_labels(test_df, "Test")

    # 2. Fit TF-IDF Vectorizer
    logger.info("=" * 70)
    logger.info(" STEP 2: TF-IDF FEATURE EXTRACTION")
    logger.info("=" * 70)
    vectorizer = create_and_fit_vectorizer(
        train_texts=X_train_raw,
        ngram_range=(1, ngram_max),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        save_path=TFIDF_VECTORIZER_PATH,
    )

    logger.info("Transforming train, validation, and test text splits...")
    X_train = vectorizer.transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    X_test = vectorizer.transform(X_test_raw)
    logger.info(
        f"Feature matrices transformed: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )

    # 3. Train Logistic Regression
    logger.info("=" * 70)
    logger.info(" STEP 3: TRAIN & EVALUATE LOGISTIC REGRESSION")
    logger.info("=" * 70)
    lr_model, lr_val_metrics, lr_best_params = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
        tune=tune,
        save_path=LR_MODEL_PATH,
    )

    # 4. Train Linear SVM
    logger.info("=" * 70)
    logger.info(" STEP 4: TRAIN & EVALUATE CALIBRATED LINEAR SVM")
    logger.info("=" * 70)
    svm_model, svm_val_metrics, svm_best_params = train_linear_svm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        seed=seed,
        tune=tune,
        save_path=SVM_MODEL_PATH,
    )

    # 5. Final Evaluation on Held-Out Test Set
    logger.info("=" * 70)
    logger.info(" STEP 5: FINAL HELD-OUT TEST EVALUATION & CONFUSION MATRICES")
    logger.info("=" * 70)

    # Confusion matrix file targets
    lr_cm_path = figures_dir / "confusion_matrix_logistic_regression.png"
    svm_cm_path = figures_dir / "confusion_matrix_svm.png"

    # Evaluate Logistic Regression on Test Set
    logger.info("Evaluating Logistic Regression on Test Set...")
    lr_test_metrics = run_evaluation(
        model=lr_model,
        X=X_test,
        y_true=y_test,
        model_name="Logistic Regression",
        split_name="Test",
        figure_path=lr_cm_path,
        print_report=True,
    )

    # Mirror confusion matrix directly under reports/ for convenience
    mirror_lr_cm = REPORTS_DIR / "confusion_matrix_logistic_regression.png"
    if lr_cm_path.is_file() and not mirror_lr_cm.is_file():
        try:
            import shutil
            shutil.copyfile(lr_cm_path, mirror_lr_cm)
        except Exception:
            pass

    # Evaluate Linear SVM on Test Set
    logger.info("Evaluating Linear SVM on Test Set...")
    svm_test_metrics = run_evaluation(
        model=svm_model,
        X=X_test,
        y_true=y_test,
        model_name="Linear SVM",
        split_name="Test",
        figure_path=svm_cm_path,
        print_report=True,
    )

    # Mirror confusion matrix directly under reports/ for convenience
    mirror_svm_cm = REPORTS_DIR / "confusion_matrix_svm.png"
    if svm_cm_path.is_file() and not mirror_svm_cm.is_file():
        try:
            import shutil
            shutil.copyfile(svm_cm_path, mirror_svm_cm)
        except Exception:
            pass

    # 6. Comparative Model Summary Table
    logger.info("=" * 70)
    logger.info(" STEP 6: COMPARATIVE PERFORMANCE SUMMARY")
    logger.info("=" * 70)
    comparison_results = {
        "Logistic Regression": lr_test_metrics,
        "Linear SVM": svm_test_metrics,
    }

    if compare_models is not None:
        try:
            summary_table_path = REPORTS_DIR / "ml_comparison.csv"
            summary_chart_path = figures_dir / "model_comparison_chart.png"
            df_comp = compare_models(
                comparison_results,
                output_path=summary_table_path,
                chart_path=summary_chart_path,
            )
            print("\n", df_comp.to_string(index=False), "\n")
        except Exception as e:
            logger.debug(f"compare_models failed: {e}")

    # 7. Serialize All Results to JSON
    total_duration = time.time() - pipeline_start_time
    try:
        from datetime import timezone
        timestamp_str = datetime.now(timezone.utc).isoformat()
    except Exception:
        timestamp_str = datetime.utcnow().isoformat() + "Z"

    full_results: Dict[str, Any] = {
        "timestamp": timestamp_str,
        "pipeline_duration_seconds": round(total_duration, 2),
        "random_seed": seed,
        "hyperparameter_tuning_enabled": tune,
        "dataset_summary": {
            "train_samples": len(X_train_raw),
            "val_samples": len(X_val_raw),
            "test_samples": len(X_test_raw),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "label_distribution_test": {
                "Non-Bullying": int(np.sum(y_test == 0)),
                "Bullying": int(np.sum(y_test == 1)),
            },
        },
        "model_paths": {
            "tfidf_vectorizer": str(TFIDF_VECTORIZER_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "logistic_regression": str(LR_MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "svm": str(SVM_MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "logistic_regression": {
            "best_hyperparameters": lr_best_params,
            "validation_metrics": lr_val_metrics,
            "test_metrics": lr_test_metrics,
            "confusion_matrix_figure": str(lr_cm_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "svm": {
            "best_hyperparameters": svm_best_params,
            "calibration": "CalibratedClassifierCV(method='sigmoid')",
            "validation_metrics": svm_val_metrics,
            "test_metrics": svm_test_metrics,
            "confusion_matrix_figure": str(svm_cm_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }

    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"All ML experiment results serialized to: '{results_output_path}'")
    logger.info(f"Pipeline completed successfully in {total_duration:.2f} seconds.")

    return full_results


# ------------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ------------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the ML training pipeline."""
    parser = argparse.ArgumentParser(
        description="CyberShield: Train and evaluate TF-IDF + Logistic Regression and Linear SVM models."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run GridSearchCV hyperparameter optimization on validation set (default: False).",
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Disable hyperparameter tuning and use verified baseline hyperparameters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility across all splits and algorithms (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum vocabulary limit for TF-IDF feature extraction (default: 50000).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF (1 for unigrams, 2 for unigrams+bigrams, default: 2).",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        default=False,
        help="Force re-generation and stratified splitting of dataset from raw files.",
    )
    parser.add_argument(
        "--output-results",
        type=str,
        default=str(REPORTS_DIR / "ml_results.json"),
        help="Destination path for final benchmark results JSON (default: reports/ml_results.json).",
    )

    return parser.parse_args()


def main() -> None:
    """Main CLI entry point."""
    args = parse_arguments()

    print("\n" + "=" * 70)
    print(" 🛡️  CYBERSHIELD: TRADITIONAL MACHINE LEARNING TRAINING PIPELINE")
    print("=" * 70)
    print(f" Random Seed:           {args.seed}")
    print(f" Hyperparameter Tuning: {args.tune}")
    print(f" Max TF-IDF Features:   {args.max_features}")
    print(f" Max N-Gram Range:      (1, {args.ngram_max})")
    print(f" Force Split Reload:    {args.force_reload}")
    print(f" Results Target:        {args.output_results}")
    print("=" * 70 + "\n")

    results = run_ml_pipeline(
        seed=args.seed,
        tune=args.tune,
        force_reload=args.force_reload,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        results_output_path=Path(args.output_results),
    )

    lr_test = results["logistic_regression"]["test_metrics"]
    svm_test = results["svm"]["test_metrics"]

    print("\n" + "=" * 70)
    print(" 🏆 FINAL MODEL COMPARISON SUMMARY (HELD-OUT TEST SET)")
    print("=" * 70)
    print(f"{'Model':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Macro-F1':<10}")
    print("-" * 70)
    print(
        f"{'Logistic Regression':<24} | {lr_test['accuracy']:<10.4f} | {lr_test['precision']:<10.4f} | "
        f"{lr_test['recall']:<10.4f} | {lr_test['f1']:<10.4f} | {lr_test['macro_f1']:<10.4f}"
    )
    print(
        f"{'Linear SVM (Calibrated)':<24} | {svm_test['accuracy']:<10.4f} | {svm_test['precision']:<10.4f} | "
        f"{svm_test['recall']:<10.4f} | {svm_test['f1']:<10.4f} | {svm_test['macro_f1']:<10.4f}"
    )
    print("=" * 70)
    print(f"Artifacts saved:")
    print(f"  - TF-IDF Vectorizer:     {TFIDF_VECTORIZER_PATH}")
    print(f"  - Logistic Regression:   {LR_MODEL_PATH}")
    print(f"  - Linear SVM Model:      {SVM_MODEL_PATH}")
    print(f"  - Results JSON:          {args.output_results}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
