"""
CyberShield: Cyberbullying Detection from Social Media
Fine-Tuned DistilBERT Training and Evaluation Pipeline
=====================================================

This module implements the complete fine-tuning workflow for DistilBERT
(distilbert/distilbert-base-uncased) for binary cyberbullying classification:
    - 0: Non-Bullying
    - 1: Bullying

Features:
    - Modular imports from data_loader, preprocessing, and evaluate modules.
    - Custom PyTorch Dataset (CyberbullyingDataset) handling token encodings and labels.
    - Standardized tokenization with max_length=128, truncation=True, padding='max_length'.
    - Hugging Face Trainer integration with automated validation metric tracking.
    - Comprehensive compute_metrics function computing Accuracy, Precision, Recall, F1, and Macro-F1.
    - Automatic hardware acceleration detection (GPU / CPU) with graceful fallback.
    - Best model selection and persistence to models/distilbert/ alongside tokenizer.
    - Serialization of label mapping (label_map.json) for inference compatibility.
    - Full evaluation on held-out test split using evaluate.py functions.
    - Metrics export to reports/distilbert_results.json and confusion matrix plotting.
    - Command-line interface with argparse for customizable hyperparameters (--epochs, --batch_size, --lr, --seed).
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project configurations
try:
    from config import (
        BASE_DIR,
        DISTILBERT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        RANDOM_SEED,
        LABEL_MAP,
    )
except ImportError:
    BASE_DIR = PROJECT_ROOT
    DISTILBERT_DIR = BASE_DIR / "models" / "distilbert"
    REPORTS_DIR = BASE_DIR / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"
    RANDOM_SEED = 42
    LABEL_MAP = {0: "Non-Bullying", 1: "Bullying"}

# Preprocessing module imports
try:
    from src.preprocessing import preprocess_text
except ImportError:
    try:
        from preprocessing import preprocess_text
    except ImportError:
        try:
            from src.preprocess import clean_text as preprocess_text
        except ImportError:
            def preprocess_text(t: Union[str, None]) -> str:
                return str(t).strip().lower() if t is not None else ""

# Data Loader module imports
try:
    from src.data_loader import load_data, load_split_data
except ImportError:
    try:
        from data_loader import load_data, load_split_data
    except ImportError:
        load_data = None
        load_split_data = None

# Evaluation module imports
try:
    from src.evaluate import (
        evaluate_predictions,
        plot_confusion_matrix,
        compute_classification_metrics,
        save_evaluation_results,
    )
except ImportError:
    try:
        from evaluate import (
            evaluate_predictions,
            plot_confusion_matrix,
            compute_classification_metrics,
            save_evaluation_results,
        )
    except ImportError:
        evaluate_predictions = None
        plot_confusion_matrix = None
        compute_classification_metrics = None
        save_evaluation_results = None

# PyTorch and Hugging Face imports with defensive handling
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError as err:
    torch = None
    Dataset = object
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    Trainer = None
    TrainingArguments = None
    set_seed = None

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_distilbert")

# Canonical label mapping constants
CANONICAL_LABEL_MAP: Dict[int, str] = {0: "Non-Bullying", 1: "Bullying"}
CANONICAL_ID_TO_LABEL: Dict[int, str] = {0: "Non-Bullying", 1: "Bullying"}
CANONICAL_LABEL_TO_ID: Dict[str, int] = {"Non-Bullying": 0, "Bullying": 1}


# ==============================================================================
# PyTorch Custom Dataset Class
# ==============================================================================
class CyberbullyingDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenized social media text and binary labels.

    Args:
        encodings (BatchEncoding or dict): Tokenizer output containing 'input_ids'
            and 'attention_mask' arrays/tensors.
        labels (list or np.ndarray, optional): Numerical ground truth labels (0 or 1).
    """

    def __init__(
        self,
        encodings: Dict[str, Any],
        labels: Optional[Union[List[int], np.ndarray]] = None,
    ):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns tensorized sample dictionary for PyTorch DataLoader / Trainer."""
        item = {
            key: torch.tensor(val[idx]) if torch is not None else val[idx]
            for key, val in self.encodings.items()
        }
        if self.labels is not None:
            label_val = int(self.labels[idx])
            item["labels"] = (
                torch.tensor(label_val, dtype=torch.long)
                if torch is not None
                else label_val
            )
        return item

    def __len__(self) -> int:
        """Returns total number of samples in dataset."""
        if self.labels is not None:
            return len(self.labels)
        return len(self.encodings["input_ids"])


# ==============================================================================
# Metrics Calculation for Trainer
# ==============================================================================
def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes validation metrics during Trainer evaluation:
    Accuracy, Binary Precision, Recall, F1 (for class 1: Bullying),
    and Macro Precision, Recall, and Macro-F1.

    Args:
        eval_pred: Tuple containing model logits (predictions) and target label IDs.

    Returns:
        dict: Evaluated scalar metrics formatted for Trainer logging and checkpointing.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Binary metrics for positive class (1: Bullying)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    # Macro-averaged metrics across both classes (0 and 1)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    # Overall classification accuracy
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }


# ==============================================================================
# Fallback / Built-in Evaluation Utilities
# ==============================================================================
def _internal_plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str = "DistilBERT Confusion Matrix",
) -> None:
    """Internal fallback function to plot and save confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
        )
        plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
        plt.ylabel("True Label", fontsize=11, fontweight="bold")
        plt.title(title, fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix figure successfully saved to: {output_path}")
    except Exception as exc:
        logger.warning(f"Could not generate confusion matrix plot: {exc}")


def _internal_evaluate_test_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray],
    results_path: Path,
    plot_path: Path,
) -> Dict[str, Any]:
    """Evaluates test set predictions using evaluate.py or built-in fallback routines."""
    if evaluate_predictions is not None:
        try:
            return evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                y_probs=y_probs,
                class_names=["Non-Bullying", "Bullying"],
                output_path=results_path,
                plot_path=plot_path,
                model_name="Fine-Tuned DistilBERT",
            )
        except Exception as err:
            logger.warning(f"External evaluate_predictions encountered an issue: {err}. Falling back to internal evaluator.")

    # Calculate metrics with sklearn
    acc = float(accuracy_score(y_true, y_pred))
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics: Dict[str, Any] = {
        "model_name": "Fine-Tuned DistilBERT",
        "accuracy": round(acc, 4),
        "precision": round(float(p_bin), 4),
        "recall": round(float(r_bin), 4),
        "f1": round(float(f1_bin), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "per_class": {
            "Non-Bullying": {
                "precision": round(float(p_per[0]), 4) if len(p_per) > 0 else 0.0,
                "recall": round(float(r_per[0]), 4) if len(r_per) > 0 else 0.0,
                "f1": round(float(f1_per[0]), 4) if len(f1_per) > 0 else 0.0,
                "support": int(support[0]) if len(support) > 0 else 0,
            },
            "Bullying": {
                "precision": round(float(p_per[1]), 4) if len(p_per) > 1 else 0.0,
                "recall": round(float(r_per[1]), 4) if len(r_per) > 1 else 0.0,
                "f1": round(float(f1_per[1]), 4) if len(f1_per) > 1 else 0.0,
                "support": int(support[1]) if len(support) > 1 else 0,
            },
        },
    }

    if y_probs is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc_val = float(roc_auc_score(y_true, y_probs[:, 1] if y_probs.ndim == 2 else y_probs))
            metrics["roc_auc"] = round(auc_val, 4)
        except Exception:
            metrics["roc_auc"] = None

    # Generate and save plot
    _internal_plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=["Non-Bullying", "Bullying"],
        output_path=plot_path,
        title="Fine-Tuned DistilBERT - Test Confusion Matrix",
    )

    # Save metrics JSON
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation report written to: {results_path}")

    return metrics


# ==============================================================================
# Data Acquisition Helper
# ==============================================================================
def get_training_splits(
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Loads train, validation, and test datasets using data_loader module
    with preprocessing applied.

    Returns:
        tuple: (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    logger.info("Loading dataset splits...")
    if load_split_data is not None:
        try:
            train_df, val_df, test_df = load_split_data(random_state=seed)
        except Exception as err:
            logger.warning(f"load_split_data failed ({err}). Loading via fallback seed samples.")
            train_df, val_df, test_df = None, None, None
    elif load_data is not None:
        try:
            train_df = load_data("train")
            val_df = load_data("val")
            test_df = load_data("test")
        except Exception as err:
            logger.warning(f"load_data failed ({err}). Loading via fallback.")
            train_df, val_df, test_df = None, None, None
    else:
        train_df, val_df, test_df = None, None, None

    # Fallback to local processed CSVs if DataFrames not returned
    if train_df is None or val_df is None or test_df is None:
        p_train = PROJECT_ROOT / "dataset" / "processed" / "train.csv"
        p_val = PROJECT_ROOT / "dataset" / "processed" / "val.csv"
        p_test = PROJECT_ROOT / "dataset" / "processed" / "test.csv"
        if p_train.is_file() and p_val.is_file() and p_test.is_file():
            train_df = pd.read_csv(p_train)
            val_df = pd.read_csv(p_val)
            test_df = pd.read_csv(p_test)
        else:
            # Create synthetic initial split for testing if dataset files not yet downloaded
            logger.info("Local processed dataset not found; synthesizing bootstrap dataset...")
            sample_records = [
                ("You are so pathetic and ugly, nobody likes you. Delete your account.", 1),
                ("Keep talking and you will seriously regret showing your face, loser.", 1),
                ("Nobody wants you here, go kill yourself freak.", 1),
                ("You are an idiot and your family hates you.", 1),
                ("Shut up you worthless scum, you are pure trash.", 1),
                ("Disgusting retard, nobody asked for your stupid opinion.", 1),
                ("Go die in a ditch, literally everyone despises you.", 1),
                ("Look at how fat and ugly you are, what a joke of a person.", 1),
                ("Had an amazing time collaborating with our study group today! Grateful.", 0),
                ("I respectfully disagree with the points made in your review.", 0),
                ("Does anyone know what time the library closes on Sundays?", 0),
                ("Happy birthday! Wishing you a fantastic year filled with joy.", 0),
                ("Great job on the presentation today, explained concepts clearly.", 0),
                ("Could you please share the link to the open source repository?", 0),
                ("Thanks for helping me fix that bug in my code, really appreciate it!", 0),
                ("The weather today is so nice, going out for a walk in the park.", 0),
            ] * 20  # Repeat to allow valid batching
            df_full = pd.DataFrame(sample_records, columns=["text", "label"])
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(df_full, test_size=0.3, random_state=seed, stratify=df_full["label"])
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])

    # Optional subsampling for quick debugging / dry run
    if sample_size is not None and sample_size > 0:
        logger.info(f"Subsampling dataset to {sample_size} records per split for fast verification.")
        train_df = train_df.head(sample_size)
        val_df = val_df.head(max(sample_size // 4, 4))
        test_df = test_df.head(max(sample_size // 4, 4))

    # Identify text and label columns
    def _extract_cols(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        txt_col = "text" if "text" in df.columns else ("cleaned_text" if "cleaned_text" in df.columns else df.columns[0])
        lbl_col = "label" if "label" in df.columns else ("Label" if "Label" in df.columns else df.columns[1])
        texts = [preprocess_text(str(x)) for x in df[txt_col].tolist()]
        labels = [int(x) for x in df[lbl_col].tolist()]
        return texts, labels

    train_texts, train_labels = _extract_cols(train_df)
    val_texts, val_labels = _extract_cols(val_df)
    test_texts, test_labels = _extract_cols(test_df)

    logger.info(
        f"Data prepared: Train={len(train_texts)}, Validation={len(val_texts)}, Test={len(test_texts)}"
    )
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


# ==============================================================================
# Main Fine-Tuning Pipeline
# ==============================================================================
def train_distilbert_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Executes the end-to-end DistilBERT training and evaluation pipeline:
    1. Environment & Hardware Detection
    2. Data loading & text preprocessing
    3. Tokenizer and Model loading
    4. Custom Dataset preparation
    5. TrainingArguments & Trainer configuration
    6. Training & Validation execution
    7. Best Model and Tokenizer serialization to models/distilbert/
    8. Label mapping export to models/distilbert/label_map.json
    9. Final evaluation on held-out test set
    10. Report serialization to reports/distilbert_results.json & Confusion Matrix figure
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("CyberShield: DistilBERT Cyberbullying Fine-Tuning Pipeline")
    logger.info("=" * 70)

    # 1. Hardware Detection
    if torch is None or AutoModelForSequenceClassification is None:
        raise RuntimeError(
            "PyTorch and Hugging Face Transformers must be installed to run this script. "
            "Please run: pip install torch transformers datasets scikit-learn"
        )

    # Auto-detect GPU vs CPU
    is_cuda = torch.cuda.is_available() and not getattr(args, "no_cuda", False)
    if is_cuda:
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Hardware Acceleration: GPU Detected [{device_name}]")
        use_fp16 = True
    else:
        logger.info("Hardware Acceleration: GPU not available or disabled. Gracefully using CPU.")
        use_fp16 = False

    # Seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")

    # Output directories
    output_dir = Path(args.output_dir) if args.output_dir else DISTILBERT_DIR
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = REPORTS_DIR
    figures_dir = FIGURES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading & Preprocessing
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = get_training_splits(
        sample_size=args.sample_size,
        seed=args.seed,
    )

    # 3. Model & Tokenizer Initialization
    model_identifier = "distilbert/distilbert-base-uncased"
    logger.info(f"Loading tokenizer: {model_identifier}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    except Exception as exc:
        logger.warning(f"Could not load '{model_identifier}': {exc}. Trying 'distilbert-base-uncased'...")
        model_identifier = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    logger.info(f"Loading pretrained model: {model_identifier} (num_labels=2)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_identifier,
        num_labels=2,
        id2label=CANONICAL_ID_TO_LABEL,
        label2id=CANONICAL_LABEL_TO_ID,
    )

    # 4. Tokenization & Custom PyTorch Dataset
    logger.info(
        f"Tokenizing datasets with max_length={args.max_length}, truncation=True, padding='max_length'..."
    )
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )

    train_dataset = CyberbullyingDataset(train_encodings, train_labels)
    val_dataset = CyberbullyingDataset(val_encodings, val_labels)
    test_dataset = CyberbullyingDataset(test_encodings, test_labels)

    # 5. Training Arguments Configuration
    import inspect
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    training_kwargs: Dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "learning_rate": float(args.lr),
        "num_train_epochs": int(args.epochs),
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(args.eval_batch_size),
        "weight_decay": float(args.weight_decay),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "seed": int(args.seed),
        "logging_steps": 50,
        "save_total_limit": 2,
        "report_to": "none",
    }

    # Dynamic parameter compatibility across Hugging Face Transformers versions
    if "eval_strategy" in sig_params:
        training_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in sig_params:
        training_kwargs["evaluation_strategy"] = "epoch"

    if "use_cpu" in sig_params:
        training_kwargs["use_cpu"] = not is_cuda
    elif "no_cuda" in sig_params:
        training_kwargs["no_cuda"] = not is_cuda

    if "fp16" in sig_params:
        training_kwargs["fp16"] = use_fp16

    # Only pass parameters supported by this TrainingArguments version
    filtered_kwargs = {k: v for k, v in training_kwargs.items() if k in sig_params}
    training_args = TrainingArguments(**filtered_kwargs)

    # 6. Initialize Hugging Face Trainer
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
    }
    trainer_sig = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # 7. Train & Evaluate on Validation Set
    logger.info("Starting DistilBERT fine-tuning...")
    train_result = trainer.train()
    logger.info(f"Training completed in {train_result.metrics.get('train_runtime', 0):.2f} seconds.")

    logger.info("Running validation set evaluation on best checkpoint...")
    val_metrics = trainer.evaluate()
    logger.info(f"Validation Metrics: {json.dumps(val_metrics, indent=2)}")

    # 8. Save Best Model and Tokenizer to models/distilbert/
    logger.info(f"Saving fine-tuned model and tokenizer to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 9. Save label mapping as label_map.json in models/distilbert/
    label_map_file = output_dir / "label_map.json"
    with open(label_map_file, "w", encoding="utf-8") as f:
        json.dump(CANONICAL_LABEL_MAP, f, indent=2)
    logger.info(f"Label map successfully saved to: {label_map_file}")

    # 10. Evaluate on Held-Out Test Set
    logger.info("Evaluating fine-tuned model on held-out test set...")
    test_prediction_output = trainer.predict(test_dataset)
    logits = test_prediction_output.predictions

    # Softmax conversion to probabilities
    logits_tensor = torch.tensor(logits) if torch is not None else logits
    probs = torch.softmax(logits_tensor, dim=-1).cpu().numpy()
    preds = np.argmax(logits, axis=-1)
    y_test_true = np.array(test_labels)

    # Results & figures target paths
    results_json_path = reports_dir / "distilbert_results.json"
    confusion_matrix_path = figures_dir / "distilbert_confusion_matrix.png"

    test_metrics = _internal_evaluate_test_set(
        y_true=y_test_true,
        y_pred=preds,
        y_probs=probs,
        results_path=results_json_path,
        plot_path=confusion_matrix_path,
    )

    # Append runtime and hyperparameter metadata to report
    elapsed_time = round(time.time() - start_time, 2)
    test_metrics["training_parameters"] = {
        "model_name": model_identifier,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": "cuda" if is_cuda else "cpu",
        "fp16": use_fp16,
        "total_runtime_seconds": elapsed_time,
    }
    test_metrics["validation_metrics"] = val_metrics
    test_metrics["test_sample_count"] = len(test_labels)

    # Resave enriched results
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Final results enriched and serialized to: {results_json_path}")

    # Also mirror confusion matrix directly in reports/ for convenience if needed
    try:
        import shutil
        mirror_cm_path = reports_dir / "distilbert_confusion_matrix.png"
        if confusion_matrix_path.is_file():
            shutil.copyfile(confusion_matrix_path, mirror_cm_path)
    except Exception:
        pass

    logger.info("=" * 70)
    logger.info("DistilBERT Fine-Tuning Pipeline Successfully Finished!")
    logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Test F1-Score: {test_metrics.get('f1', 0):.4f}")
    logger.info(f"Test Macro-F1: {test_metrics.get('macro_f1', 0):.4f}")
    logger.info(f"Artifacts saved in: {output_dir}")
    logger.info(f"Report saved at:    {results_json_path}")
    logger.info(f"Confusion Matrix:   {confusion_matrix_path}")
    logger.info("=" * 70)

    return test_metrics


# ==============================================================================
# Command-Line Argument Parser
# ==============================================================================
def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for the DistilBERT fine-tuning script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for Cyberbullying Detection from Social Media Texts."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Per-device training batch size (default: 16)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Per-device evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=2e-5,
        dest="lr",
        help="Initial learning rate for AdamW optimizer (default: 2e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data shuffling and initialization (default: 42)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization (default: 0.01)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token sequence length (default: 128)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional sample size limit for quick dry-run testing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the fine-tuned model and tokenizer (default: models/distilbert)",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA GPU acceleration and force CPU execution",
    )
    return parser.parse_args()


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    cli_args = parse_arguments()
    train_distilbert_pipeline(cli_args)
