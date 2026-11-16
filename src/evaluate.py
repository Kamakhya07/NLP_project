"""Cyberbullying Detection from Social Media - Model Evaluation Module.

This module provides comprehensive evaluation metrics, diagnostics, and visualization:
1. `evaluate_model`: Computes accuracy, precision, recall, f1, macro-f1, ROC-AUC, and log-loss.
2. `print_classification_report`: Formatted classification report using scikit-learn.
3. `plot_confusion_matrix`: Seaborn/Matplotlib heatmap saved to `reports/`.
4. `compare_models`: Generates comparison table (CSV) and grouped bar chart (PNG) saved to `reports/`.
5. `save_evaluation_results` / `load_evaluation_results`: JSON persistence in `reports/`.

Location: D:\\TY\\NLP-project
Standard Label Mapping:
    0: 'Non-Bullying'
    1: 'Bullying'
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ==============================================================================
# Path & Label Constants
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_EVALUATION_JSON = REPORTS_DIR / "evaluation_results.json"
DEFAULT_CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
DEFAULT_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
DEFAULT_COMPARISON_CHART = REPORTS_DIR / "model_comparison.png"

CLASS_NAMES = ["Non-Bullying", "Bullying"]
LABEL_MAP = {0: "Non-Bullying", 1: "Bullying"}


# ==============================================================================
# Evaluation Metrics Functions
# ==============================================================================
def evaluate_model(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    y_proba: Optional[Union[Sequence[float], np.ndarray, Sequence[Sequence[float]]]] = None,
    model_name: str = "",
) -> Dict[str, Any]:
    """Evaluate binary classification performance metrics.

    Calculates:
    - accuracy: Overall accuracy score
    - precision: Positive class ('Bullying' / 1) precision
    - recall: Positive class ('Bullying' / 1) recall
    - f1: Positive class ('Bullying' / 1) F1 score
    - macro_f1: Macro-averaged F1 score across both classes (0 and 1)
    - roc_auc: Area Under the ROC Curve (if y_proba is provided)
    - log_loss: Cross-entropy loss (if y_proba is provided)

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Optional predicted probabilities for positive class or (N, 2) array.
        model_name: Optional descriptive model name.

    Returns:
        Dictionary containing metric scores rounded to 4 decimal places.

    Raises:
        ValueError: If y_true and y_pred have mismatched lengths or are empty.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    if len(y_t) == 0:
        raise ValueError("Cannot evaluate empty ground truth labels (y_true is empty).")
    if len(y_t) != len(y_p):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_t)} samples but y_pred has {len(y_p)} samples."
        )

    # Core required metrics
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    rec = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Per-class breakdowns
    p_per, r_per, f1_per, support = precision_recall_fscore_support(
        y_t, y_p, labels=[0, 1], zero_division=0
    )

    results: Dict[str, Any] = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": int(len(y_t)),
        "class_metrics": {
            "Non-Bullying (0)": {
                "precision": round(float(p_per[0]), 4),
                "recall": round(float(r_per[0]), 4),
                "f1": round(float(f1_per[0]), 4),
                "support": int(support[0]),
            },
            "Bullying (1)": {
                "precision": round(float(p_per[1]), 4),
                "recall": round(float(r_per[1]), 4),
                "f1": round(float(f1_per[1]), 4),
                "support": int(support[1]),
            },
        },
    }

    # Optional probability-based metrics (ROC-AUC & Log-Loss)
    if y_proba is not None:
        try:
            probs = np.asarray(y_proba, dtype=float)
            # Handle 2D probability matrix: extract column for positive class (1)
            if probs.ndim == 2:
                if probs.shape[1] == 2:
                    pos_probs = probs[:, 1]
                    prob_matrix = probs
                elif probs.shape[1] == 1:
                    pos_probs = probs.ravel()
                    prob_matrix = np.column_stack([1.0 - pos_probs, pos_probs])
                else:
                    pos_probs = probs.ravel()
                    prob_matrix = None
            else:
                pos_probs = probs.ravel()
                prob_matrix = np.column_stack([1.0 - pos_probs, pos_probs])

            # ROC-AUC requires at least two classes present in y_true
            if len(np.unique(y_t)) > 1:
                results["roc_auc"] = round(float(roc_auc_score(y_t, pos_probs)), 4)
            else:
                results["roc_auc"] = None

            # Binary Log-Loss (cross-entropy)
            if prob_matrix is not None:
                # Clip probabilities to prevent log(0)
                clipped_probs = np.clip(prob_matrix, 1e-15, 1.0 - 1e-15)
                results["log_loss"] = round(float(log_loss(y_t, clipped_probs)), 4)
        except Exception as exc:
            logger.debug(f"Could not compute probability-based metrics: {exc}")
            results["roc_auc"] = None
            results["log_loss"] = None

    logger.info(
        f"Evaluated [{model_name or 'Model'}]: "
        f"Acc={results['accuracy']:.4f}, Prec={results['precision']:.4f}, "
        f"Rec={results['recall']:.4f}, F1={results['f1']:.4f}, Macro-F1={results['macro_f1']:.4f}"
    )
    return results


def print_classification_report(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    target_names: Optional[List[str]] = None,
    model_name: str = "",
    digits: int = 4,
) -> str:
    """Print and return a formatted scikit-learn classification report.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        target_names: Class display names (default: ['Non-Bullying', 'Bullying']).
        model_name: Optional model header title.
        digits: Floating point decimal places.

    Returns:
        String representation of the classification report.
    """
    if target_names is None:
        target_names = CLASS_NAMES

    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    header = f"=== Classification Report: {model_name or 'Model'} ==="
    report_text = classification_report(
        y_t,
        y_p,
        target_names=target_names,
        labels=[0, 1],
        digits=digits,
        zero_division=0,
    )

    full_output = f"\n{header}\n{report_text}\n"
    print(full_output)
    return report_text


# ==============================================================================
# Visualization Functions
# ==============================================================================
def plot_confusion_matrix(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
    classes: Optional[List[str]] = None,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = "Blues",
) -> Tuple[np.ndarray, Path]:
    """Compute and plot a stylized confusion matrix heatmap using Seaborn, saving to `reports/`.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        classes: List of class display names (default: ['Non-Bullying', 'Bullying']).
        model_name: Model title displayed on plot.
        save_path: Output file path. If None, saves to `reports/{safe_model_name}_confusion_matrix.png`.
        normalize: Whether to normalize matrix by true class support (row-wise).
        figsize: Matplotlib figure dimensions (width, height).
        cmap: Seaborn colormap name (default: 'Blues').

    Returns:
        Tuple of (confusion_matrix_array, saved_file_path).
    """
    if classes is None:
        classes = CLASS_NAMES

    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    # Raw counts confusion matrix
    cm_counts = confusion_matrix(y_t, y_p, labels=[0, 1])

    # Normalized confusion matrix
    if normalize:
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_data = np.divide(cm_counts, row_sums, where=row_sums != 0)
        fmt = ".2%"
    else:
        cm_data = cm_counts
        fmt = "d"

    # Resolve save path
    if save_path is None:
        safe_name = re_safe_name(model_name)
        out_path = REPORTS_DIR / f"{safe_name}_confusion_matrix.png"
    else:
        out_path = Path(save_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=1.0,
        linecolor="#dddddd",
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )

    norm_suffix = " (Normalized)" if normalize else ""
    ax.set_title(f"{model_name}\nConfusion Matrix{norm_suffix}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to: '{out_path}'")
    return cm_counts, out_path


def compare_models(
    results_dict: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]]],
    save_dir: Union[str, Path] = REPORTS_DIR,
    chart_filename: str = "model_comparison.png",
    table_filename: str = "model_comparison.csv",
) -> pd.DataFrame:
    """Create a structured comparison table and grouped bar chart comparing multiple models.

    Saves:
    - Comparison table to `{save_dir}/{table_filename}`
    - Comparison bar chart to `{save_dir}/{chart_filename}`

    Args:
        results_dict: Dictionary mapping model names to evaluation metric dictionaries,
                      or a list of evaluation metric dictionaries.
        save_dir: Directory where reports should be saved (default: `reports/`).
        chart_filename: Name of chart image file (default: 'model_comparison.png').
        table_filename: Name of comparison CSV file (default: 'model_comparison.csv').

    Returns:
        pandas DataFrame containing the comparative performance metrics.

    Raises:
        ValueError: If results_dict is empty.
    """
    if not results_dict:
        raise ValueError("Cannot compare empty model results dictionary.")

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Standardize input into a list of metric dictionaries
    rows: List[Dict[str, Any]] = []
    if isinstance(results_dict, dict):
        for name, metrics in results_dict.items():
            row = dict(metrics)
            if not row.get("model_name"):
                row["model_name"] = str(name)
            rows.append(row)
    elif isinstance(results_dict, list):
        rows = [dict(m) for m in results_dict]

    # Build comparison DataFrame
    table_data = []
    for r in rows:
        table_data.append({
            "Model": r.get("model_name", "Unknown"),
            "Accuracy": float(r.get("accuracy", 0.0)),
            "Precision": float(r.get("precision", 0.0)),
            "Recall": float(r.get("recall", 0.0)),
            "F1-Score": float(r.get("f1", 0.0)),
            "Macro-F1": float(r.get("macro_f1", 0.0)),
            "ROC-AUC": r.get("roc_auc", np.nan),
        })

    df = pd.DataFrame(table_data)

    # Save CSV comparison table
    csv_path = out_dir / table_filename
    df.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info(f"Saved model comparison table to: '{csv_path}'")

    # Generate grouped comparison bar chart
    chart_path = out_dir / chart_filename
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "Macro-F1"]

    # Melt DataFrame for seaborn grouped bar plot
    melted_df = df.melt(
        id_vars="Model",
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = sns.color_palette("Set2", n_colors=len(df["Model"].unique()))

    barplot = sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    # Annotate bars with numeric scores
    for p in barplot.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                xytext=(0, 2),
                textcoords="offset points",
            )

    ax.set_title("Cyberbullying Detection Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Evaluation Metric", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylim(0.0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Model", loc="lower right", framealpha=0.9)
    plt.tight_layout()

    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved model comparison bar chart to: '{chart_path}'")
    return df


# ==============================================================================
# JSON Persistence Functions
# ==============================================================================
def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and non-serializable objects to native Python."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    elif isinstance(obj, Path):
        return str(obj)
    elif pd.isna(obj):
        return None
    return obj


def save_evaluation_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    file_path: Union[str, Path] = DEFAULT_EVALUATION_JSON,
) -> Path:
    """Save evaluation results dictionary or list to a formatted JSON file in `reports/`.

    Args:
        results: Dictionary or list containing evaluation metrics.
        file_path: Output file path (default: `reports/evaluation_results.json`).

    Returns:
        Path to saved JSON file.

    Raises:
        IOError: If writing to the file fails.
    """
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_results = _make_json_serializable(results)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved evaluation results JSON to: '{out_path}' ({out_path.stat().st_size} bytes)")
        return out_path
    except Exception as exc:
        logger.error(f"Failed to write evaluation results JSON to '{out_path}': {exc}")
        raise IOError(f"Error saving evaluation results to {out_path}: {exc}") from exc


def load_evaluation_results(
    file_path: Union[str, Path] = DEFAULT_EVALUATION_JSON,
) -> Dict[str, Any]:
    """Load evaluation results from a JSON file.

    Args:
        file_path: Path to evaluation results JSON file (default: `reports/evaluation_results.json`).

    Returns:
        Loaded dictionary containing evaluation results.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If JSON parsing fails.
    """
    in_path = Path(file_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Evaluation results JSON not found at: '{in_path}'")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded evaluation results from: '{in_path}'")
    return data


def re_safe_name(name: str) -> str:
    """Helper to convert human model name to safe lowercase file identifier."""
    return re_sub(r"[^a-zA-Z0-9_-]", "_", name.lower()).strip("_")


def re_sub(pattern: str, repl: str, string: str) -> str:
    import re
    return re.sub(pattern, repl, string)


# ==============================================================================
# CLI / Testing Entry Point
# ==============================================================================
if __name__ == "__main__":
    print("\n--- Running src/evaluate.py Verification Test ---")

    np.random.seed(42)
    n_test = 500

    # 1. Generate realistic synthetic ground truth and predictions for all 3 project models:
    #    - TF-IDF + LogisticRegression
    #    - TF-IDF + LinearSVM
    #    - Fine-tuned DistilBERT
    y_true = np.random.choice([0, 1], size=n_test, p=[0.60, 0.40])

    # Model 1: Logistic Regression (~87% accuracy)
    lr_flips = np.random.rand(n_test) < 0.13
    y_pred_lr = np.where(lr_flips, 1 - y_true, y_true)
    y_proba_lr = np.clip(
        np.where(y_pred_lr == 1, np.random.uniform(0.65, 0.95, n_test), np.random.uniform(0.05, 0.35, n_test)),
        0.0,
        1.0,
    )

    # Model 2: Linear SVM (~89% accuracy)
    svm_flips = np.random.rand(n_test) < 0.11
    y_pred_svm = np.where(svm_flips, 1 - y_true, y_true)
    # Calibrated probabilities for SVM
    y_proba_svm = np.clip(
        np.where(y_pred_svm == 1, np.random.uniform(0.70, 0.98, n_test), np.random.uniform(0.02, 0.30, n_test)),
        0.0,
        1.0,
    )

    # Model 3: Fine-tuned DistilBERT (~94% accuracy)
    bert_flips = np.random.rand(n_test) < 0.06
    y_pred_bert = np.where(bert_flips, 1 - y_true, y_true)
    y_proba_bert = np.clip(
        np.where(y_pred_bert == 1, np.random.uniform(0.85, 0.99, n_test), np.random.uniform(0.01, 0.15, n_test)),
        0.0,
        1.0,
    )

    print("\n[Step 1: Evaluate individual models]")
    eval_lr = evaluate_model(y_true, y_pred_lr, y_proba_lr, model_name="TF-IDF + LogisticRegression")
    eval_svm = evaluate_model(y_true, y_pred_svm, y_proba_svm, model_name="TF-IDF + LinearSVM")
    eval_bert = evaluate_model(y_true, y_pred_bert, y_proba_bert, model_name="Fine-tuned DistilBERT")

    # 2. Print Classification Report
    print("\n[Step 2: Print classification report]")
    print_classification_report(y_true, y_pred_bert, model_name="Fine-tuned DistilBERT")

    # 3. Plot Confusion Matrices and save to reports/
    print("\n[Step 3: Plot and save confusion matrices to reports/]")
    cm_lr, lr_cm_path = plot_confusion_matrix(
        y_true, y_pred_lr, model_name="TF-IDF + LogisticRegression", save_path=REPORTS_DIR / "lr_confusion_matrix.png"
    )
    cm_svm, svm_cm_path = plot_confusion_matrix(
        y_true, y_pred_svm, model_name="TF-IDF + LinearSVM", save_path=REPORTS_DIR / "svm_confusion_matrix.png"
    )
    cm_bert, bert_cm_path = plot_confusion_matrix(
        y_true, y_pred_bert, model_name="Fine-tuned DistilBERT", save_path=REPORTS_DIR / "distilbert_confusion_matrix.png"
    )
    # Also save primary confusion_matrix.png
    plot_confusion_matrix(
        y_true, y_pred_bert, model_name="Best Model (DistilBERT)", save_path=DEFAULT_CONFUSION_MATRIX_PATH
    )

    # 4. Compare Models and save table + bar chart
    print("\n[Step 4: Compare models and save table + bar chart to reports/]")
    all_results = {
        "TF-IDF + LogisticRegression": eval_lr,
        "TF-IDF + LinearSVM": eval_svm,
        "Fine-tuned DistilBERT": eval_bert,
    }
    comparison_df = compare_models(
        all_results,
        save_dir=REPORTS_DIR,
        chart_filename="model_comparison.png",
        table_filename="model_comparison.csv",
    )
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))

    # 5. Save evaluation results as JSON to reports/
    print("\n[Step 5: Save evaluation results as JSON]")
    json_path = save_evaluation_results(all_results, file_path=DEFAULT_EVALUATION_JSON)
    loaded_json = load_evaluation_results(json_path)
    assert len(loaded_json) == 3, "All 3 models should be present in saved JSON!"
    print(f"[PASS] Successfully verified JSON persistence ({len(loaded_json)} models).")

    # Verify generated files in reports/
    print("\n[Verification of generated reports/ files]")
    for p in [
        DEFAULT_CONFUSION_MATRIX_PATH,
        REPORTS_DIR / "lr_confusion_matrix.png",
        REPORTS_DIR / "svm_confusion_matrix.png",
        REPORTS_DIR / "distilbert_confusion_matrix.png",
        DEFAULT_COMPARISON_CSV,
        DEFAULT_COMPARISON_CHART,
        DEFAULT_EVALUATION_JSON,
    ]:
        assert p.is_file(), f"Expected file {p} does not exist!"
        print(f"  [OK] {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    print("\nsrc/evaluate.py passed all tests successfully!\n")
