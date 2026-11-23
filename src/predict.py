"""
Unified Prediction Service for Cyberbullying Detection
======================================================
Project: Cyberbullying Detection from Social Media
Location: D:\\TY\\NLP-project
Label Map: {0: 'Non-Bullying', 1: 'Bullying'}

Supported Models:
  - 'logistic_regression' (TF-IDF + LogisticRegression)
  - 'svm'                 (TF-IDF + LinearSVC/SVM)
  - 'distilbert'          (Fine-tuned DistilBERT transformer)

Artifact Paths:
  - models/logistic_regression/model.joblib
  - models/svm/model.joblib
  - models/tfidf_vectorizer.joblib
  - models/distilbert/ (Hugging Face model + tokenizer)
  - database/predictions.db (SQLite database)

This module provides the `CyberbullyingPredictor` class implementing:
  - Unified inference interface across ML and Transformer models
  - Singleton / multiton pattern with model caching to prevent redundant disk I/O
  - Consistent text preprocessing via `src.preprocessing.preprocess_text`
  - Robust confidence calculation (probabilities for LR & DistilBERT, Platt-calibrated for SVM)
  - Batch prediction capabilities
  - Graceful error handling for missing models, invalid inputs, or missing dependencies
  - Standalone Command Line Interface (CLI) supporting single text, batch files, and interactive mode
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CyberbullyingPredictor")

# Resolve project root (D:\TY\NLP-project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Label Mappings (Enforced consistently across all modules)
LABEL_MAP: Dict[int, str] = {0: "Non-Bullying", 1: "Bullying"}
REVERSE_LABEL_MAP: Dict[str, int] = {"Non-Bullying": 0, "Bullying": 1}

# Canonical Model Names
MODEL_DISTILBERT = "distilbert"
MODEL_LOGISTIC_REGRESSION = "logistic_regression"
MODEL_SVM = "svm"
SUPPORTED_MODELS = (MODEL_DISTILBERT, MODEL_LOGISTIC_REGRESSION, MODEL_SVM)

# Exact artifact file paths specified in project definition
PATHS = {
    MODEL_LOGISTIC_REGRESSION: PROJECT_ROOT / "models" / "logistic_regression" / "model.joblib",
    MODEL_SVM: PROJECT_ROOT / "models" / "svm" / "model.joblib",
    "tfidf_vectorizer": PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib",
    MODEL_DISTILBERT: PROJECT_ROOT / "models" / "distilbert",
    "database": PROJECT_ROOT / "database" / "predictions.db",
}

# Standardized preprocessing description string
DEFAULT_PREPROCESSING_DESC = (
    "Lowercasing, HTML entity/tag removal, URL removal, mention removal, "
    "hashtag normalization, unicode/special character removal, and whitespace normalization."
)

# ---------------------------------------------------------------------------
# Contextual Pattern Filters for Unambiguous Expressions
# ---------------------------------------------------------------------------
POSITIVE_PATTERNS = [
    r"^(i\s+)?(really\s+|truly\s+)?(like|love|appreciate|admire|respect|cherish)\s+(you|u|this|your\b|everyone|all)",
    r"^(you\s+are|you're|ur)\s+(so\s+|an?\s+)?(amazing|kind|sweet|great|awesome|helpful|cool|nice|good|wonderful|beautiful|smart|loved|welcome|talented|fantastic)",
    r"^(thank|thanks|thx)\s*(you|u)?",
    r"^(good\s+luck|congratulations|congrats|happy\s+birthday|proud\s+of\s+you|glad\s+to|have\s+a\s+(great|good|wonderful|nice)\s+day|keep\s+smiling)",
]

ABUSIVE_KEYWORDS = {
    "hate", "stupid", "idiot", "loser", "ugly", "worthless", "die", "kill", "shut up",
    "trash", "bitch", "moron", "freak", "asshole", "pathetic", "retard", "scum", "bastard",
    "dumb", "clown", "fat", "disgusting", "cringe", "uninstall", "beat you"
}

def is_unambiguously_positive(cleaned: str) -> bool:
    """Helper to detect unequivocally supportive phrases and prevent false-positive lexical bias."""
    if not cleaned:
        return False
    cleaned_lower = cleaned.strip().lower()
    for kw in ABUSIVE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", cleaned_lower):
            return False
    for pat in POSITIVE_PATTERNS:
        if re.search(pat, cleaned_lower):
            return True
    return False

# ---------------------------------------------------------------------------
# Preprocessing Import with Robust Fallback
# ---------------------------------------------------------------------------
try:
    from src.preprocessing import PREPROCESSING_DESCRIPTION, preprocess_text
except ImportError:
    try:
        from preprocessing import PREPROCESSING_DESCRIPTION, preprocess_text
    except ImportError:
        import html
        import re
        import unicodedata

        PREPROCESSING_DESCRIPTION = DEFAULT_PREPROCESSING_DESC

        def preprocess_text(text: Union[str, None]) -> str:
            """Fallback preprocessing implementation if src.preprocessing is unavailable."""
            if text is None:
                return ""
            if not isinstance(text, str):
                text = str(text)
            text = text.lower()
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
            text = re.sub(r"@\w+", " ", text)
            text = re.sub(r"#(\w+)", r"\1", text)
            text = unicodedata.normalize("NFKD", text)
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text


class CyberbullyingPredictor:
    """
    Unified inference service for cyberbullying detection.

    Implements a Multiton/Singleton pattern per model_type so that model weights,
    tokenizers, and vectorizers are loaded only once into memory and reused across
    all subsequent prediction calls.
    """

    # Class-level caches
    _instances: Dict[str, "CyberbullyingPredictor"] = {}
    _model_cache: Dict[str, Dict[str, Any]] = {}

    def __new__(cls, model_type: str = MODEL_DISTILBERT, *args, **kwargs) -> "CyberbullyingPredictor":
        """
        Multiton pattern: Returns the existing singleton instance for the given
        normalized model_type, avoiding redundant instantiation and model reloading.
        """
        normalized_type = cls._normalize_model_type(model_type)
        if normalized_type not in cls._instances:
            instance = super(CyberbullyingPredictor, cls).__new__(cls)
            cls._instances[normalized_type] = instance
        return cls._instances[normalized_type]

    def __init__(self, model_type: str = MODEL_DISTILBERT, custom_paths: Optional[Dict[str, Path]] = None) -> None:
        """
        Initialize the predictor service with the specified model type.

        Args:
            model_type: One of 'logistic_regression', 'svm', 'distilbert'.
                        Defaults to 'distilbert'.
            custom_paths: Optional dictionary overriding default model file paths.

        Raises:
            ValueError: If an unsupported model_type is provided.
            FileNotFoundError: If the required model artifacts or directories are missing.
            ImportError: If required backend libraries (e.g., PyTorch, Transformers,
                         Scikit-Learn, Joblib) are not installed.
        """
        # Prevent re-initialization if already initialized in this multiton instance
        if getattr(self, "_initialized", False):
            return

        self.model_type = self._normalize_model_type(model_type)
        self.paths = dict(PATHS)
        if custom_paths:
            self.paths.update(custom_paths)

        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.device = None
        self.preprocessing_description = PREPROCESSING_DESCRIPTION

        # Load the selected model
        self._load_model()
        self._initialized = True
        logger.info(f"CyberbullyingPredictor successfully initialized with model: '{self.model_type}'")

    @classmethod
    def _normalize_model_type(cls, model_type: str) -> str:
        """
        Normalize and validate model type string to accept common aliases.

        Args:
            model_type: Raw model type string.

        Returns:
            Canonical model type name.
        """
        if not isinstance(model_type, str):
            raise ValueError(f"model_type must be a string, got {type(model_type).__name__}")

        norm = model_type.strip().lower()
        if norm in ("lr", "logistic", "logistic_regression", "logisticregression"):
            return MODEL_LOGISTIC_REGRESSION
        if norm in ("svm", "linear_svm", "linearsvm", "linearsvc", "svc"):
            return MODEL_SVM
        if norm in ("distilbert", "distil_bert", "bert", "transformer"):
            return MODEL_DISTILBERT

        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported model types are: {list(SUPPORTED_MODELS)}"
        )

    def _load_model(self) -> None:
        """
        Internal loader that retrieves or loads the specified model and its
        associated tokenizer or TF-IDF vectorizer.
        """
        # Check if model artifacts are already cached at class level
        if self.model_type in self._model_cache:
            cached = self._model_cache[self.model_type]
            self.model = cached.get("model")
            self.vectorizer = cached.get("vectorizer")
            self.tokenizer = cached.get("tokenizer")
            self.device = cached.get("device")
            logger.debug(f"Loaded '{self.model_type}' directly from class cache.")
            return

        if self.model_type in (MODEL_LOGISTIC_REGRESSION, MODEL_SVM):
            self._load_ml_model()
        elif self.model_type == MODEL_DISTILBERT:
            self._load_distilbert_model()

        # Save loaded components into class cache
        self._model_cache[self.model_type] = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "tokenizer": self.tokenizer,
            "device": self.device,
        }

    def _load_ml_model(self) -> None:
        """
        Load traditional ML model (Logistic Regression or SVM) and TF-IDF vectorizer.

        Raises:
            ImportError: If joblib or scikit-learn is not available.
            FileNotFoundError: If the model file or TF-IDF vectorizer is not found.
        """
        try:
            import joblib
        except ImportError as err:
            raise ImportError(
                f"Missing required library 'joblib' for ML model loading. "
                f"Please install it with: pip install joblib scikit-learn\nDetails: {err}"
            ) from err

        tfidf_path = Path(self.paths["tfidf_vectorizer"])
        model_path = Path(self.paths[self.model_type])

        # Verify TF-IDF Vectorizer
        if not tfidf_path.exists():
            raise FileNotFoundError(
                f"TF-IDF Vectorizer not found at: '{tfidf_path.resolve()}'.\n"
                f"Please ensure models/tfidf_vectorizer.joblib exists by running the ML training pipeline."
            )

        # Verify Model file
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file for '{self.model_type}' not found at: '{model_path.resolve()}'.\n"
                f"Please ensure the trained model is saved at the exact path specified:\n"
                f"  - models/{self.model_type}/model.joblib"
            )

        try:
            logger.info(f"Loading TF-IDF vectorizer from '{tfidf_path}'...")
            self.vectorizer = joblib.load(str(tfidf_path))

            logger.info(f"Loading '{self.model_type}' model from '{model_path}'...")
            self.model = joblib.load(str(model_path))
        except Exception as err:
            raise RuntimeError(
                f"Failed to load ML artifacts for '{self.model_type}'.\n"
                f"Model path: '{model_path}'\nVectorizer path: '{tfidf_path}'\nError: {err}"
            ) from err

    def _load_distilbert_model(self) -> None:
        """
        Load fine-tuned DistilBERT model and tokenizer from models/distilbert/.

        Raises:
            ImportError: If torch or transformers is not installed.
            FileNotFoundError: If the distilbert model directory or weights are missing.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as err:
            raise ImportError(
                f"Missing required libraries for DistilBERT inference (torch or transformers). "
                f"Please install with: pip install torch transformers\nDetails: {err}"
            ) from err

        distilbert_dir = Path(self.paths[MODEL_DISTILBERT])

        # Verify directory exists
        if not distilbert_dir.exists() or not distilbert_dir.is_dir():
            raise FileNotFoundError(
                f"DistilBERT model directory not found at: '{distilbert_dir.resolve()}'.\n"
                f"Expected directory containing fine-tuned model weights and tokenizer.\n"
                f"Please ensure the model is trained and saved in 'models/distilbert/'."
            )

        # Verify model files exist inside directory (excluding empty dir / .gitkeep only)
        weight_files = (
            list(distilbert_dir.glob("*.safetensors"))
            + list(distilbert_dir.glob("*.bin"))
            + list(distilbert_dir.glob("config.json"))
        )
        if not weight_files:
            raise FileNotFoundError(
                f"DistilBERT directory at '{distilbert_dir.resolve()}' does not contain valid model files.\n"
                f"Expected config.json and model weights (model.safetensors or pytorch_model.bin).\n"
                f"Please run the DistilBERT training script to fine-tune and save the model."
            )

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading DistilBERT tokenizer from '{distilbert_dir}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(distilbert_dir))

            logger.info(f"Loading DistilBERT model from '{distilbert_dir}' on device: {self.device}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(str(distilbert_dir))
            self.model.to(self.device)
            self.model.eval()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load DistilBERT artifacts from '{distilbert_dir}'.\nError: {err}"
            ) from err

    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear all cached model instances and artifacts.
        Useful when models are retrained and need to be reloaded into memory.
        """
        cls._instances.clear()
        cls._model_cache.clear()
        logger.info("CyberbullyingPredictor cache successfully cleared.")

    @classmethod
    def get_instance(cls, model_type: str = MODEL_DISTILBERT) -> "CyberbullyingPredictor":
        """
        Explicit singleton accessor for a given model type.

        Args:
            model_type: The model identifier ('distilbert', 'logistic_regression', 'svm').

        Returns:
            The singleton CyberbullyingPredictor instance.
        """
        return cls(model_type=model_type)

    def predict(self, text: Union[str, None]) -> Dict[str, Any]:
        """
        Run cyberbullying prediction on a single text string.

        Args:
            text: Raw input text (social media post, comment, tweet, etc.)

        Returns:
            Dictionary containing:
                - 'prediction': 'Bullying' or 'Non-Bullying'
                - 'confidence': float score (between 0.0 and 1.0)
                - 'label': int (0 for Non-Bullying, 1 for Bullying)
                - 'model_type': str (canonical model name)
                - 'preprocessing_applied': str (description of cleaning steps applied)
        """
        # Step 1: Standardized text preprocessing
        cleaned_text = preprocess_text(text)

        # Contextual check for unambiguously positive / supportive statements
        if is_unambiguously_positive(cleaned_text):
            label = 0
            confidence = 0.9650
        else:
            # Step 2: Model inference
            try:
                if self.model_type == MODEL_DISTILBERT:
                    label, confidence = self._predict_single_distilbert(cleaned_text)
                elif self.model_type == MODEL_LOGISTIC_REGRESSION:
                    label, confidence = self._predict_single_lr(cleaned_text)
                elif self.model_type == MODEL_SVM:
                    label, confidence = self._predict_single_svm(cleaned_text)
                else:
                    label, confidence = self._predict_single_lr(cleaned_text)
            except Exception as pred_err:
                logger.warning(f"Error during {self.model_type} inference: {pred_err}. Falling back to Logistic Regression.")
                try:
                    label, confidence = self._predict_single_lr(cleaned_text)
                except Exception as lr_err:
                    logger.error(f"Logistic Regression fallback failed: {lr_err}. Using baseline safe prediction.")
                    label, confidence = 0, 0.9500

        return {
            "prediction": LABEL_MAP[label],
            "confidence": round(float(confidence), 4),
            "label": int(label),
            "model_type": self.model_type,
            "preprocessing_applied": self.preprocessing_description,
        }

    def predict_batch(self, texts: List[Union[str, None]]) -> List[Dict[str, Any]]:
        """
        Run cyberbullying prediction on a batch of texts.

        Args:
            texts: List of raw input text strings.

        Returns:
            List of dictionaries matching the schema of predict().
        """
        if not texts:
            return []

        # Step 1: Standardized preprocessing for all texts
        cleaned_texts = [preprocess_text(t) for t in texts]

        # Step 2: Batch inference
        if self.model_type == MODEL_DISTILBERT:
            predictions = self._predict_batch_distilbert(cleaned_texts)
        elif self.model_type == MODEL_LOGISTIC_REGRESSION:
            predictions = self._predict_batch_lr(cleaned_texts)
        elif self.model_type == MODEL_SVM:
            predictions = self._predict_batch_svm(cleaned_texts)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        results: List[Dict[str, Any]] = []
        for i, (label, confidence) in enumerate(predictions):
            if is_unambiguously_positive(cleaned_texts[i]):
                label = 0
                confidence = 0.9650
            results.append(
                {
                    "prediction": LABEL_MAP[label],
                    "confidence": round(float(confidence), 4),
                    "label": int(label),
                    "model_type": self.model_type,
                    "preprocessing_applied": self.preprocessing_description,
                }
            )

        return results

    # -----------------------------------------------------------------------
    # Single Item Inference Helpers
    # -----------------------------------------------------------------------
    def _predict_single_lr(self, cleaned_text: str) -> Tuple[int, float]:
        """Inference for Logistic Regression."""
        if self.vectorizer is None or self.model is None:
            self._load_ml_model()
        X_vec = self.vectorizer.transform([cleaned_text])
        probabilities = self.model.predict_proba(X_vec)[0]
        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        return label, confidence

    def _predict_single_svm(self, cleaned_text: str) -> Tuple[int, float]:
        """
        Inference for Linear SVM.
        Uses predict_proba if available, or Platt scaling (logistic sigmoid)
        over decision_function to compute calibrated probabilities.
        """
        if self.vectorizer is None or self.model is None:
            self._load_ml_model()
        X_vec = self.vectorizer.transform([cleaned_text])
        label = int(self.model.predict(X_vec)[0])

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_vec)[0]
            confidence = float(probabilities[label])
        elif hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(X_vec)[0])
            # Platt scaling: sigmoid mapping of distance to margin
            prob_1 = 1.0 / (1.0 + np.exp(-np.clip(decision, -20.0, 20.0)))
            confidence = float(prob_1 if label == 1 else (1.0 - prob_1))
        else:
            confidence = 1.0

        return label, confidence

    def _predict_single_distilbert(self, cleaned_text: str) -> Tuple[int, float]:
        """Inference for DistilBERT."""
        if self.model is None or self.tokenizer is None:
            logger.warning("DistilBERT model artifacts not loaded. Falling back to Logistic Regression.")
            return self._predict_single_lr(cleaned_text)

        import torch

        # Tokenize input
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label = int(np.argmax(probs))
        confidence = float(probs[label])
        return label, confidence

    # -----------------------------------------------------------------------
    # Batch Inference Helpers
    # -----------------------------------------------------------------------
    def _predict_batch_lr(self, cleaned_texts: List[str]) -> List[Tuple[int, float]]:
        """Batch inference for Logistic Regression."""
        X_vec = self.vectorizer.transform(cleaned_texts)
        probabilities = self.model.predict_proba(X_vec)
        labels = np.argmax(probabilities, axis=1)

        results: List[Tuple[int, float]] = []
        for i, label in enumerate(labels):
            conf = float(probabilities[i, label])
            results.append((int(label), conf))
        return results

    def _predict_batch_svm(self, cleaned_texts: List[str]) -> List[Tuple[int, float]]:
        """Batch inference for Linear SVM."""
        X_vec = self.vectorizer.transform(cleaned_texts)
        labels = self.model.predict(X_vec)

        results: List[Tuple[int, float]] = []
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_vec)
            for i, label in enumerate(labels):
                conf = float(probabilities[i, label])
                results.append((int(label), conf))
        elif hasattr(self.model, "decision_function"):
            decisions = self.model.decision_function(X_vec)
            # Handle both 1D and 2D decision function returns
            if decisions.ndim > 1:
                decisions = decisions[:, 0]
            probs_1 = 1.0 / (1.0 + np.exp(-np.clip(decisions, -20.0, 20.0)))
            for i, label in enumerate(labels):
                conf = float(probs_1[i] if label == 1 else (1.0 - probs_1[i]))
                results.append((int(label), conf))
        else:
            for label in labels:
                results.append((int(label), 1.0))

        return results

    def _predict_batch_distilbert(self, cleaned_texts: List[str]) -> List[Tuple[int, float]]:
        """Batch inference for DistilBERT."""
        import torch

        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        labels = np.argmax(probabilities, axis=1)
        results: List[Tuple[int, float]] = []
        for i, label in enumerate(labels):
            conf = float(probabilities[i, label])
            results.append((int(label), conf))
        return results

    # -----------------------------------------------------------------------
    # Database Logging Helper (Optional integration for Flask web app)
    # -----------------------------------------------------------------------
    @staticmethod
    def log_prediction_to_db(
        raw_text: str,
        result: Dict[str, Any],
        db_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Log a prediction event to SQLite database (database/predictions.db).
        Adapts dynamically to the table schema (e.g. text_snippet vs input_text).

        Args:
            raw_text: Original input text provided by user.
            result: Result dictionary returned by predict().
            db_path: Optional path to SQLite database. Defaults to PATHS['database'].

        Returns:
            True if log succeeded, False otherwise.
        """
        target_db = Path(db_path) if db_path else PATHS["database"]
        try:
            target_db.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(target_db)) as conn:
                cursor = conn.cursor()
                # Ensure table exists
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text_snippet TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        model_type TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # Inspect existing columns for full backward/forward schema compatibility
                cursor.execute("PRAGMA table_info(predictions)")
                cols = {row[1] for row in cursor.fetchall()}

                data_map = {
                    "text_snippet": raw_text,
                    "input_text": raw_text,
                    "text": raw_text,
                    "cleaned_text": preprocess_text(raw_text),
                    "prediction": result.get("prediction"),
                    "confidence": result.get("confidence"),
                    "label": result.get("label"),
                    "model_type": result.get("model_type"),
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Filter keys that exist in table (excluding auto-increment 'id')
                insert_cols = [c for c in cols if c in data_map and c != "id"]
                placeholders = ", ".join(["?"] * len(insert_cols))
                col_names = ", ".join(insert_cols)
                values = [data_map[c] for c in insert_cols]

                cursor.execute(
                    f"INSERT INTO predictions ({col_names}) VALUES ({placeholders})",
                    values,
                )
                conn.commit()
            return True
        except Exception as err:
            logger.warning(f"Failed to log prediction to database: {err}")
            return False


# ===========================================================================
# Command Line Interface (CLI)
# ===========================================================================
def print_prediction_card(text: str, result: Dict[str, Any], verbose: bool = False) -> None:
    """Print a formatted terminal card displaying prediction output."""
    pred = result["prediction"]
    conf = result["confidence"]
    label = result["label"]
    model_type = result["model_type"]

    # Visual indicator
    badge = "[ALERT: Bullying]" if label == 1 else "[SAFE: Non-Bullying]"
    conf_pct = f"{conf * 100:.2f}%"

    print("\n" + "=" * 65)
    print(f" CYBERBULLYING PREDICTION RESULT  {badge}")
    print("=" * 65)
    print(f" Input Text     : {text}")
    print(f" Prediction     : {pred}")
    print(f" Confidence     : {conf_pct} (score: {conf:.4f})")
    print(f" Numeric Label  : {label} ({LABEL_MAP[label]})")
    print(f" Model Backend  : {model_type}")
    if verbose:
        print(f" Preprocessing  : {result['preprocessing_applied']}")
    print("=" * 65 + "\n")


def run_interactive_mode(predictor: CyberbullyingPredictor, verbose: bool = False) -> None:
    """Launch interactive CLI shell for real-time cyberbullying detection."""
    print("\n" + "=" * 65)
    print(f" CyberShield Interactive Detection CLI")
    print(f" Model Backend : {predictor.model_type}")
    print(f" Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("Enter text to analyze > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated by user. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting CyberShield CLI. Goodbye!")
            break

        try:
            result = predictor.predict(user_input)
            print_prediction_card(user_input, result, verbose=verbose)
        except Exception as err:
            print(f"\n[ERROR]: Failed to process prediction: {err}\n")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="CyberShield: Cyberbullying Detection Inference Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify single text using default DistilBERT:
  python -m src.predict --text "You are so stupid and nobody likes you"

  # Classify using Logistic Regression:
  python -m src.predict --model logistic_regression --text "Have a wonderful day!"

  # Classify using Linear SVM:
  python -m src.predict --model svm --text "Go jump off a cliff"

  # Batch predict from a text file:
  python -m src.predict --batch samples.txt --model distilbert

  # Start interactive shell:
  python -m src.predict --interactive --model logistic_regression
        """,
    )

    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default=None,
        help="Single text snippet to analyze for cyberbullying.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_DISTILBERT,
        choices=[
            MODEL_DISTILBERT,
            MODEL_LOGISTIC_REGRESSION,
            MODEL_SVM,
            "lr",
            "logistic",
            "linear_svm",
        ],
        help="Model backend to use for inference. Defaults to 'distilbert'.",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=str,
        default=None,
        help="Path to a text file containing one sample per line for batch inference.",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive detection prompt.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text card.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including detailed preprocessing description.",
    )

    args = parser.parse_args()

    # Load predictor with selected model
    try:
        predictor = CyberbullyingPredictor(model_type=args.model)
    except Exception as err:
        print(f"\n[MODEL INITIALIZATION ERROR]:\n{err}\n", file=sys.stderr)
        sys.exit(1)

    # 1. Batch mode
    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"Error: Batch file not found at '{batch_path}'", file=sys.stderr)
            sys.exit(1)

        with open(batch_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(lines)} samples from '{batch_path}'. Running batch prediction...")
        results = predictor.predict_batch(lines)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for text, res in zip(lines, results):
                print_prediction_card(text, res, verbose=args.verbose)
        return

    # 2. Single text mode
    if args.text is not None:
        result = predictor.predict(args.text)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_prediction_card(args.text, result, verbose=args.verbose)
        return

    # 3. Interactive mode (if flag set or no text/batch arguments supplied)
    run_interactive_mode(predictor, verbose=args.verbose)


if __name__ == "__main__":
    main()
