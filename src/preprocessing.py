"""Cyberbullying Detection from Social Media - Preprocessing Module.

This module provides:
1. `preprocess_text`: Conservative text cleaning that lowercases text, removes URLs
   and @mentions, and normalizes whitespace while strictly preserving emojis and punctuation.
2. `preprocess_dataframe`: Batch preprocessing applied directly to pandas DataFrames.
3. TF-IDF feature extraction pipeline:
   - `build_tfidf_vectorizer`: Instantiates TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000).
   - `fit_tfidf`: Fits TF-IDF on training texts and saves to `models/tfidf_vectorizer.joblib`.
   - `transform_tfidf`: Transforms texts using a fitted or loaded vectorizer.
   - `save_vectorizer` / `load_vectorizer`: Joblib persistence utilities.

Location: D:\\TY\\NLP-project
Path: models/tfidf_vectorizer.joblib
"""

import html
import logging
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocessing")

# ==============================================================================
# Path Constants
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_VECTORIZER_PATH = DEFAULT_MODELS_DIR / "tfidf_vectorizer.joblib"

# Regular expressions compiled for performance
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_REGEX = re.compile(r"@\w+", flags=re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"\s+")


# ==============================================================================
# Text Preprocessing
# ==============================================================================
def preprocess_text(text: Any) -> str:
    """Conservatively cleans raw social media text for cyberbullying classification.

    Preserves critical cyberbullying indicators:
    - Retains ALL emojis (e.g., 😡, 🤬, 💀, 🤡, 🖕) which carry strong sentiment and harassment signals.
    - Retains ALL punctuation marks (e.g., '!', '?', '***') which express aggression and emphasis.

    Cleaning steps:
    1. Null/type safety: safely converts non-string or NaN/None inputs to empty strings.
    2. HTML unescaping: converts entities like &amp;, &lt;, &gt; to standard characters.
    3. Unicode normalization: applies NFKC normalization while preserving emoji glyphs.
    4. URL removal: strips web addresses (http://, https://, www.*).
    5. Mention removal: strips social media handles (@username).
    6. Lowercasing: standardizes text casing.
    7. Whitespace normalization: collapses multiple spaces, tabs, and newlines into a single space and strips ends.

    Args:
        text: Input text, string, or object.

    Returns:
        Cleaned, normalized string.
    """
    if text is None or pd.isna(text):
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)

    # 2. Unicode normalization (NFKC decomposes compatibility chars while preserving emojis)
    text = unicodedata.normalize("NFKC", text)

    # 3. Remove URLs
    text = URL_REGEX.sub(" ", text)

    # 4. Remove user mentions (@username)
    text = MENTION_REGEX.sub(" ", text)

    # 5. Convert to lowercase
    text = text.lower()

    # 6. Normalize whitespace (collapses runs of whitespace and strips leading/trailing space)
    text = WHITESPACE_REGEX.sub(" ", text).strip()

    return text


# Alias for seamless cross-module compatibility
clean_text = preprocess_text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    target_col: Optional[str] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply conservative text preprocessing to a pandas DataFrame column.

    Args:
        df: Input pandas DataFrame.
        text_col: Name of column containing raw text (default: 'text').
        target_col: Optional destination column name. If None, overwrites `text_col`.
        inplace: Whether to modify DataFrame in-place (default: False).

    Returns:
        DataFrame with preprocessed text column.

    Raises:
        KeyError: If text_col is not found in DataFrame.
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Specified text column '{text_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    out_df = df if inplace else df.copy()
    dest_col = target_col if target_col is not None else text_col

    out_df[dest_col] = out_df[text_col].apply(preprocess_text)
    logger.debug(f"Preprocessed {len(out_df):,} rows in column '{dest_col}'")
    return out_df


# ==============================================================================
# TF-IDF Vectorization Pipeline
# ==============================================================================
def build_tfidf_vectorizer(
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_features: int = 50000,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Build and configure a scikit-learn TfidfVectorizer according to project specifications.

    Default specifications:
    - ngram_range=(1, 2): unigrams and bigrams
    - min_df=2: ignore terms appearing in fewer than 2 documents
    - max_features=50000: top 50,000 most frequent n-grams
    - sublinear_tf=True: replaces tf with 1 + log(tf) for robust term weighting

    Args:
        ngram_range: Lower and upper boundary of range of n-values (default: (1, 2)).
        min_df: Minimum document frequency threshold (default: 2).
        max_features: Maximum vocabulary size (default: 50000).
        **kwargs: Additional keyword arguments passed to TfidfVectorizer.

    Returns:
        Unfitted TfidfVectorizer instance.
    """
    # Merge default sublinear_tf if not explicitly overridden
    vectorizer_params = {
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_features": max_features,
        "sublinear_tf": kwargs.pop("sublinear_tf", True),
    }
    vectorizer_params.update(kwargs)

    vectorizer = TfidfVectorizer(**vectorizer_params)
    logger.info(
        f"Initialized TfidfVectorizer(ngram_range={ngram_range}, min_df={min_df}, "
        f"max_features={max_features})"
    )
    return vectorizer


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    path: Union[str, Path] = DEFAULT_VECTORIZER_PATH,
) -> Path:
    """Save fitted TF-IDF vectorizer to disk using joblib.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        path: Destination file path (default: `models/tfidf_vectorizer.joblib`).

    Returns:
        Path to the saved vectorizer file.

    Raises:
        ValueError: If vectorizer is not fitted.
        IOError: If saving fails.
    """
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("Cannot save unfitted TfidfVectorizer. Fit vectorizer before saving.")

    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(vectorizer, save_path)
        logger.info(
            f"Saved TF-IDF vectorizer ({len(vectorizer.vocabulary_):,} features) to '{save_path}' "
            f"({save_path.stat().st_size / 1024:.1f} KB)"
        )
        return save_path
    except Exception as e:
        logger.error(f"Failed to save TF-IDF vectorizer to '{save_path}': {e}")
        raise IOError(f"Error saving vectorizer to {save_path}: {e}") from e


def load_vectorizer(
    path: Union[str, Path] = DEFAULT_VECTORIZER_PATH,
) -> TfidfVectorizer:
    """Load a fitted TfidfVectorizer from disk using joblib.

    Args:
        path: Path to serialized vectorizer file (default: `models/tfidf_vectorizer.joblib`).

    Returns:
        Loaded TfidfVectorizer instance.

    Raises:
        FileNotFoundError: If the vectorizer file does not exist.
        TypeError: If loaded object is not a TfidfVectorizer.
    """
    load_path = Path(path)
    if not load_path.is_file():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at '{load_path}'. "
            "Please run fit_tfidf() first or train baseline models."
        )

    try:
        vectorizer = joblib.load(load_path)
        if not isinstance(vectorizer, TfidfVectorizer):
            raise TypeError(
                f"Expected TfidfVectorizer instance from '{load_path}', got {type(vectorizer)}"
            )
        vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else "unknown"
        logger.info(f"Loaded TF-IDF vectorizer from '{load_path}' (Vocabulary size: {vocab_size})")
        return vectorizer
    except Exception as e:
        logger.error(f"Failed to load TF-IDF vectorizer from '{load_path}': {e}")
        raise


def fit_tfidf(
    train_texts: Union[pd.Series, List[str], np.ndarray, Iterable[str]],
    save_path: Union[str, Path] = DEFAULT_VECTORIZER_PATH,
    vectorizer: Optional[TfidfVectorizer] = None,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
    max_features: int = 50000,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on training texts and save to disk.

    Args:
        train_texts: Collection of raw or preprocessed training text strings.
        save_path: File path to save fitted vectorizer (default: `models/tfidf_vectorizer.joblib`).
        vectorizer: Optional pre-configured TfidfVectorizer. If None, builds one with specified args.
        ngram_range: Lower and upper boundary of range of n-values (default: (1, 2)).
        min_df: Minimum document frequency threshold (default: 2).
        max_features: Maximum vocabulary size (default: 50000).
        **kwargs: Additional parameters passed to `build_tfidf_vectorizer`.

    Returns:
        Fitted TfidfVectorizer instance.
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            **kwargs,
        )

    # Convert inputs to clean string list/iterable
    if isinstance(train_texts, pd.Series):
        texts_list = train_texts.fillna("").tolist()
    else:
        texts_list = [str(t) if t is not None else "" for t in train_texts]

    logger.info(f"Fitting TF-IDF vectorizer on {len(texts_list):,} training texts...")
    vectorizer.fit(texts_list)
    logger.info(f"TF-IDF vectorizer fitted successfully. Vocabulary size: {len(vectorizer.vocabulary_):,}")

    if save_path:
        save_vectorizer(vectorizer, path=save_path)

    return vectorizer


def transform_tfidf(
    texts: Union[pd.Series, List[str], np.ndarray, Iterable[str]],
    vectorizer: Optional[TfidfVectorizer] = None,
    vectorizer_path: Union[str, Path] = DEFAULT_VECTORIZER_PATH,
) -> csr_matrix:
    """Transform texts using a fitted vectorizer or loaded from disk.

    Args:
        texts: Collection of text strings to transform.
        vectorizer: Fitted TfidfVectorizer. If None, loads from `vectorizer_path`.
        vectorizer_path: Path to load vectorizer if not provided directly.

    Returns:
        scipy.sparse.csr_matrix containing TF-IDF feature representations.

    Raises:
        FileNotFoundError: If vectorizer is not provided and vectorizer_path is missing.
    """
    if vectorizer is None:
        vectorizer = load_vectorizer(vectorizer_path)

    # Convert inputs to clean string list
    if isinstance(texts, pd.Series):
        texts_list = texts.fillna("").tolist()
    else:
        texts_list = [str(t) if t is not None else "" for t in texts]

    features = vectorizer.transform(texts_list)
    logger.debug(f"Transformed {features.shape[0]:,} texts into {features.shape[1]:,} features.")
    return features


# ==============================================================================
# CLI / Testing Entry Point
# ==============================================================================
if __name__ == "__main__":
    print("\n--- Running src/preprocessing.py Verification Test ---")

    # 1. Test conservative cleaning
    raw_samples = [
        "Hey @toxic_user, visit http://malicious-site.com/attack NOW! You are a complete loser 😡🤬!!!",
        "Congratulations @bestie on your graduation 🎉🎓!! So happy for you <3 :)",
        "Why would you even say that??? That's horrible... #Bullying #StopIt",
        "Check this out: https://t.co/abc123xyz and tell @john what you think! 💀🤡",
        None,
        "   Normal text with     lots of    whitespace...   ",
    ]

    print("\n[Test 1: Conservative Text Preprocessing]")
    for sample in raw_samples:
        cleaned = preprocess_text(sample)
        print(f"  Raw:     {sample!r}")
        print(f"  Cleaned: {cleaned!r}\n")

    # Verify emoji and punctuation retention
    emoji_sample = "You suck 😡 and you should leave! Why??? #loser @target"
    emoji_cleaned = preprocess_text(emoji_sample)
    assert "😡" in emoji_cleaned, "Emoji should be preserved!"
    assert "!" in emoji_cleaned and "?" in emoji_cleaned, "Punctuation should be preserved!"
    assert "@target" not in emoji_cleaned, "Mention should be removed!"
    assert "#loser" in emoji_cleaned, "Hashtag text should be preserved!"
    print("[PASS] Emojis and punctuation correctly preserved, mentions removed.")

    # 2. Test DataFrame Preprocessing
    df_sample = pd.DataFrame({
        "text": [
            "You are ugly @loser! http://bit.ly/123",
            "Great effort today @everyone! Keep it up 👍",
        ],
        "label": [1, 0],
    })
    preprocessed_df = preprocess_dataframe(df_sample, text_col="text")
    print("\n[Test 2: DataFrame Preprocessing]")
    print(preprocessed_df)

    # 3. Test TF-IDF Vectorizer Build, Fit, Save, Load, and Transform
    print("\n[Test 3: TF-IDF Vectorization Pipeline]")
    sample_corpus = [
        "you are a disgusting loser and nobody likes you",
        "delete your account immediately you ugly clown 🤡",
        "had such a wonderful time studying today with good friends",
        "happy birthday! wishing you a peaceful and joyful year ahead",
        "disgusting freak, get off the internet",
        "congratulations on the great achievement and hard work",
    ]

    # Fit and save
    test_vec_path = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
    vec = fit_tfidf(
        sample_corpus,
        save_path=test_vec_path,
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
    )
    print(f"Fitted vectorizer features count: {len(vec.vocabulary_)}")

    # Load from disk
    loaded_vec = load_vectorizer(test_vec_path)
    assert len(loaded_vec.vocabulary_) == len(vec.vocabulary_)
    print("[PASS] Loaded vectorizer successfully from disk.")

    # Transform
    test_queries = ["you are an ugly loser 😡", "wonderful friends and great time"]
    feats = transform_tfidf(test_queries, vectorizer=loaded_vec)
    print(f"Transformed test queries matrix shape: {feats.shape}")
    assert feats.shape[0] == 2
    assert feats.shape[1] == len(vec.vocabulary_)
    print("[PASS] Transform completed successfully.")

    print("\nsrc/preprocessing.py passed all tests successfully!\n")
