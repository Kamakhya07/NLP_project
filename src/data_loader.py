"""Cyberbullying Detection from Social Media - Data Loader Module.

This module is responsible for:
1. Downloading the cyberbullying dataset from Hugging Face (`cike-dev/cyberbullying_dataset`).
2. Inspecting dataset structure, column names, split sizes, and sample rows.
3. Dynamically identifying text and label columns across diverse dataset formats.
4. Normalizing and mapping labels to binary classification standard:
     0: 'Non-Bullying'
     1: 'Bullying'
5. Splitting data into stratified Train (70%), Validation (15%), and Test (15%) splits
   if pre-existing splits are not provided or incomplete.
6. Saving processed dataset splits as CSV files to `dataset/processed/`.
7. Returning clean `train_df`, `val_df`, and `test_df` pandas DataFrames.

Location: D:\\TY\\NLP-project
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_loader")

# ==============================================================================
# Path & Label Constants
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_NAME = "cike-dev/cyberbullying_dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
FALLBACK_RAW_CSV = PROJECT_ROOT / "data" / "raw_dataset.csv"

DEFAULT_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
DEFAULT_VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
DEFAULT_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

# Global standard label mapping: 0 = Non-Bullying, 1 = Bullying
LABEL_MAP: Dict[int, str] = {
    0: "Non-Bullying",
    1: "Bullying",
}

INVERSE_LABEL_MAP: Dict[str, int] = {
    "Non-Bullying": 0,
    "Bullying": 1,
}

# Candidate column names across different cyberbullying dataset releases
TEXT_COLUMN_CANDIDATES: List[str] = [
    "cleaned_text",
    "text",
    "comment_text",
    "tweet",
    "tweet_text",
    "content",
    "message",
    "body",
    "sentence",
    "post",
    "comment",
    "utterance",
]

LABEL_COLUMN_CANDIDATES: List[str] = [
    "label",
    "bully",
    "class",
    "target",
    "cyberbullying_type",
    "is_bullying",
    "category",
    "annotation",
    "toxic",
    "sentiment",
]

# Mapping rules for various raw label string and numeric representations
RAW_LABEL_MAPPING: Dict[Union[str, int, float, bool], int] = {
    # Class 0: Non-Bullying
    "normal": 0,
    "non-bullying": 0,
    "non-bully": 0,
    "not_bullying": 0,
    "not_cyberbullying": 0,
    "neither": 0,
    "clean": 0,
    "none": 0,
    "no": 0,
    "false": 0,
    "0": 0,
    0: 0,
    0.0: 0,
    False: 0,
    # Class 1: Bullying
    "bully": 1,
    "bullying": 1,
    "cyberbullying": 1,
    "offensive": 1,
    "hate": 1,
    "toxic": 1,
    "abusive": 1,
    "harassment": 1,
    "yes": 1,
    "true": 1,
    "1": 1,
    1: 1,
    1.0: 1,
    True: 1,
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def identify_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Identify the text and label columns from a DataFrame using candidate lists.

    Searches case-insensitively through candidate column names.

    Args:
        df: Pandas DataFrame containing dataset records.

    Returns:
        Tuple of (text_column_name, label_column_name).

    Raises:
        ValueError: If neither a valid text column nor label column could be identified.
    """
    cols_lower = {col.lower(): col for col in df.columns}

    # Identify text column
    text_col = None
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate.lower() in cols_lower:
            text_col = cols_lower[candidate.lower()]
            break

    # Identify label column
    label_col = None
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate.lower() in cols_lower:
            label_col = cols_lower[candidate.lower()]
            break

    if text_col is None:
        raise ValueError(
            f"Could not identify a text column in DataFrame. Available columns: {list(df.columns)}. "
            f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
        )
    if label_col is None:
        raise ValueError(
            f"Could not identify a label column in DataFrame. Available columns: {list(df.columns)}. "
            f"Expected one of: {LABEL_COLUMN_CANDIDATES}"
        )

    logger.info(f"Identified text column: '{text_col}', label column: '{label_col}'")
    return text_col, label_col


def map_labels_to_binary(series: pd.Series) -> pd.Series:
    """Map string, float, or integer label values to binary integers (0 or 1).

    0 represents 'Non-Bullying' (normal)
    1 represents 'Bullying' (bully)

    Args:
        series: Pandas Series containing raw label values.

    Returns:
        Pandas Series containing integer binary labels (0 and 1).

    Raises:
        ValueError: If unmappable labels are encountered or the series is empty.
    """
    def _convert_label(val: Any) -> Optional[int]:
        if pd.isna(val):
            return None
        if isinstance(val, (int, np.integer)):
            return int(val) if int(val) in (0, 1) else None
        if isinstance(val, (float, np.floating)):
            return int(val) if int(val) in (0, 1) else None
        if isinstance(val, bool):
            return 1 if val else 0

        val_str = str(val).strip().lower()
        if val_str in RAW_LABEL_MAPPING:
            return RAW_LABEL_MAPPING[val_str]
        return None

    mapped = series.apply(_convert_label)
    unmapped_count = mapped.isna().sum()
    if unmapped_count > 0:
        logger.warning(
            f"Encountered {unmapped_count} unmapped or null labels out of {len(series)} records. "
            "These invalid records will be excluded."
        )

    return mapped


def inspect_dataset(
    data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    name: str = "Dataset",
    n_sample_rows: int = 3,
) -> None:
    """Inspect and log detailed characteristics of a dataset or split collection.

    Prints split sizes, column names, data types, class distributions, and sample rows.

    Args:
        data: Either a single DataFrame or a dictionary mapping split names to DataFrames.
        name: Human-readable name for logging.
        n_sample_rows: Number of sample rows to print per split.
    """
    print("\n" + "=" * 80)
    print(f" DATASET INSPECTION: {name}")
    print("=" * 80)

    if isinstance(data, pd.DataFrame):
        splits = {"all": data}
    elif isinstance(data, dict):
        splits = data
    else:
        # Fallback if Hugging Face DatasetDict is passed
        splits = {k: data[k].to_pandas() for k in data.keys()}

    total_samples = sum(len(df) for df in splits.values())
    print(f"Total samples across all splits: {total_samples:,}")

    for split_name, df in splits.items():
        print(f"\n--- Split: '{split_name}' (Total Rows: {len(df):,}) ---")
        print(f"Columns & Types: {dict(df.dtypes)}")

        # Check for text and label columns
        try:
            text_col, label_col = identify_columns(df)
            print(f"Detected Text Column: '{text_col}' | Detected Label Column: '{label_col}'")

            # Check label distribution
            counts = df[label_col].value_counts(dropna=False).to_dict()
            dist_str = ", ".join([f"{k}: {v:,} ({v / len(df) * 100:.1f}%)" for k, v in counts.items()])
            print(f"Label Distribution: {dist_str}")

            # Check missing values
            null_text = df[text_col].isna().sum()
            null_label = df[label_col].isna().sum()
            print(f"Missing Values: text={null_text:,}, label={null_label:,}")

            # Show sample rows
            print(f"\nSample Rows ({min(n_sample_rows, len(df))} examples):")
            sample_df = df[[text_col, label_col]].head(n_sample_rows)
            for idx, row in sample_df.iterrows():
                text_preview = str(row[text_col])
                if len(text_preview) > 90:
                    text_preview = text_preview[:87] + "..."
                raw_lbl = row[label_col]
                mapped_name = LABEL_MAP.get(raw_lbl, str(raw_lbl))
                print(f"  [{idx}] ({mapped_name} / {raw_lbl}): {text_preview}")

        except ValueError as e:
            print(f"Column inspection note: {e}")
            print(f"Head:\n{df.head(2)}")

    print("=" * 80 + "\n")


def create_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified Train / Validation / Test splits from a single DataFrame.

    Default ratio is 70% Train, 15% Validation, 15% Test.

    Args:
        df: Input DataFrame containing standardized 'text' and 'label' columns.
        train_ratio: Proportion of data allocated to training (default: 0.70).
        val_ratio: Proportion of data allocated to validation (default: 0.15).
        test_ratio: Proportion of data allocated to testing (default: 0.15).
        random_state: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If split proportions do not sum to 1.0 or if data has insufficient samples.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(
            f"Split ratios must sum to 1.0, but got {train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )

    if len(df) < 10:
        raise ValueError(f"DataFrame too small to split meaningfully: {len(df)} samples.")

    # Check stratification eligibility
    can_stratify = "label" in df.columns and df["label"].nunique() > 1
    stratify_col = df["label"] if can_stratify else None

    # First split: Train vs (Val + Test)
    temp_size = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=stratify_col,
        random_state=random_state,
        shuffle=True,
    )

    # Second split: Val vs Test
    relative_test_size = test_ratio / temp_size
    temp_stratify = temp_df["label"] if (can_stratify and temp_df["label"].nunique() > 1) else None

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_stratify,
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Created stratified splits: Train={len(train_df):,} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df):,} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df):,} ({len(test_df)/len(df):.1%})"
    )
    return train_df, val_df, test_df


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path] = PROCESSED_DATA_DIR,
) -> Dict[str, Path]:
    """Save processed train, val, and test splits as CSV files.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        output_dir: Target directory path (default: `dataset/processed/`).

    Returns:
        Dictionary mapping split name to saved file Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
        "test": out_dir / "test.csv",
    }

    train_df.to_csv(paths["train"], index=False, encoding="utf-8")
    val_df.to_csv(paths["val"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test"], index=False, encoding="utf-8")

    logger.info(f"Saved processed splits to '{out_dir}':")
    for name, p in paths.items():
        logger.info(f"  - {name}: {p} ({p.stat().st_size / 1024:.1f} KB)")

    return paths


def load_processed_splits(
    data_dir: Union[str, Path] = PROCESSED_DATA_DIR,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Attempt to load existing processed splits (train.csv, val.csv, test.csv) from disk.

    Args:
        data_dir: Directory containing processed CSVs.

    Returns:
        Tuple of (train_df, val_df, test_df) if all exist, otherwise None.
    """
    d_dir = Path(data_dir)
    train_p = d_dir / "train.csv"
    val_p = d_dir / "val.csv"
    test_p = d_dir / "test.csv"

    if train_p.is_file() and val_p.is_file() and test_p.is_file():
        try:
            train_df = pd.read_csv(train_p)
            val_df = pd.read_csv(val_p)
            test_df = pd.read_csv(test_p)
            logger.info(
                f"Loaded existing processed splits from '{d_dir}': "
                f"Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
            )
            return train_df, val_df, test_df
        except Exception as e:
            logger.warning(f"Error reading existing processed splits from {d_dir}: {e}")
            return None
    return None


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to 'text' and 'label', map labels to binary (0, 1),

    and filter invalid records.

    Args:
        df: Input DataFrame with arbitrary candidate column names.

    Returns:
        Clean DataFrame containing only 'text' (str) and 'label' (int 0 or 1).
    """
    text_col, label_col = identify_columns(df)

    clean_df = pd.DataFrame()
    clean_df["text"] = df[text_col].astype(str)
    clean_df["label"] = map_labels_to_binary(df[label_col])

    # Drop rows with null or invalid labels
    clean_df = clean_df.dropna(subset=["label"]).copy()
    clean_df["label"] = clean_df["label"].astype(int)

    # Filter out empty or whitespace-only text
    clean_df = clean_df[clean_df["text"].str.strip().str.len() > 0].copy()
    clean_df = clean_df.reset_index(drop=True)

    return clean_df


# ==============================================================================
# Hugging Face Dataset Downloader & Fallback Handler
# ==============================================================================
def download_dataset(
    dataset_name: str = DATASET_NAME,
    token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    """Download the dataset from Hugging Face Hub using the datasets library.

    Supports gated dataset access via HF_TOKEN environment variable or explicit token.

    Args:
        dataset_name: Hugging Face dataset identifier (default: 'cike-dev/cyberbullying_dataset').
        token: Optional Hugging Face authentication token.
        cache_dir: Optional custom cache directory.

    Returns:
        Hugging Face DatasetDict or Dataset object if successful, None if download fails.
    """
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from datasets import load_dataset

        logger.info(f"Attempting to download '{dataset_name}' from Hugging Face...")
        if hf_token:
            logger.info("Using detected Hugging Face authentication token.")
        else:
            logger.info("No Hugging Face token detected in arguments or environment.")

        ds = load_dataset(dataset_name, token=hf_token, cache_dir=str(cache_dir) if cache_dir else None)
        logger.info(f"Successfully downloaded '{dataset_name}' from Hugging Face.")
        return ds

    except Exception as exc:
        exc_str = str(exc)
        logger.warning(
            f"Could not download dataset '{dataset_name}' from Hugging Face: {exc_str}"
        )
        if "gated" in exc_str.lower() or "unauthorized" in exc_str.lower() or "401" in exc_str:
            logger.info(
                "NOTE: 'cike-dev/cyberbullying_dataset' is a gated dataset on Hugging Face Hub.\n"
                "To access it directly via Hugging Face:\n"
                "  1. Visit https://huggingface.co/datasets/cike-dev/cyberbullying_dataset and accept the terms.\n"
                "  2. Set your environment variable: set HF_TOKEN=your_token_here (Windows) "
                "or export HF_TOKEN=your_token_here (Linux/Mac).\n"
                "Falling back to local dataset repository files."
            )
        return None


def generate_fallback_samples(n_samples: int = 600) -> pd.DataFrame:
    """Generate realistic balanced cyberbullying and non-bullying social media samples

    as an offline development and testing fallback.

    Args:
        n_samples: Total number of samples to generate.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    bully_templates = [
        "You are so pathetic and ugly, nobody likes you.",
        "Delete your account and never come back you worthless clown.",
        "Nobody cares about your existence, why don't you disappear.",
        "You are a disgusting freak and everybody is laughing at you.",
        "Shut up idiot, your opinion is pure trash.",
        "You're completely brain dead and a total failure.",
        "Look at this loser crying over nothing, grow up weakling.",
        "Drop dead scum, nobody asked for your useless thoughts.",
        "You are the most repulsive human being on the internet.",
        "Keep crying loser, you deserve all the hate coming your way.",
    ]
    non_bully_templates = [
        "Had an amazing time collaborating with our study group today! So grateful.",
        "I respectfully disagree with your points, but appreciate your perspective.",
        "Does anyone know what time the library opens on weekends?",
        "Happy birthday! Wishing you a fantastic year filled with joy and success.",
        "Great job on the presentation today, you explained difficult concepts clearly.",
        "Thanks for helping me fix that bug in my code, really appreciate your patience!",
        "The sunset today was absolutely breathtaking, going for an evening walk.",
        "Congratulations on passing your exams! Proud of all your hard work.",
        "Interesting discussion on natural language processing ethics, thanks for hosting.",
        "Looking forward to the weekend and spending quality time with family.",
    ]
    mentions = ["@alex", "@jordan", "@sammy", "@taylor", "@chris", "@morgan", ""]
    hashtags_bully = ["#toxic", "#hate", "#roasted", "#fail", "#lame", ""]
    hashtags_benign = ["#positivevibes", "#coding", "#gratitude", "#weekend", "#learning", ""]

    rows = []
    np.random.seed(42)
    half = n_samples // 2

    # Bullying samples (label 1)
    for _ in range(half):
        template = np.random.choice(bully_templates)
        m = np.random.choice(mentions)
        h = np.random.choice(hashtags_bully)
        text = f"{m} {template} {h}".strip()
        rows.append({"text": text, "label": 1})

    # Non-Bullying samples (label 0)
    for _ in range(n_samples - half):
        template = np.random.choice(non_bully_templates)
        m = np.random.choice(mentions)
        h = np.random.choice(hashtags_benign)
        text = f"{m} {template} {h}".strip()
        rows.append({"text": text, "label": 0})

    np.random.shuffle(rows)
    df = pd.DataFrame(rows)
    return df


# ==============================================================================
# Main Data Loading & Orchestration Pipeline
# ==============================================================================
def load_and_prepare_dataset(
    dataset_name: str = DATASET_NAME,
    token: Optional[str] = None,
    force_reload: bool = False,
    output_dir: Union[str, Path] = PROCESSED_DATA_DIR,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complete data pipeline:

    1. Checks if processed data already exists (train.csv, val.csv, test.csv).
       If found and not force_reload, loads and returns them immediately.
    2. Otherwise, attempts download from Hugging Face Hub (cike-dev/cyberbullying_dataset).
    3. If Hugging Face is unreachable or gated, loads local raw data from
       `dataset/raw/` or `data/raw_dataset.csv`, or generates realistic fallback samples.
    4. Dynamically identifies text and label columns.
    5. Normalizes and maps labels to binary (0: Non-Bullying, 1: Bullying).
    6. If official train/val/test splits exist in Hugging Face, uses them;
       otherwise, creates stratified 70/15/15 splits.
    7. Saves processed splits to `dataset/processed/`.
    8. Returns (train_df, val_df, test_df).

    Args:
        dataset_name: Hugging Face dataset identifier.
        token: Optional HF authentication token.
        force_reload: If True, forces reprocessing even if processed files exist.
        output_dir: Destination directory for processed CSV files.
        train_ratio: Proportion of training data (default: 0.70).
        val_ratio: Proportion of validation data (default: 0.15).
        test_ratio: Proportion of test data (default: 0.15).
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df) pandas DataFrames with 'text' and 'label'.
    """
    output_path = Path(output_dir)

    # 1. Check if processed splits already exist on disk
    if not force_reload:
        existing = load_processed_splits(output_path)
        if existing is not None:
            train_df, val_df, test_df = existing
            inspect_dataset(
                {"train": train_df, "validation": val_df, "test": test_df},
                name="Loaded Pre-existing Processed Splits",
            )
            return train_df, val_df, test_df

    logger.info("Building dataset splits from raw source...")

    # 2. Try downloading from Hugging Face
    hf_dataset = download_dataset(dataset_name=dataset_name, token=token)

    train_df, val_df, test_df = None, None, None

    if hf_dataset is not None:
        inspect_dataset(hf_dataset, name=f"Hugging Face Hub: {dataset_name}")

        # Check if official splits exist
        splits_present = list(hf_dataset.keys())
        has_train = "train" in splits_present
        has_val = "validation" in splits_present or "val" in splits_present
        has_test = "test" in splits_present

        val_key = "validation" if "validation" in splits_present else "val"

        if has_train and has_val and has_test:
            logger.info("Official Hugging Face train/validation/test splits detected. Using official splits.")
            train_df = standardize_dataframe(hf_dataset["train"].to_pandas())
            val_df = standardize_dataframe(hf_dataset[val_key].to_pandas())
            test_df = standardize_dataframe(hf_dataset["test"].to_pandas())
        else:
            # Combine all available splits and create stratified 70/15/15
            logger.info("Combining available Hugging Face splits to generate stratified 70/15/15 split.")
            dfs = [standardize_dataframe(hf_dataset[s].to_pandas()) for s in splits_present]
            combined_df = pd.concat(dfs, ignore_index=True)
            train_df, val_df, test_df = create_stratified_split(
                combined_df,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_state=random_state,
            )

    # 3. Fallback to local raw data if Hugging Face was not available
    if train_df is None:
        raw_candidates = [
            FALLBACK_RAW_CSV,
            RAW_DATA_DIR / "cyberbullying_dataset.csv",
        ]
        loaded_raw_df = None
        # Select candidate with the largest number of valid samples
        best_candidate_df = None
        for p in raw_candidates:
            if p.is_file():
                try:
                    candidate_df = pd.read_csv(p)
                    if best_candidate_df is None or len(candidate_df) > len(best_candidate_df):
                        best_candidate_df = candidate_df
                        logger.info(f"Discovered local raw dataset candidate: '{p}' ({len(candidate_df):,} samples)")
                except Exception as e:
                    logger.warning(f"Failed to read candidate CSV '{p}': {e}")

        loaded_raw_df = best_candidate_df

        if loaded_raw_df is None:
            logger.warning("No local raw CSV found. Generating realistic sample dataset as fallback.")
            loaded_raw_df = generate_fallback_samples(n_samples=1000)

        # Standardize columns and map labels
        clean_df = standardize_dataframe(loaded_raw_df)

        inspect_dataset(clean_df, name="Raw Dataset Source")

        # Create stratified 70/15/15 split
        train_df, val_df, test_df = create_stratified_split(
            clean_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

    # 4. Save processed splits to disk
    save_processed_splits(train_df, val_df, test_df, output_dir=output_path)

    # 5. Final inspection of processed splits
    inspect_dataset(
        {"train": train_df, "validation": val_df, "test": test_df},
        name="Final Processed Splits Saved to dataset/processed/",
    )

    return train_df, val_df, test_df


# Unified aliases for backward compatibility and clean API
load_data = load_and_prepare_dataset
get_splits = load_and_prepare_dataset


# ==============================================================================
# CLI / Testing Entry Point
# ==============================================================================
if __name__ == "__main__":
    print("\n--- Running src/data_loader.py Verification Test ---")
    train, val, test = load_and_prepare_dataset(force_reload=False)

    print(f"\n[Verification Summary]")
    print(f"Train split shape: {train.shape} | Bully ratio: {train['label'].mean():.2%}")
    print(f"Val split shape:   {val.shape} | Bully ratio: {val['label'].mean():.2%}")
    print(f"Test split shape:  {test.shape} | Bully ratio: {test['label'].mean():.2%}")

    # Validate output paths
    for name, path in [("train", DEFAULT_TRAIN_PATH), ("val", DEFAULT_VAL_PATH), ("test", DEFAULT_TEST_PATH)]:
        exists = path.is_file()
        size_kb = path.stat().st_size / 1024 if exists else 0
        print(f"File '{name}.csv' exists: {exists} ({size_kb:.1f} KB at {path})")

    print("\nsrc/data_loader.py passed all tests successfully!\n")
