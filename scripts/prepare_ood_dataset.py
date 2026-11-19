"""
Prepare External Out-of-Distribution (OOD) Benchmark Dataset
Sources: Waseem & Hovy (NAACL 2016) and Davidson et al. (2017)
Consolidates ~35,000 tweets into standardized format: text, label (0=Non-Bullying, 1=Bullying/Offensive)
"""

import os
import re
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path("D:/TY/NLP-project")
OOD_DIR = BASE_DIR / "dataset" / "external_ood"
OOD_DIR.mkdir(parents=True, exist_ok=True)

# Path to cloned repository scratch folder
SOURCE_DIR = Path(r"C:\Users\kamakhya\.gemini\antigravity\brain\5e384dbb-08c0-41bc-ac7b-62f35b1561ce\scratch\dhaval_repo")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    logger.info("Loading Waseem & Hovy scraped tweets...")
    df_waseem = pd.read_csv(SOURCE_DIR / "labeled_tweets.csv")
    logger.info(f"Loaded {len(df_waseem)} rows from labeled_tweets.csv")
    
    logger.info("Loading Davidson et al. public dataset...")
    df_davidson = pd.read_csv(SOURCE_DIR / "public_data_labeled.csv")
    logger.info(f"Loaded {len(df_davidson)} rows from public_data_labeled.csv")

    # Keep necessary columns: full_text, label
    df_waseem = df_waseem[["full_text", "label"]].copy()
    df_davidson = df_davidson[["full_text", "label"]].copy()

    # Combine
    df_combined = pd.concat([df_waseem, df_davidson], ignore_index=True)
    logger.info(f"Combined total before deduplication: {len(df_combined)}")

    # Deduplicate
    df_combined.drop_duplicates(subset=["full_text"], inplace=True)
    logger.info(f"Combined total after deduplication: {len(df_combined)}")

    # Map labels to binary: 0 = Non-offensive, 1 = Offensive
    # Normalize label strings
    df_combined["label_clean"] = df_combined["label"].astype(str).str.strip().str.lower()
    
    # Check values
    logger.info(f"Raw label distribution:\n{df_combined['label_clean'].value_counts()}")

    label_map = {
        "offensive": 1,
        "hate speech": 1,
        "non-offensive": 0,
        "neither": 0
    }

    df_combined["target"] = df_combined["label_clean"].map(label_map)
    df_combined.dropna(subset=["target"], inplace=True)
    df_combined["target"] = df_combined["target"].astype(int)

    # Clean text
    df_combined["cleaned_text"] = df_combined["full_text"].apply(clean_text)
    
    # Filter out empty or trivially short text (< 3 chars)
    df_combined = df_combined[df_combined["cleaned_text"].str.len() >= 3].copy()
    
    output_df = df_combined[["full_text", "cleaned_text", "target"]].rename(
        columns={"full_text": "raw_text", "cleaned_text": "text", "target": "label"}
    )

    logger.info(f"Final OOD benchmark records: {len(output_df)}")
    logger.info(f"Final Label counts:\n{output_df['label'].value_counts(normalize=True)}")

    output_path = OOD_DIR / "waseem_davidson_ood.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Successfully saved OOD benchmark dataset to {output_path}")

if __name__ == "__main__":
    main()
