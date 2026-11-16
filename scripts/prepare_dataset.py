"""
Robust Dataset Ingestion and Preprocessing Pipeline.
Combines benchmark social media offensive/bullying tweets (CardiffNLP TweetEval)
with balanced conversational, supportive, and offensive samples to eliminate
spurious lexical correlation (e.g. pronoun bias on 'you' or 'like').
"""

import os
import re
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = Path("D:/TY/NLP-project")
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
DATA_DIR = BASE_DIR / "data"

for d in [RAW_DIR, PROCESSED_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def clean_text_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_balanced_dataset():
    logger.info("Loading CardiffNLP TweetEval Offensive benchmark dataset...")
    ds_te = load_dataset("cardiffnlp/tweet_eval", "offensive")
    
    df_te_train = ds_te["train"].to_pandas()[["text", "label"]]
    df_te_val = ds_te["validation"].to_pandas()[["text", "label"]]
    df_te_test = ds_te["test"].to_pandas()[["text", "label"]]
    df_te = pd.concat([df_te_train, df_te_val, df_te_test], ignore_index=True)
    logger.info(f"Loaded {len(df_te)} records from TweetEval.")

    # High-quality natural conversational expressions to break spurious single-word biases
    positive_phrases = [
        "i like you", "i really like you", "i like you so much", "i like you a lot",
        "i just like you", "i like you friend", "we all like you", "everyone likes you",
        "i like your post", "i like your idea", "i like your vibe", "i like your profile",
        "i like hanging out with you", "i like talking to you", "i like this so much",
        "i love you", "i love you so much", "love you all", "i love this picture",
        "you are amazing", "you are an amazing and kind person", "you are truly wonderful",
        "you are the best friend ever", "you look great today", "you have a great smile",
        "you did an awesome job today", "proud of you", "so proud of what you achieved",
        "thank you so much for your help", "i appreciate you and your time", "thanks for being there for me",
        "you are very smart and hardworking", "you inspire me every day", "wish you all the best",
        "happy birthday to you", "hope you have a wonderful day", "congratulations to you",
        "good luck with your exams", "you are welcome anytime", "glad to see you here",
        "you are fantastic", "great to collaborate with you", "enjoy your weekend with friends",
        "you are so sweet and helpful", "you have a wonderful family", "you did great in the presentation",
        "had an amazing time collaborating with our study group today! so grateful for supportive friends."
    ]

    # Clear bullying/harassment samples with diverse syntaxes
    bullying_phrases = [
        "i hate you", "i hate you so much", "i hate your guts",
        "i hate you, you are ugly and worthless", "you are so stupid and useless",
        "nobody likes you get lost", "you are ugly and worthless",
        "i will beat you up after school", "kill yourself pathetic loser",
        "ur such a loser lmao go die", "shut up idiot nobody asked for your opinion",
        "you have no friends and never will", "you are a complete waste of space",
        "delete your account you clown", "you disgust me stay away",
        "you are pathetic and weak", "everybody is laughing at how dumb you are",
        "you deserve all the pain you get", "disgusting freak nobody cares about you",
        "stop breathing our air trash", "you are the worst human being alive"
    ]

    extra_rows = []
    for p in positive_phrases:
        for _ in range(12):
            extra_rows.append({"text": p, "label": 0})
            extra_rows.append({"text": p + "!", "label": 0})
            extra_rows.append({"text": "hey " + p, "label": 0})

    for b in bullying_phrases:
        for _ in range(10):
            extra_rows.append({"text": b, "label": 1})
            extra_rows.append({"text": b + "!", "label": 1})

    df_extra = pd.DataFrame(extra_rows)
    df_full = pd.concat([df_te, df_extra], ignore_index=True)
    df_full = df_full.drop_duplicates(subset=["text"]).dropna()
    df_full["clean_text"] = df_full["text"].apply(clean_text_basic)
    df_full = df_full[df_full["clean_text"].str.len() > 1].reset_index(drop=True)

    logger.info(f"Total dataset size after deduplication: {len(df_full)} samples")
    logger.info(f"Class distribution:\n{df_full['label'].value_counts()}")

    # Stratified train (75%), val (12.5%), test (12.5%)
    train_df, temp_df = train_test_split(
        df_full, test_size=0.25, random_state=RANDOM_SEED, stratify=df_full["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )

    logger.info(f"Train split: {len(train_df)}, Val split: {len(val_df)}, Test split: {len(test_df)}")

    # Save to dataset/processed and data/
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    train_df.to_csv(DATA_DIR / "processed_train.csv", index=False)
    test_df.to_csv(DATA_DIR / "processed_test.csv", index=False)
    df_full.to_csv(RAW_DIR / "cyberbullying_dataset.csv", index=False)
    df_full.to_csv(DATA_DIR / "raw_dataset.csv", index=False)

    logger.info("Dataset files successfully saved to disk.")

if __name__ == "__main__":
    build_balanced_dataset()
