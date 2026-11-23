"""
Topic Modeling Module using Latent Dirichlet Allocation (LDA).
Fulfills Course Outcome CO3: Topic Modeling & Multilingual NLP.
Discovers semantic themes in social media text (e.g. Interpersonal Insults,
Appearance Slurs, Hostile Silencing, Community Support).
"""

import os
import re
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LDA_MODEL_PATH = MODELS_DIR / "lda_model.joblib"
LDA_VEC_PATH = MODELS_DIR / "lda_vectorizer.joblib"

TOPIC_DESCRIPTIONS = {
    0: {
        "name": "Direct Hostility & Aggression",
        "description": "Explicit insults, direct targeting, and hostile personal abuse.",
        "badge_color": "danger"
    },
    1: {
        "name": "Social & Ideological Conflict",
        "description": "Public debates, controversy, polarized social discourse.",
        "badge_color": "warning"
    },
    2: {
        "name": "Degradation & Slurs",
        "description": "Dehumanizing slurs, appearance shaming, and pejorative labeling.",
        "badge_color": "danger"
    },
    3: {
        "name": "Supportive Community & Gratitude",
        "description": "Compliments, encouraging feedback, friendly everyday exchanges.",
        "badge_color": "success"
    }
}

class LDATopicModeler:
    """Manages LDA topic extraction and runtime inference."""

    def __init__(self, n_topics: int = 4):
        self.n_topics = n_topics
        self.vectorizer = None
        self.lda_model = None
        self._load_or_train()

    def _load_or_train(self):
        """Loads serialized LDA artifacts or fits new ones if missing."""
        if LDA_MODEL_PATH.exists() and LDA_VEC_PATH.exists():
            try:
                self.lda_model = joblib.load(LDA_MODEL_PATH)
                self.vectorizer = joblib.load(LDA_VEC_PATH)
                logger.info("LDA Topic Model and Vectorizer loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached LDA model: {e}. Refitting...")

        self.fit()

    def fit(self, max_samples: int = 10000):
        """Fits Latent Dirichlet Allocation on the training dataset."""
        train_path = PROJECT_ROOT / "dataset" / "processed" / "train.csv"
        if not train_path.exists():
            logger.warning("Train CSV not found for LDA fitting. Skipping...")
            return

        df = pd.read_csv(train_path)
        texts = df["clean_text"].dropna().astype(str).tolist()[:max_samples]

        logger.info(f"Fitting LDA on {len(texts)} samples across {self.n_topics} latent topics...")

        self.vectorizer = CountVectorizer(
            stop_words="english",
            max_df=0.90,
            min_df=3,
            max_features=2500,
            ngram_range=(1, 2)
        )
        dtm = self.vectorizer.fit_transform(texts)

        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method="online",
            max_iter=10
        )
        self.lda_model.fit(dtm)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.lda_model, LDA_MODEL_PATH)
        joblib.dump(self.vectorizer, LDA_VEC_PATH)
        logger.info("LDA Topic Model successfully trained and serialized.")

    def get_topic_keywords(self, top_n: int = 5) -> Dict[int, List[str]]:
        """Returns the top words associated with each latent topic."""
        if not self.lda_model or not self.vectorizer:
            return {}

        feature_names = self.vectorizer.get_feature_names_out()
        topic_words = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[: -top_n - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            topic_words[topic_idx] = top_features
        return topic_words

    def predict_topic(self, text: str) -> Dict[str, Any]:
        """Infers the dominant topic and probability distribution for a text sample."""
        if not self.lda_model or not self.vectorizer:
            return {
                "topic_id": 0,
                "topic_name": "General Communication",
                "confidence": 0.5,
                "keywords": [],
                "distribution": [0.25, 0.25, 0.25, 0.25]
            }

        vec = self.vectorizer.transform([text.lower()])
        topic_dist = self.lda_model.transform(vec)[0]
        dominant_idx = int(topic_dist.argmax())
        confidence = float(topic_dist[dominant_idx])

        meta = TOPIC_DESCRIPTIONS.get(dominant_idx, {
            "name": f"Topic {dominant_idx}",
            "description": "Extracted semantic cluster",
            "badge_color": "secondary"
        })

        keywords = self.get_topic_keywords(top_n=4).get(dominant_idx, [])

        return {
            "topic_id": dominant_idx,
            "topic_name": meta["name"],
            "topic_description": meta["description"],
            "badge_color": meta["badge_color"],
            "confidence": round(confidence * 100, 1),
            "keywords": keywords,
            "distribution": [round(float(p) * 100, 1) for p in topic_dist]
        }
