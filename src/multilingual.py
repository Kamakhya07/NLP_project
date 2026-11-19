"""
Multilingual NLP and Code-Mixed Language Processing Module.
Fulfills Course Outcome CO3: Topic Modeling & Multilingual NLP.
Provides:
  1. Automated Language Identification (English, Hindi, Spanish, French, German, etc.)
  2. Code-Mixed Hinglish (Hindi written in Roman script) Detection & Normalization
  3. Cross-lingual semantic transfer for cyberbullying classification.
"""

import re
import logging
from typing import Dict, Any, Tuple, List
import langdetect
from langdetect import DetectorFactory

# Set seed for deterministic language detection
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)

# Hinglish (Romanized Hindi) Lexicon for code-mixed social media comments
HINGLISH_ABUSIVE_TERMS = {
    "pagal": "insane idiot",
    "kutta": "dog / derogatory slur",
    "kutti": "bitch / abusive slur",
    "kamina": "scoundrel",
    "kaminey": "scoundrels",
    "chutiya": "profane idiot",
    "gaddha": "donkey / dumb",
    "suar": "pig / derogatory",
    "bakwas": "nonsense garbage",
    "ghatiya": "worthless low-class",
    "marja": "go die",
    "sharam": "shame",
    "besharam": "shameless",
    "saale": "abusive prefix",
    "tera": "your",
    "teri": "your",
    "tujhe": "you",
    "aukat": "status / worthlessness",
    "dhokhebaaz": "traitor",
    "chor": "thief",
    "chup": "shut up"
}

HINGLISH_SUPPORTIVE_TERMS = {
    "badiya": "great awesome",
    "shandar": "magnificent",
    "shukriya": "thank you",
    "dhanyawad": "thank you so much",
    "dost": "friend",
    "dosti": "friendship",
    "sundar": "beautiful",
    "pyara": "lovely sweet",
    "pyari": "lovely sweet",
    "khoobsurat": "gorgeous",
    "badhai": "congratulations",
    "accha": "good nice",
    "zabardast": "fantastic"
}

ISO_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ur": "Urdu",
    "bn": "Bengali",
    "ar": "Arabic"
}

class MultilingualProcessor:
    """Detects languages and processes multilingual or code-mixed input."""

    def __init__(self):
        pass

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Identifies language, with special handling for Hinglish code-mixing."""
        if not text or not text.strip():
            return {"lang_code": "en", "language": "English", "is_hinglish": False}

        text_lower = text.strip().lower()

        # Check for Hinglish code-mixing
        hinglish_words_found = []
        for word in text_lower.split():
            clean_w = re.sub(r"[^\w]", "", word)
            if clean_w in HINGLISH_ABUSIVE_TERMS or clean_w in HINGLISH_SUPPORTIVE_TERMS:
                hinglish_words_found.append(clean_w)

        if len(hinglish_words_found) >= 1:
            return {
                "lang_code": "hi-en",
                "language": "Hinglish (Code-Mixed)",
                "is_hinglish": True,
                "detected_terms": hinglish_words_found
            }

        # Use langdetect for standard ISO languages
        try:
            detected_code = langdetect.detect(text)
            lang_name = ISO_LANG_NAMES.get(detected_code, detected_code.upper())
            return {
                "lang_code": detected_code,
                "language": lang_name,
                "is_hinglish": False,
                "detected_terms": []
            }
        except Exception:
            return {
                "lang_code": "en",
                "language": "English (Default)",
                "is_hinglish": False,
                "detected_terms": []
            }

    def normalize_and_translate(self, text: str) -> Tuple[str, bool, List[str]]:
        """
        Translates or normalizes code-mixed terms to English semantics
        so our contextual classifiers can evaluate them seamlessly.
        """
        lang_info = self.detect_language(text)
        is_hinglish = lang_info["is_hinglish"]
        detected_terms = lang_info.get("detected_terms", [])

        if not is_hinglish:
            return text, False, []

        tokens = text.split()
        normalized_tokens = []
        translated_glossary = []

        for token in tokens:
            punct_strip = re.sub(r"[^\w]", "", token).lower()
            if punct_strip in HINGLISH_ABUSIVE_TERMS:
                trans = HINGLISH_ABUSIVE_TERMS[punct_strip]
                normalized_tokens.append(trans)
                translated_glossary.append(f"{punct_strip} → {trans}")
            elif punct_strip in HINGLISH_SUPPORTIVE_TERMS:
                trans = HINGLISH_SUPPORTIVE_TERMS[punct_strip]
                normalized_tokens.append(trans)
                translated_glossary.append(f"{punct_strip} → {trans}")
            else:
                normalized_tokens.append(token)

        normalized_text = " ".join(normalized_tokens)
        return normalized_text, True, translated_glossary
