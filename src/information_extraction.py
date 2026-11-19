"""
Information Extraction (IE) Module for Cyberbullying Analytics.
Fulfills Course Outcome CO2: Sentiment Analysis & Information Extraction.
Extracts:
  1. Attack Target Identification (Second-person, Third-person, Identity Group)
  2. Extracted Affective Keywords (Hostile vs. Supportive lexicon)
  3. Syntactic Structure (Imperative, Interrogative, Exclamatory)
  4. Toxicity Severity Intensity (Low, Medium, Severe)
"""

import re
from typing import Dict, List, Any

HOSTILE_LEXICON = {
    "hate", "stupid", "idiot", "loser", "ugly", "worthless", "die", "kill", "shut up",
    "trash", "bitch", "moron", "freak", "asshole", "pathetic", "retard", "scum", "bastard",
    "dumb", "clown", "fat", "disgusting", "cringe", "uninstall", "annoying", "crybaby"
}

SUPPORTIVE_LEXICON = {
    "love", "like", "amazing", "kind", "great", "awesome", "proud", "sweet", "best",
    "thank", "thanks", "helpful", "good", "wonderful", "beautiful", "smart", "welcome",
    "grateful", "congratulations", "inspire", "fantastic"
}

class InformationExtractor:
    """Extracts structured entities, targets, and sentiment indicators from comment text."""

    def __init__(self):
        pass

    def extract_information(self, text: str, is_bullying: bool, confidence: float) -> Dict[str, Any]:
        """Extracts structured linguistic information from the input text."""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))

        # 1. Target Extraction
        target = "General / Unspecified"
        if any(w in words for w in ["you", "your", "ur", "u", "yourself"]):
            target = "Direct Second-Person Target (User Addressee)"
        elif any(w in words for w in ["he", "she", "they", "them", "him", "her"]):
            target = "Third-Person Individual or Group"
        elif any(w in words for w in ["we", "us", "everyone", "people"]):
            target = "Collective Community / Group"

        # 2. Extract Affective Keywords
        extracted_hostile = list(words.intersection(HOSTILE_LEXICON))
        extracted_supportive = list(words.intersection(SUPPORTIVE_LEXICON))

        # 3. Syntactic Form
        syntactic_style = "Declarative Statement"
        if text.strip().endswith("?"):
            syntactic_style = "Interrogative Challenge / Question"
        elif text.strip().endswith("!"):
            syntactic_style = "Exclamatory (High Arousal Emphasis)"
        elif text_lower.startswith(("shut up", "get lost", "go die", "delete", "stop")):
            syntactic_style = "Direct Imperative Command"

        # 4. Severity Assessment
        if not is_bullying:
            severity = "None (Benign / Constructive)"
            severity_badge = "success"
        elif confidence > 85.0:
            severity = "High Severity (Explicit Hostility)"
            severity_badge = "danger"
        elif confidence > 70.0:
            severity = "Moderate Severity (Abrasive Language)"
            severity_badge = "warning"
        else:
            severity = "Low / Ambiguous Severity"
            severity_badge = "secondary"

        return {
            "target": target,
            "syntactic_style": syntactic_style,
            "severity": severity,
            "severity_badge": severity_badge,
            "hostile_keywords": extracted_hostile,
            "supportive_keywords": extracted_supportive,
            "keyword_count": len(extracted_hostile) + len(extracted_supportive)
        }
