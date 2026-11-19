"""
Explainable AI (XAI) Token-Level Feature Attribution Engine
===========================================================
Computes local token attribution weights for linear models (Logistic Regression,
Linear SVM) and permutation/occlusion attribution for Transformer models.

Produces:
  - Token-level signed attributions: positive (bullying trigger) vs negative (civil/mitigating)
  - Normalized relative percentage impact for each word in the input text
  - Visual badge metadata for rendering in the Web UI
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TokenExplainer:
    def __init__(self, predictor=None):
        self.predictor = predictor

    def explain(self, text: str, predictor=None) -> Dict[str, Any]:
        """
        Generates token-level feature attribution for the input text.
        Returns a dictionary with token breakdown, positive triggers, negative mitigators, and UI badges.
        """
        p = predictor or self.predictor
        if not p or not text or not text.strip():
            return {
                "available": False,
                "tokens": [],
                "top_positive": [],
                "top_negative": []
            }

        model_type = getattr(p, "model_type", "logistic_regression")
        vectorizer = getattr(p, "vectorizer", None)
        model = getattr(p, "model", None)

        # For linear models (Logistic Regression and Calibrated Linear SVM)
        if vectorizer is not None and model is not None and model_type in ("logistic_regression", "svm"):
            try:
                # Extract linear model coefficients
                # If CalibratedClassifierCV, average coefficients across calibrated base estimators
                if hasattr(model, "coef_"):
                    coef = model.coef_[0]
                elif hasattr(model, "calibrated_classifiers_"):
                    coefs = [clf.estimator.coef_[0] for clf in model.calibrated_classifiers_ if hasattr(clf.estimator, "coef_")]
                    coef = np.mean(coefs, axis=0) if coefs else None
                else:
                    coef = None

                if coef is not None:
                    # Tokenize the input text preserving word order
                    words = re.findall(r"\b\w+\b|[!?,]", text)
                    feature_names = vectorizer.get_feature_names_out()
                    vocab = vectorizer.vocabulary_

                    token_details = []
                    total_abs_weight = 0.0

                    for word in words:
                        w_clean = word.lower()
                        weight = 0.0
                        if w_clean in vocab:
                            idx = vocab[w_clean]
                            weight = float(coef[idx])
                        
                        abs_w = abs(weight)
                        total_abs_weight += abs_w

                        token_details.append({
                            "word": word,
                            "weight": round(weight, 4),
                            "abs_weight": abs_w,
                            "direction": "bullying" if weight > 0.05 else ("supportive" if weight < -0.05 else "neutral")
                        })

                    # Calculate relative percentage impact
                    total_div = total_abs_weight if total_abs_weight > 0 else 1.0
                    for t in token_details:
                        t["impact_pct"] = round((t["abs_weight"] / total_div) * 100, 1)

                    # Top positive (toxic) and top negative (supportive) words
                    sorted_pos = sorted([t for t in token_details if t["weight"] > 0], key=lambda x: x["weight"], reverse=True)
                    sorted_neg = sorted([t for t in token_details if t["weight"] < 0], key=lambda x: x["weight"])

                    return {
                        "available": True,
                        "model_type": model_type,
                        "tokens": token_details,
                        "top_positive": sorted_pos[:5],
                        "top_negative": sorted_neg[:5],
                        "total_features": len(token_details)
                    }
            except Exception as e:
                logger.warning(f"Error computing linear token attribution: {e}")

        # Fallback for DistilBERT or general case: Leave-one-word-out occlusion
        try:
            words = text.split()
            if len(words) <= 1:
                return {"available": False, "tokens": []}

            base_res = p.predict(text)
            base_prob = base_res.get("probabilities", {}).get("Bullying", 0.5)

            token_details = []
            total_abs = 0.0
            for i, word in enumerate(words):
                # Remove word i
                subset = " ".join([w for j, w in enumerate(words) if j != i])
                if not subset.strip():
                    continue
                sub_res = p.predict(subset)
                sub_prob = sub_res.get("probabilities", {}).get("Bullying", 0.5)
                # If removing the word drops bullying prob, word contributed positively to bullying
                diff = float(base_prob - sub_prob)
                total_abs += abs(diff)
                token_details.append({
                    "word": word,
                    "weight": round(diff, 4),
                    "abs_weight": abs(diff),
                    "direction": "bullying" if diff > 0.03 else ("supportive" if diff < -0.03 else "neutral")
                })

            total_div = total_abs if total_abs > 0 else 1.0
            for t in token_details:
                t["impact_pct"] = round((t["abs_weight"] / total_div) * 100, 1)

            sorted_pos = sorted([t for t in token_details if t["weight"] > 0], key=lambda x: x["weight"], reverse=True)
            sorted_neg = sorted([t for t in token_details if t["weight"] < 0], key=lambda x: x["weight"])

            return {
                "available": True,
                "model_type": model_type,
                "tokens": token_details,
                "top_positive": sorted_pos[:5],
                "top_negative": sorted_neg[:5],
                "total_features": len(token_details)
            }
        except Exception as e:
            logger.warning(f"Error in occlusion explainer: {e}")
            return {"available": False, "tokens": []}
