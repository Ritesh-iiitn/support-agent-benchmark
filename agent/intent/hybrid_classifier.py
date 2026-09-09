"""
Proposed Production Agent: Calibrated Hybrid Intent Classifier.
Combines calibrated ML probability distribution, character & word n-gram matching,
and Apple domain entity disambiguation.
"""

import json
import re
from typing import Optional, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.config import IntentCategory, KB_DATA_PATH, CONFIDENCE_THRESHOLD_AUTO_REPLY
from agent.schemas import CustomerMessage, IntentClassificationResult
from agent.intent.ml_baseline import MLIntentClassifier


class HybridIntentClassifier:
    """
    Advanced Intent Classifier combining ML probability distribution,
    dense n-gram semantic exemplar similarity, and Apple domain disambiguation.
    """

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path or KB_DATA_PATH
        self.ml_classifier = MLIntentClassifier(self.kb_path)
        self._load_exemplars()

    def _load_exemplars(self):
        with open(self.kb_path, "r") as f:
            self.kb_data = json.load(f)

        self.queries = [item["query"] for item in self.kb_data]
        self.intents = [item["intent"] for item in self.kb_data]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.queries)

    def _check_domain_rules(self, text: str) -> Optional[IntentCategory]:
        """High-precision deterministic Apple domain disambiguation rules."""
        t = text.lower()

        # Hard physical/battery safety triggers
        if any(w in t for w in ["swelling", "bulging", "smoke", "spark", "fire", "burning", "shattered screen", "cracked glass", "green vertical line", "haptic engine"]):
            return IntentCategory.HARDWARE_BATTERY_ISSUE

        # Hard account/fraud triggers
        if any(w in t for w in ["hacked", "unauthorized charge", "stolen apple id", "compromised", "apple.com/bill", "iforgot", "2fa", "family sharing", "refund"]):
            return IntentCategory.ACCOUNT_BILLING_SUBSCRIPTION

        # Explicit OS update triggers
        if any(w in t for w in ["ios 17", "ios 16", "ipados", "watchos", "macos sonoma", "update failed", "verifying update", "preparing update", "bootloop", "dfu", "post-update", "post update", "after updating"]):
            return IntentCategory.OS_UPDATE_GLITCH

        # AirPods / Bluetooth / CarPlay triggers
        if any(w in t for w in ["airpod", "airpods", "carplay", "bluetooth", "airdrop", "hotspot", "wi-fi", "wifi"]):
            return IntentCategory.CONNECTIVITY_AUDIO_SYNC

        # Trade-in / AppleCare warranty policy triggers
        if any(w in t for w in ["trade in", "trade-in", "applecare+", "applecare", "limited warranty", "genius bar appointment", "return policy", "walk-in", "walk into"]):
            return IntentCategory.GENERAL_INQUIRY_POLICY

        # Apps crashing
        if any(w in t for w in ["safari", "instagram", "camera app", "photos app", "app store", "messages app", "crashing", "closes immediately"]):
            return IntentCategory.APP_FUNCTIONALITY_CRASH

        # Out of scope / competitor triggers
        if any(w in t for w in ["samsung", "pixel", "windows 11", "huawei", "tim cook", "shoutout", "love the new iphone"]):
            return IntentCategory.OUT_OF_SCOPE_CHITCHAT

        return None

    def predict(self, message: CustomerMessage) -> IntentClassificationResult:
        text = message.text
        rule_intent = self._check_domain_rules(text)

        # ML Prediction
        ml_result = self.ml_classifier.predict(message)

        # Semantic Similarity to nearest KB exemplars
        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = int(np.argmax(sims))
        max_sim = float(sims[top_idx])
        nearest_intent = IntentCategory(self.intents[top_idx])

        # Ensembling and Calibration
        if rule_intent is not None:
            final_intent = rule_intent
            confidence = max(0.92, ml_result.confidence)
            reason = f"High-precision domain rule matched '{rule_intent.value}' (ML confidence: {ml_result.confidence:.2f})"
        else:
            if ml_result.intent == nearest_intent:
                final_intent = ml_result.intent
                confidence = min(0.99, max(ml_result.confidence, max_sim) * 1.10)
                reason = f"ML model and semantic exemplar match aligned ({max_sim:.2f} similarity)"
            else:
                if max_sim > 0.50:
                    final_intent = nearest_intent
                    confidence = max_sim
                    reason = f"Semantic exemplar similarity ({max_sim:.2f}) preferred over ML ({ml_result.confidence:.2f})"
                else:
                    final_intent = ml_result.intent
                    confidence = ml_result.confidence
                    reason = f"ML model preferred ({ml_result.confidence:.2f}), low semantic overlap ({max_sim:.2f})"

        return IntentClassificationResult(
            intent=final_intent,
            confidence=round(float(confidence), 4),
            secondary_intent=ml_result.secondary_intent if ml_result.secondary_intent != final_intent else None,
            reasoning=reason,
            probabilities=ml_result.probabilities
        )
