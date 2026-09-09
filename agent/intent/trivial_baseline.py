"""
Baseline 1: Trivial Intent Classifier.
Uses simple majority class prediction with naive unigram keyword regexes.
"""

import re
from agent.config import IntentCategory
from agent.schemas import CustomerMessage, IntentClassificationResult


class TrivialIntentClassifier:
    """Trivial baseline predicting majority class or naive single-word matches."""

    def __init__(self, majority_class: IntentCategory = IntentCategory.OS_UPDATE_GLITCH):
        self.majority_class = majority_class
        self.keywords = {
            IntentCategory.OS_UPDATE_GLITCH: ["update", "ios", "macos", "version", "lag", "drain"],
            IntentCategory.HARDWARE_BATTERY_ISSUE: ["battery", "screen", "cracked", "broken", "drop", "water", "charge"],
            IntentCategory.ACCOUNT_BILLING_SUBSCRIPTION: ["bill", "charge", "refund", "subscription", "apple id", "password", "itunes"],
            IntentCategory.CONNECTIVITY_AUDIO_SYNC: ["airpod", "bluetooth", "wifi", "connect", "audio", "mic", "carplay"],
            IntentCategory.APP_FUNCTIONALITY_CRASH: ["app", "crash", "freeze", "safari", "camera", "storage"],
            IntentCategory.GENERAL_INQUIRY_POLICY: ["trade", "warranty", "applecare", "store", "buy", "policy"],
            IntentCategory.OUT_OF_SCOPE_CHITCHAT: ["love", "hello", "hi", "good", "thanks", "pixel", "samsung", "windows"]
        }

    def predict(self, message: CustomerMessage) -> IntentClassificationResult:
        text = message.text.lower()
        matched_intent = self.majority_class
        max_matches = 0

        for intent, kw_list in self.keywords.items():
            matches = sum(1 for kw in kw_list if re.search(r'\b' + re.escape(kw) + r'\b', text))
            if matches > max_matches:
                max_matches = matches
                matched_intent = intent

        confidence = 0.50 if max_matches > 0 else 0.25

        return IntentClassificationResult(
            intent=matched_intent,
            confidence=confidence,
            reasoning=f"Trivial keyword match (matches={max_matches})" if max_matches > 0 else "Trivial majority class default",
            probabilities={intent.value: (1.0 if intent == matched_intent else 0.0) for intent in IntentCategory}
        )
