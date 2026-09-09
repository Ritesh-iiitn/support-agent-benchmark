"""
Baseline 2: Simple ML Intent Classifier (TF-IDF + Calibrated Logistic Regression).
"""

import json
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agent.config import IntentCategory, KB_DATA_PATH
from agent.schemas import CustomerMessage, IntentClassificationResult


class MLIntentClassifier:
    """TF-IDF + Logistic Regression Intent Classifier."""

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path or KB_DATA_PATH
        self.model: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        self._train()

    def _train(self):
        """Trains the ML model on the historical KB resolution dataset."""
        with open(self.kb_path, "r") as f:
            kb_data = json.load(f)

        texts = [item["query"] for item in kb_data]
        labels = [item["intent"] for item in kb_data]

        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                sublinear_tf=True,
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=500,
                class_weight="balanced",
                random_state=42
            ))
        ])

        self.model.fit(texts, labels)
        self.classes_ = self.model.classes_

    def predict(self, message: CustomerMessage) -> IntentClassificationResult:
        text = message.text
        probs = self.model.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        predicted_intent_str = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        # Sort probabilities
        sorted_indices = np.argsort(probs)[::-1]
        secondary_intent_str = self.classes_[sorted_indices[1]] if len(sorted_indices) > 1 else None

        prob_dict = {self.classes_[i]: float(probs[i]) for i in range(len(self.classes_))}

        return IntentClassificationResult(
            intent=IntentCategory(predicted_intent_str),
            confidence=round(confidence, 4),
            secondary_intent=IntentCategory(secondary_intent_str) if secondary_intent_str else None,
            reasoning=f"TF-IDF Logistic Regression prediction (confidence: {confidence:.2%})",
            probabilities=prob_dict
        )
