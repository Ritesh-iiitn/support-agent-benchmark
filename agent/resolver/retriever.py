"""
Historical Resolution Knowledge Base Retriever.
Indexes verified AppleSupport resolutions and retrieves top-k relevant resolution contexts.
"""

import json
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.config import KB_DATA_PATH, IntentCategory, MAX_RETRIEVED_CONTEXTS
from agent.schemas import HistoricalContextMatch


class HistoricalKBRetriever:
    """Retrieves top-k historical AppleSupport resolutions matching the user's query."""

    def __init__(self, kb_path: Optional[str] = None):
        self.kb_path = kb_path or KB_DATA_PATH
        self.kb_records = []
        self._load_and_index()

    def _load_and_index(self):
        with open(self.kb_path, "r") as f:
            self.kb_records = json.load(f)

        self.queries = [r["query"] for r in self.kb_records]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.queries)

    def retrieve(
        self,
        query: str,
        filter_intent: Optional[IntentCategory] = None,
        top_k: int = MAX_RETRIEVED_CONTEXTS
    ) -> List[HistoricalContextMatch]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Rank indices by descending similarity
        ranked_indices = np.argsort(similarities)[::-1]

        matches: List[HistoricalContextMatch] = []
        seen_responses = set()

        for idx in ranked_indices:
            rec = self.kb_records[idx]
            sim = float(similarities[idx])
            rec_intent = IntentCategory(rec["intent"])

            # Filter by intent if requested
            if filter_intent is not None and rec_intent != filter_intent:
                continue

            # Deduplicate similar responses in top-k
            resp_text = rec["response"]
            if resp_text in seen_responses:
                continue
            seen_responses.add(resp_text)

            matches.append(HistoricalContextMatch(
                query=rec["query"],
                response=rec["response"],
                intent=rec_intent,
                similarity_score=round(sim, 4),
                kb_url=rec.get("kb_url")
            ))

            if len(matches) >= top_k:
                break

        return matches
