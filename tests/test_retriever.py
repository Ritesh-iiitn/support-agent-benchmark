import pytest
from agent.config import IntentCategory
from agent.resolver import HistoricalKBRetriever


def test_historical_retriever():
    retriever = HistoricalKBRetriever()
    matches = retriever.retrieve("Left AirPod won't connect or charge", top_k=3)
    assert len(matches) > 0
    assert matches[0].similarity_score > 0.0
    assert any("airpod" in m.query.lower() for m in matches)


def test_retriever_intent_filter():
    retriever = HistoricalKBRetriever()
    matches = retriever.retrieve(
        "battery life issue",
        filter_intent=IntentCategory.OS_UPDATE_GLITCH,
        top_k=2
    )
    for m in matches:
        assert m.intent == IntentCategory.OS_UPDATE_GLITCH
