import pytest
from agent.config import TWITTER_CHAR_LIMIT
from agent.schemas import CustomerMessage
from agent.core_pipeline import ProductionAgentPipeline, SimpleMLPipeline, TrivialBaselinePipeline


def test_production_pipeline_e2e():
    pipeline = ProductionAgentPipeline()
    msg = CustomerMessage(text="My iPhone 14 battery drains so fast after iOS 17 update!")
    resp = pipeline.process_message(msg)

    assert resp.message_id == msg.id
    assert resp.intent_result.intent is not None
    assert resp.draft_reply is not None
    assert resp.draft_reply.char_count <= TWITTER_CHAR_LIMIT
    assert resp.processing_time_ms > 0.0


def test_all_pipelines_runnable():
    msg = CustomerMessage(text="Need help with my Apple device")
    for pipe in [TrivialBaselinePipeline(), SimpleMLPipeline(), ProductionAgentPipeline()]:
        resp = pipe.process_message(msg)
        assert resp.intent_result is not None
        assert resp.escalation_decision is not None
        assert resp.draft_reply is not None
