"""
End-to-End AI Support Agent Pipelines.
Exposes:
1. TrivialBaselinePipeline (Baseline 1)
2. SimpleMLPipeline (Baseline 2)
3. ProductionAgentPipeline (Proposed Agent)
"""

import time
from typing import Optional, List
from agent.config import EscalationAction, IntentCategory
from agent.schemas import CustomerMessage, AgentResponse, HistoricalContextMatch
from agent.intent import TrivialIntentClassifier, MLIntentClassifier, HybridIntentClassifier
from agent.resolver import HistoricalKBRetriever, GroundedReplyGenerator
from agent.resolver.generator import TrivialReplyGenerator, VerbatimMLReplyGenerator
from agent.escalation import (
    TrivialEscalationEngine,
    SimpleMLEscalationEngine,
    ProposedEscalationEngine
)


class TrivialBaselinePipeline:
    """Baseline 1: Majority Class Intent + Canned Template + Always Auto-Reply."""

    def __init__(self):
        self.intent_classifier = TrivialIntentClassifier()
        self.generator = TrivialReplyGenerator()
        self.escalation_engine = TrivialEscalationEngine()

    def process_message(self, message: CustomerMessage) -> AgentResponse:
        start_t = time.perf_counter()

        intent_res = self.intent_classifier.predict(message)
        escalation_dec = self.escalation_engine.decide(message, intent_res, [])
        draft_reply = self.generator.generate(message, intent_res.intent)

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return AgentResponse(
            message_id=message.id,
            original_text=message.text,
            intent_result=intent_res,
            escalation_decision=escalation_dec,
            draft_reply=draft_reply,
            retrieved_contexts=[],
            processing_time_ms=round(duration_ms, 2)
        )


class SimpleMLPipeline:
    """Baseline 2: TF-IDF Logistic Regression + Verbatim Top-1 Retrieval + Heuristic Escalation."""

    def __init__(self, kb_path: Optional[str] = None):
        self.intent_classifier = MLIntentClassifier(kb_path)
        self.retriever = HistoricalKBRetriever(kb_path)
        self.generator = VerbatimMLReplyGenerator()
        self.escalation_engine = SimpleMLEscalationEngine()

    def process_message(self, message: CustomerMessage) -> AgentResponse:
        start_t = time.perf_counter()

        intent_res = self.intent_classifier.predict(message)
        retrieved_contexts = self.retriever.retrieve(message.text, filter_intent=intent_res.intent, top_k=2)
        escalation_dec = self.escalation_engine.decide(message, intent_res, retrieved_contexts)
        draft_reply = self.generator.generate(message, intent_res.intent, retrieved_contexts)

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return AgentResponse(
            message_id=message.id,
            original_text=message.text,
            intent_result=intent_res,
            escalation_decision=escalation_dec,
            draft_reply=draft_reply,
            retrieved_contexts=retrieved_contexts,
            processing_time_ms=round(duration_ms, 2)
        )


class ProductionAgentPipeline:
    """
    Proposed Production Pipeline:
    Hybrid Calibrated Intent Classification + Contextual RAG Synthesis + Multi-factor Safety & Escalation.
    """

    def __init__(self, kb_path: Optional[str] = None):
        self.intent_classifier = HybridIntentClassifier(kb_path)
        self.retriever = HistoricalKBRetriever(kb_path)
        self.generator = GroundedReplyGenerator()
        self.escalation_engine = ProposedEscalationEngine()

    def process_message(self, message: CustomerMessage) -> AgentResponse:
        start_t = time.perf_counter()

        # Step 1: Intent Classification & Calibration
        intent_res = self.intent_classifier.predict(message)

        # Step 2: Historical Resolution Retrieval
        retrieved_contexts = self.retriever.retrieve(
            message.text,
            filter_intent=intent_res.intent,
            top_k=3
        )

        # Step 3: Safety & Escalation Decision
        escalation_dec = self.escalation_engine.decide(
            message,
            intent_res,
            retrieved_contexts
        )

        # Step 4: Grounded Reply Generation
        force_dm = (
            escalation_dec.action == EscalationAction.ESCALATE_TO_HUMAN or
            "PII" in escalation_dec.reason or
            intent_res.intent == IntentCategory.ACCOUNT_BILLING_SUBSCRIPTION
        )
        draft_reply = self.generator.generate(
            message=message,
            intent=intent_res.intent,
            retrieved_contexts=retrieved_contexts,
            force_dm=force_dm
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0

        return AgentResponse(
            message_id=message.id,
            original_text=message.text,
            intent_result=intent_res,
            escalation_decision=escalation_dec,
            draft_reply=draft_reply,
            retrieved_contexts=retrieved_contexts,
            processing_time_ms=round(duration_ms, 2)
        )
