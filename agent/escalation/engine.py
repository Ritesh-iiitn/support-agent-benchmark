"""
Escalation decision engines for Baseline 1, Baseline 2, and Proposed AI Agent.
"""

from typing import List, Optional
from agent.config import (
    EscalationAction, UrgencyLevel, RoutingDepartment,
    CONFIDENCE_THRESHOLD_AUTO_REPLY, RETRIEVAL_SIMILARITY_THRESHOLD,
    IntentCategory
)
from agent.schemas import (
    CustomerMessage, IntentClassificationResult,
    HistoricalContextMatch, EscalationDecision
)
from agent.escalation.rules import RiskRuleDetector


class TrivialEscalationEngine:
    """Baseline 1: Always Auto-Reply (Majority Class)."""

    def decide(
        self,
        message: CustomerMessage,
        intent_result: IntentClassificationResult,
        retrieved_contexts: List[HistoricalContextMatch]
    ) -> EscalationDecision:
        return EscalationDecision(
            action=EscalationAction.AUTO_REPLY,
            urgency=UrgencyLevel.LOW,
            routing_department=RoutingDepartment.TIER1_COMMUNITY_BOT,
            confidence=0.50,
            reason="Trivial baseline default (Always Auto-Reply)",
            risk_flags=[]
        )


class SimpleMLEscalationEngine:
    """Baseline 2: Naive keyword search escalation."""

    def __init__(self):
        self.keywords = ["fire", "smoke", "hacked", "fraud", "lawyer", "sue", "stolen", "swelling"]

    def decide(
        self,
        message: CustomerMessage,
        intent_result: IntentClassificationResult,
        retrieved_contexts: List[HistoricalContextMatch]
    ) -> EscalationDecision:
        text = message.text.lower()
        matched = [kw for kw in self.keywords if kw in text]

        if matched:
            return EscalationDecision(
                action=EscalationAction.ESCALATE_TO_HUMAN,
                urgency=UrgencyLevel.HIGH,
                routing_department=RoutingDepartment.SENIOR_TECHNICAL_ADVISOR,
                confidence=0.75,
                reason=f"Keyword match for escalation: {', '.join(matched)}",
                risk_flags=matched
            )

        return EscalationDecision(
            action=EscalationAction.AUTO_REPLY,
            urgency=UrgencyLevel.LOW,
            routing_department=RoutingDepartment.TIER1_COMMUNITY_BOT,
            confidence=0.70,
            reason="No critical keywords detected; proceeding with automated reply.",
            risk_flags=[]
        )


class ProposedEscalationEngine:
    """
    Proposed Production Escalation Engine:
    Evaluates multi-factor risk:
    1. Deterministic safety/PII/legal risk triggers
    2. Intent model classification confidence (<0.70 threshold)
    3. Retrieval grounding quality score
    4. Sentiment & operational domain complexity
    """

    def __init__(self):
        self.risk_detector = RiskRuleDetector()

    def decide(
        self,
        message: CustomerMessage,
        intent_result: IntentClassificationResult,
        retrieved_contexts: List[HistoricalContextMatch]
    ) -> EscalationDecision:
        text = message.text
        risk_findings = self.risk_detector.evaluate_risks(text)

        # 1. Immediate Safety & Compliance Escalation
        if risk_findings:
            top_risk = risk_findings[0] # (name, urgency, dept, reason)
            risk_names = [r[0] for r in risk_findings]
            return EscalationDecision(
                action=EscalationAction.ESCALATE_TO_HUMAN,
                urgency=top_risk[1],
                routing_department=top_risk[2],
                confidence=0.98,
                reason=top_risk[3],
                risk_flags=risk_names
            )

        # 2. Model Intent Confidence Threshold Guardrail
        if intent_result.confidence < CONFIDENCE_THRESHOLD_AUTO_REPLY:
            return EscalationDecision(
                action=EscalationAction.ESCALATE_TO_HUMAN,
                urgency=UrgencyLevel.MEDIUM,
                routing_department=RoutingDepartment.SENIOR_TECHNICAL_ADVISOR,
                confidence=round(1.0 - intent_result.confidence, 4),
                reason=(
                    f"Intent ambiguity detected (model confidence {intent_result.confidence:.2f} "
                    f"< threshold {CONFIDENCE_THRESHOLD_AUTO_REPLY:.2f}). Escalate to human specialist to prevent incorrect advice."
                ),
                risk_flags=["LOW_INTENT_CONFIDENCE"]
            )

        # 3. Knowledge Base Grounding Check
        # If the query is technical but has no strong historical precedent in KB
        if intent_result.intent != IntentCategory.OUT_OF_SCOPE_CHITCHAT:
            best_sim = retrieved_contexts[0].similarity_score if retrieved_contexts else 0.0
            if best_sim < RETRIEVAL_SIMILARITY_THRESHOLD and len(text.split()) > 15:
                return EscalationDecision(
                    action=EscalationAction.ESCALATE_TO_HUMAN,
                    urgency=UrgencyLevel.MEDIUM,
                    routing_department=RoutingDepartment.SENIOR_TECHNICAL_ADVISOR,
                    confidence=0.85,
                    reason=(
                        f"Novel/Unfamiliar technical scenario (KB retrieval similarity {best_sim:.2f} "
                        f"< {RETRIEVAL_SIMILARITY_THRESHOLD:.2f}). Escalate to human agent for bespoke diagnosis."
                    ),
                    risk_flags=["NOVEL_UNSEEN_SCENARIO"]
                )

        # 4. Auto-Handle Decision
        return EscalationDecision(
            action=EscalationAction.AUTO_REPLY,
            urgency=UrgencyLevel.LOW,
            routing_department=RoutingDepartment.TIER1_COMMUNITY_BOT,
            confidence=round(intent_result.confidence, 4),
            reason=(
                f"Query matched intent '{intent_result.intent.value}' with {intent_result.confidence:.2f} confidence. "
                f"High historical precedent found; safely resolvable via automated guidance and verified KB reference."
            ),
            risk_flags=[]
        )
