import pytest
from agent.config import EscalationAction, UrgencyLevel
from agent.schemas import CustomerMessage, IntentClassificationResult
from agent.config import IntentCategory
from agent.escalation import ProposedEscalationEngine, RiskRuleDetector


def test_risk_rule_detector_safety():
    detector = RiskRuleDetector()
    risks = detector.evaluate_risks("My phone battery is smoking and bulging!")
    assert len(risks) > 0
    assert risks[0][0] == "CRITICAL_SAFETY_HAZARD"
    assert risks[0][1] == UrgencyLevel.CRITICAL


def test_risk_rule_detector_pii():
    detector = RiskRuleDetector()
    risks = detector.evaluate_risks("My card is 4111 2222 3333 4444 why was I billed?")
    assert len(risks) > 0
    assert risks[0][0] == "PII_LEAK_PUBLIC_TWEET"


def test_escalation_engine_auto_vs_escalate():
    engine = ProposedEscalationEngine()

    # Normal auto-reply query
    msg_auto = CustomerMessage(text="How do I check my AppleCare warranty?")
    intent_auto = IntentClassificationResult(intent=IntentCategory.GENERAL_INQUIRY_POLICY, confidence=0.95)
    dec_auto = engine.decide(msg_auto, intent_auto, [])
    assert dec_auto.action == EscalationAction.AUTO_REPLY

    # Hazardous escalation query
    msg_esc = CustomerMessage(text="My charger sparked and melted!")
    intent_esc = IntentClassificationResult(intent=IntentCategory.HARDWARE_BATTERY_ISSUE, confidence=0.90)
    dec_esc = engine.decide(msg_esc, intent_esc, [])
    assert dec_esc.action == EscalationAction.ESCALATE_TO_HUMAN
