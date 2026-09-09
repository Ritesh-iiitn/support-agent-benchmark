import pytest
from agent.config import IntentCategory
from agent.schemas import CustomerMessage
from agent.intent import TrivialIntentClassifier, MLIntentClassifier, HybridIntentClassifier


def test_trivial_classifier():
    classifier = TrivialIntentClassifier()
    msg = CustomerMessage(text="Need help updating iOS version on my iPhone")
    res = classifier.predict(msg)
    assert res.intent in list(IntentCategory)
    assert 0.0 <= res.confidence <= 1.0


def test_ml_classifier():
    classifier = MLIntentClassifier()
    msg = CustomerMessage(text="My battery is draining fast after iOS update")
    res = classifier.predict(msg)
    assert res.intent == IntentCategory.OS_UPDATE_GLITCH
    assert res.confidence > 0.30


def test_hybrid_classifier_domain_rules():
    classifier = HybridIntentClassifier()
    
    # Critical Safety
    msg_safety = CustomerMessage(text="My battery is swelling and bulging!")
    res_safety = classifier.predict(msg_safety)
    assert res_safety.intent == IntentCategory.HARDWARE_BATTERY_ISSUE
    assert res_safety.confidence >= 0.90

    # Account / Refund
    msg_acct = CustomerMessage(text="Unauthorized charge on apple.com/bill request refund")
    res_acct = classifier.predict(msg_acct)
    assert res_acct.intent == IntentCategory.ACCOUNT_BILLING_SUBSCRIPTION

    # Trade-in
    msg_trade = CustomerMessage(text="What is the trade-in value for iPhone 13 Pro?")
    res_trade = classifier.predict(msg_trade)
    assert res_trade.intent == IntentCategory.GENERAL_INQUIRY_POLICY
