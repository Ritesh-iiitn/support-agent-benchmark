"""
Deterministic Risk, Safety, and Compliance Rule Detector for AppleSupport Agent.
"""

import re
from typing import List, Tuple
from agent.config import UrgencyLevel, RoutingDepartment


class RiskRuleDetector:
    """Detects safety hazards, PII leaks, account compromises, and legal threats."""

    def __init__(self):
        # Regular expressions for sensitive PII
        self.credit_card_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
        self.ssn_pattern = re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b')
        self.password_pattern = re.compile(r'\b(?:password|passwd|pwd)[:= ]+([^\s]+)', re.IGNORECASE)

        # Keyword dictionaries with urgency and departments
        self.safety_hazard_keywords = [
            "swelling", "bulging", "smoke", "smoking", "spark", "sparked",
            "fire", "caught on fire", "burning", "burnt", "exploded",
            "melted", "shocked", "swallowed"
        ]

        self.security_fraud_keywords = [
            "hacked", "compromised", "account takeover", "ransom",
            "stolen apple id", "unauthorized charge", "fraudulent",
            "ex-partner tracking", "stalking"
        ]

        self.legal_threat_keywords = [
            "lawyer", "attorney", "lawsuit", "sue you", "taking you to court",
            "consumer court", "legal action", "police report"
        ]

        self.hardware_failure_codes = [
            "error 4013", "error 9", "error 4005", "greyed out bluetooth",
            "baseband failed", "nand error"
        ]

    def evaluate_risks(self, text: str) -> List[Tuple[str, UrgencyLevel, RoutingDepartment, str]]:
        """
        Returns a list of detected risk tuples:
        (Risk Name, Urgency Level, Routing Department, Human Reason)
        """
        findings = []
        t = text.lower()

        # 1. Critical Physical Safety Hazards
        for kw in self.safety_hazard_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', t):
                findings.append((
                    "CRITICAL_SAFETY_HAZARD",
                    UrgencyLevel.CRITICAL,
                    RoutingDepartment.EXECUTIVE_CUSTOMER_RELATIONS,
                    f"Physical safety hazard detected ('{kw}'). High risk of battery thermal runaway or injury."
                ))
                break

        # 2. PII Exposure in Public Tweet
        if self.credit_card_pattern.search(text) or self.ssn_pattern.search(text) or self.password_pattern.search(text):
            findings.append((
                "PII_LEAK_PUBLIC_TWEET",
                UrgencyLevel.HIGH,
                RoutingDepartment.BILLING_AND_SECURITY,
                "Customer posted raw credit card, SSN, or password in a public tweet. Immediate DM handoff required."
            ))

        # 3. Account Takeover / Severe Fraud
        for kw in self.security_fraud_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', t):
                findings.append((
                    "ACCOUNT_COMPROMISE_FRAUD",
                    UrgencyLevel.HIGH,
                    RoutingDepartment.BILLING_AND_SECURITY,
                    f"Account takeover or financial fraud detected ('{kw}'). Requires identity verification by security team."
                ))
                break

        # 4. Legal / Regulatory Threats
        for kw in self.legal_threat_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', t):
                findings.append((
                    "LEGAL_THREAT",
                    UrgencyLevel.HIGH,
                    RoutingDepartment.EXECUTIVE_CUSTOMER_RELATIONS,
                    f"Customer expressed intent for legal litigation ('{kw}'). Route to executive relations."
                ))
                break

        # 5. Severe Hardware Failure / Error Codes
        for code in self.hardware_failure_codes:
            if code in t:
                findings.append((
                    "HARDWARE_NAND_RESTORE_FAILURE",
                    UrgencyLevel.MEDIUM,
                    RoutingDepartment.SENIOR_TECHNICAL_ADVISOR,
                    f"Hardware board communication error code detected ('{code}'). Requires senior technical advisor."
                ))
                break

        return findings
