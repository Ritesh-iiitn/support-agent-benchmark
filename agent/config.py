"""
System configuration and operational hyperparameters for the AppleSupport AI Agent.
"""

from enum import Enum
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.json"
HUMAN_BENCHMARK_PATH = DATA_DIR / "human_judge_benchmark.json"
KB_DATA_PATH = PROCESSED_DATA_DIR / "apple_support_kb.json"

# Brand Framing
BRAND_HANDLE = "@AppleSupport"
BRAND_NAME = "Apple Support"
TWITTER_CHAR_LIMIT = 280

# Operational Intents
class IntentCategory(str, Enum):
    OS_UPDATE_GLITCH = "os_update_glitch"
    HARDWARE_BATTERY_ISSUE = "hardware_battery_issue"
    ACCOUNT_BILLING_SUBSCRIPTION = "account_billing_subscription"
    CONNECTIVITY_AUDIO_SYNC = "connectivity_audio_sync"
    APP_FUNCTIONALITY_CRASH = "app_functionality_crash"
    GENERAL_INQUIRY_POLICY = "general_inquiry_policy"
    OUT_OF_SCOPE_CHITCHAT = "out_of_scope_chitchat"

INTENT_DESCRIPTIONS = {
    IntentCategory.OS_UPDATE_GLITCH: "Issues related to iOS/macOS/watchOS updates, battery drain after update, OS lag, update installation loops, or beta software bugs.",
    IntentCategory.HARDWARE_BATTERY_ISSUE: "Physical damage, cracked screens, swollen batteries, water damage, charging port failure, physical button defects, or severe device overheating.",
    IntentCategory.ACCOUNT_BILLING_SUBSCRIPTION: "Apple ID lockouts, two-factor authentication issues, unrecognized App Store / iTunes charges, subscription cancellations, refunds, or iCloud storage quota.",
    IntentCategory.CONNECTIVITY_AUDIO_SYNC: "Bluetooth pairing issues, AirPods audio drops or microphone failure, Wi-Fi connectivity drops, AirDrop failures, or CarPlay sync problems.",
    IntentCategory.APP_FUNCTIONALITY_CRASH: "Native or third-party apps crashing (Safari, Photos, Mail, Camera, Instagram), app freezing, storage full errors, or app store download errors.",
    IntentCategory.GENERAL_INQUIRY_POLICY: "Device trade-in valuation, AppleCare warranty coverage check, product compatibility, store appointment booking, or official policy questions.",
    IntentCategory.OUT_OF_SCOPE_CHITCHAT: "General greetings, spam, fan praise, insults without actionable support query, or non-Apple product questions (e.g. Android/Windows hardware)."
}

# Escalation Action
class EscalationAction(str, Enum):
    AUTO_REPLY = "AUTO_REPLY"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"

class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

# Escalation Routing Teams
class RoutingDepartment(str, Enum):
    TIER1_COMMUNITY_BOT = "tier1_community_bot"
    HARDWARE_GENIUS_BAR = "hardware_genius_bar"
    BILLING_AND_SECURITY = "billing_and_security"
    SENIOR_TECHNICAL_ADVISOR = "senior_technical_advisor"
    EXECUTIVE_CUSTOMER_RELATIONS = "executive_customer_relations"

# Thresholds & Calibration
# For 7 classes with uniform random prior 1/7 = 14.3%, confidence > 0.40 represents high posterior probability.
CONFIDENCE_THRESHOLD_AUTO_REPLY = 0.40
RETRIEVAL_SIMILARITY_THRESHOLD = 0.30
MAX_RETRIEVED_CONTEXTS = 3

# Cost Matrix for Escalation Evaluation
# Cost of False Auto Reply (Dangerous failure: bot gave bad/unsafe response to a critical issue)
COST_FALSE_AUTO_REPLY = 10.0
# Cost of False Escalation (Inefficiency cost: human agent unnecessarily handles simple query)
COST_FALSE_ESCALATION = 2.0
