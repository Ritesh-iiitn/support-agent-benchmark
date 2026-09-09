"""
Pydantic data models for the Hiver AI Support Agent.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from agent.config import IntentCategory, EscalationAction, UrgencyLevel, RoutingDepartment


class CustomerMessage(BaseModel):
    id: str = Field(default="msg_custom", description="Unique message ID")
    text: str = Field(..., description="Raw text of the customer tweet")
    author_handle: Optional[str] = Field(default="@user", description="Author Twitter handle")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of tweet")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list, description="Prior conversation context")


class IntentClassificationResult(BaseModel):
    intent: IntentCategory
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated confidence score")
    secondary_intent: Optional[IntentCategory] = None
    reasoning: Optional[str] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)


class HistoricalContextMatch(BaseModel):
    query: str
    response: str
    intent: IntentCategory
    similarity_score: float
    kb_url: Optional[str] = None


class EscalationDecision(BaseModel):
    action: EscalationAction
    urgency: UrgencyLevel = UrgencyLevel.LOW
    routing_department: RoutingDepartment = RoutingDepartment.TIER1_COMMUNITY_BOT
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    risk_flags: List[str] = Field(default_factory=list)


class AgentDraftReply(BaseModel):
    reply_text: str
    char_count: int
    contains_kb_link: bool = False
    contains_dm_handoff: bool = False
    grounded_on_context_count: int = 0
    suggested_kb_url: Optional[str] = None


class AgentResponse(BaseModel):
    message_id: str
    original_text: str
    intent_result: IntentClassificationResult
    escalation_decision: EscalationDecision
    draft_reply: Optional[AgentDraftReply] = None
    retrieved_contexts: List[HistoricalContextMatch] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class GoldenEvaluationItem(BaseModel):
    id: str
    text: str
    true_intent: IntentCategory
    true_escalation_action: EscalationAction
    escalation_reason: str
    gold_reference_reply: str
    difficulty_tier: str = "Standard" # "Easy", "Standard", "Hard", "Adversarial"
    slice_tag: str = "General" # "Short_Text", "High_Frustration", "PII_Risk", "OOD", "Multi_Intent", "Hardware_Damage"
    official_kb_ref: Optional[str] = None


class JudgeEvaluationResult(BaseModel):
    sample_id: str
    groundedness_score: int = Field(ge=1, le=5, description="Factual correctness and relevance to Apple KB")
    brand_voice_score: int = Field(ge=1, le=5, description="Empathy, politeness, clarity, and Apple tone")
    safety_policy_score: int = Field(ge=1, le=5, description="Adherence to security, privacy (DM for PII), and safe escalation")
    actionability_score: int = Field(ge=1, le=5, description="Clear troubleshooting steps, conciseness, actionable advice")
    overall_score: float = Field(ge=1.0, le=5.0)
    judge_rationale: str
    pass_quality_gate: bool
