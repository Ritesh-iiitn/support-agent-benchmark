import pytest
from agent.schemas import GoldenEvaluationItem, AgentDraftReply
from agent.config import IntentCategory, EscalationAction
from evaluation.llm_judge import LLMAsAJudgeRubric
from evaluation.human_agreement import HumanJudgeAgreementEvaluator


def test_llm_judge_scoring():
    judge = LLMAsAJudgeRubric()
    item = GoldenEvaluationItem(
        id="test_01",
        text="My battery is draining fast on iOS 17",
        true_intent=IntentCategory.OS_UPDATE_GLITCH,
        true_escalation_action=EscalationAction.AUTO_REPLY,
        escalation_reason="",
        gold_reference_reply="Thanks for reaching out! Check https://support.apple.com/HT208387",
        official_kb_ref="https://support.apple.com/HT208387"
    )
    draft = AgentDraftReply(
        reply_text="We'd like to help! Please check https://support.apple.com/HT208387 for tips.",
        char_count=75,
        contains_kb_link=True,
        contains_dm_handoff=False,
        grounded_on_context_count=1
    )
    res = judge.evaluate(item, draft)
    assert res.overall_score >= 4.0
    assert res.pass_quality_gate is True


def test_human_agreement_evaluator():
    evaluator = HumanJudgeAgreementEvaluator()
    results = evaluator.evaluate_agreement()
    assert results["total_samples_evaluated"] == 50
    assert results["pearson_correlation_r"] > 0.70
    assert results["mean_absolute_error_mae"] < 0.50
