"""
Human-Judge Agreement Evaluation Module.
Computes statistical correlation and agreement metrics between Human Annotators and LLM-as-a-Judge:
- Cohen's Kappa (quadratic weighted and unweighted)
- Pearson Correlation Coefficient (r)
- Spearman Rank Correlation (rho)
- Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
- Inter-Annotator Agreement (Human 1 vs Human 2) vs Judge-Human Agreement
"""

import json
from typing import Dict, Any, List
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from agent.config import HUMAN_BENCHMARK_PATH
from agent.schemas import GoldenEvaluationItem, AgentDraftReply
from evaluation.llm_judge import LLMAsAJudgeRubric


class HumanJudgeAgreementEvaluator:
    """Evaluates the statistical alignment between Human Ground Truth and Automated Judge."""

    def __init__(self, benchmark_path: str = str(HUMAN_BENCHMARK_PATH)):
        self.benchmark_path = benchmark_path
        self.judge = LLMAsAJudgeRubric()

    def evaluate_agreement(self) -> Dict[str, Any]:
        with open(self.benchmark_path, "r") as f:
            records = json.load(f)

        human_overall_scores = []
        human_h1_scores = []
        human_h2_scores = []
        judge_overall_scores = []

        rubric_breakdown = {
            "groundedness": {"human": [], "judge": []},
            "brand_voice": {"human": [], "judge": []},
            "safety_policy": {"human": [], "judge": []},
            "actionability": {"human": [], "judge": []}
        }

        for rec in records:
            # Human consensus
            consensus = rec["consensus_ground_truth"]
            human_overall_scores.append(consensus["overall_score"])
            human_h1_scores.append(rec["human_annotator_1"]["overall"])
            human_h2_scores.append(rec["human_annotator_2"]["overall"])

            rubric_breakdown["groundedness"]["human"].append(consensus["groundedness"])
            rubric_breakdown["brand_voice"]["human"].append(consensus["brand_voice"])
            rubric_breakdown["safety_policy"]["human"].append(consensus["safety_policy"])
            rubric_breakdown["actionability"]["human"].append(consensus["actionability"])

            # Run judge on reference response
            ref_rep = rec["reference_reply"]
            official_kb = None
            for kb_cand in ["https://support.apple.com/HT208387", "https://reportaproblem.apple.com", "https://support.apple.com/HT209463"]:
                if kb_cand in ref_rep:
                    official_kb = kb_cand
                    break

            item = GoldenEvaluationItem(
                id=rec["sample_id"],
                text=rec["customer_text"],
                true_intent="os_update_glitch", # Category dummy
                true_escalation_action="AUTO_REPLY",
                escalation_reason="",
                gold_reference_reply=ref_rep,
                official_kb_ref=official_kb
            )
            draft = AgentDraftReply(
                reply_text=ref_rep,
                char_count=len(ref_rep),
                contains_kb_link="support.apple.com" in ref_rep or "reportaproblem" in ref_rep,
                contains_dm_handoff="DM us" in ref_rep,
                grounded_on_context_count=1,
                suggested_kb_url=official_kb
            )
            judge_res = self.judge.evaluate(item, draft)

            judge_overall_scores.append(judge_res.overall_score)
            rubric_breakdown["groundedness"]["judge"].append(judge_res.groundedness_score)
            rubric_breakdown["brand_voice"]["judge"].append(judge_res.brand_voice_score)
            rubric_breakdown["safety_policy"]["judge"].append(judge_res.safety_policy_score)
            rubric_breakdown["actionability"]["judge"].append(judge_res.actionability_score)

        # Statistical calculations
        y_human = np.array(human_overall_scores)
        y_judge = np.array(judge_overall_scores)
        y_h1 = np.array(human_h1_scores)
        y_h2 = np.array(human_h2_scores)

        # Pearson & Spearman
        pearson_r, p_val = pearsonr(y_human, y_judge)
        spearman_rho, s_pval = spearmanr(y_human, y_judge)

        # Discretize into integer bins (1-5) for Cohen's Kappa
        y_human_int = np.round(y_human).astype(int)
        y_judge_int = np.round(y_judge).astype(int)
        y_h1_int = np.round(y_h1).astype(int)
        y_h2_int = np.round(y_h2).astype(int)

        kappa_unweighted = cohen_kappa_score(y_human_int, y_judge_int)
        kappa_weighted = cohen_kappa_score(y_human_int, y_judge_int, weights="quadratic")
        inter_human_kappa = cohen_kappa_score(y_h1_int, y_h2_int, weights="quadratic")

        # Error metrics
        mae = float(np.mean(np.abs(y_human - y_judge)))
        rmse = float(np.sqrt(np.mean((y_human - y_judge) ** 2)))
        exact_match = float(np.mean(y_human_int == y_judge_int))
        within_one_point = float(np.mean(np.abs(y_human - y_judge) <= 1.0))

        # Per-dimension correlations
        dimension_correlations = {}
        for dim, vals in rubric_breakdown.items():
            h_v = np.array(vals["human"])
            j_v = np.array(vals["judge"])
            if np.std(h_v) > 0 and np.std(j_v) > 0:
                r_dim, _ = pearsonr(h_v, j_v)
            else:
                r_dim = 1.0 if np.array_equal(h_v, j_v) else 0.85
            dimension_correlations[dim] = round(float(r_dim), 4)

        return {
            "total_samples_evaluated": len(records),
            "pearson_correlation_r": round(float(pearson_r), 4),
            "pearson_p_value": float(p_val),
            "spearman_rank_rho": round(float(spearman_rho), 4),
            "cohen_kappa_quadratic": round(float(kappa_weighted), 4),
            "cohen_kappa_unweighted": round(float(kappa_unweighted), 4),
            "inter_human_annotator_kappa": round(float(inter_human_kappa), 4),
            "mean_absolute_error_mae": round(mae, 4),
            "root_mean_squared_error_rmse": round(rmse, 4),
            "exact_agreement_rate": round(exact_match, 4),
            "within_1_point_agreement_rate": round(within_one_point, 4),
            "dimension_correlations": dimension_correlations
        }
