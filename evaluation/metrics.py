"""
Evaluation metrics suite for Intent, Escalation, and Generation quality.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from agent.config import (
    IntentCategory, EscalationAction,
    COST_FALSE_AUTO_REPLY, COST_FALSE_ESCALATION,
    TWITTER_CHAR_LIMIT
)


class EvaluationMetrics:
    """Computes comprehensive automated metrics for customer support pipelines."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.smoother = SmoothingFunction().method1

    def compute_intent_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Calculates accuracy, macro/weighted precision, recall, F1, and per-class metrics."""
        unique_labels = sorted(list(set(y_true + y_pred)))
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

        p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=unique_labels, zero_division=0)

        per_class = {}
        for idx, lbl in enumerate(unique_labels):
            per_class[lbl] = {
                "precision": round(float(p_per[idx]), 4),
                "recall": round(float(r_per[idx]), 4),
                "f1": round(float(f1_per[idx]), 4),
                "support": int(sup_per[idx])
            }

        cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()

        return {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class": per_class,
            "confusion_matrix": {
                "labels": unique_labels,
                "matrix": cm
            }
        }

    def compute_escalation_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Calculates Escalation metrics and asymmetric Cost Matrix score:
        - False Auto-Reply (y_true=ESCALATE, y_pred=AUTO): Dangerous failure (Weight = 10.0)
        - False Escalation (y_true=AUTO, y_pred=ESCALATE): Inefficiency cost (Weight = 2.0)
        """
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            pos_label=EscalationAction.ESCALATE_TO_HUMAN.value,
            average="binary",
            zero_division=0
        )

        false_auto_reply_count = 0
        false_escalation_count = 0
        true_escalations = 0
        true_auto_replies = 0

        for yt, yp in zip(y_true, y_pred):
            if yt == EscalationAction.ESCALATE_TO_HUMAN.value and yp == EscalationAction.AUTO_REPLY.value:
                false_auto_reply_count += 1
            elif yt == EscalationAction.AUTO_REPLY.value and yp == EscalationAction.ESCALATE_TO_HUMAN.value:
                false_escalation_count += 1
            elif yt == EscalationAction.ESCALATE_TO_HUMAN.value and yp == EscalationAction.ESCALATE_TO_HUMAN.value:
                true_escalations += 1
            elif yt == EscalationAction.AUTO_REPLY.value and yp == EscalationAction.AUTO_REPLY.value:
                true_auto_replies += 1

        total_samples = len(y_true)
        total_cost = (false_auto_reply_count * COST_FALSE_AUTO_REPLY) + (false_escalation_count * COST_FALSE_ESCALATION)
        cost_per_query = total_cost / total_samples if total_samples > 0 else 0.0

        return {
            "accuracy": round(float(acc), 4),
            "escalate_precision": round(float(p), 4),
            "escalate_recall": round(float(r), 4),
            "escalate_f1": round(float(f1), 4),
            "false_auto_replies (dangerous)": false_auto_reply_count,
            "false_escalations (inefficient)": false_escalation_count,
            "total_risk_weighted_cost": round(total_cost, 2),
            "cost_per_query": round(cost_per_query, 4)
        }

    def compute_generation_metrics(
        self,
        candidate_replies: List[str],
        reference_replies: List[str],
        expected_kb_urls: List[Optional[str]]
    ) -> Dict[str, Any]:
        """Calculates ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, length compliance, and KB citation rate."""
        r1_scores = []
        r2_scores = []
        rl_scores = []
        bleu_scores = []
        length_violations = 0
        correct_kb_links = 0
        total_kb_expected = 0

        for cand, ref, exp_kb in zip(candidate_replies, reference_replies, expected_kb_urls):
            # Length guardrail check
            if len(cand) > TWITTER_CHAR_LIMIT:
                length_violations += 1

            # KB Citation check
            if exp_kb is not None:
                total_kb_expected += 1
                if exp_kb in cand:
                    correct_kb_links += 1

            # ROUGE
            scores = self.rouge_scorer.score(ref, cand)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rl_scores.append(scores["rougeL"].fmeasure)

            # BLEU
            ref_tokens = ref.lower().split()
            cand_tokens = cand.lower().split()
            b_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoother)
            bleu_scores.append(b_score)

        n = len(candidate_replies)
        kb_accuracy = (correct_kb_links / total_kb_expected) if total_kb_expected > 0 else 1.0

        return {
            "rouge_1": round(float(np.mean(r1_scores)), 4),
            "rouge_2": round(float(np.mean(r2_scores)), 4),
            "rouge_l": round(float(np.mean(rl_scores)), 4),
            "bleu_4": round(float(np.mean(bleu_scores)), 4),
            "length_compliance_rate": round(float(1.0 - (length_violations / n)), 4),
            "kb_url_grounding_accuracy": round(float(kb_accuracy), 4)
        }
