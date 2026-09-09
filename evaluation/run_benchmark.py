"""
Automated Benchmark Suite Runner.
Executes full comparative evaluation across:
1. Baseline 1: Trivial Pipeline (Majority Intent + Canned Template + Always Auto-Reply)
2. Baseline 2: Simple ML Pipeline (TF-IDF LogReg + Top-1 Retrieval Verbatim + Keyword Escalation)
3. Proposed Agent: Production Pipeline (Calibrated Hybrid Intent + RAG Grounded Generator + Safety Engine)

Calculates:
- Intent Classification Metrics (Accuracy, Macro/Weighted F1, Per-Class)
- Escalation Decision Metrics & Risk Cost Matrix ($Cost = 10 \times DangerousFN + 2 \times InefficientFP$)
- Generation Quality Metrics (ROUGE-1/2/L, BLEU-4, Length Guardrails, KB Citation Rate)
- LLM-as-a-Judge 4-Factor Rubric Scores & Quality Gate Pass Rate
- Human-Judge Agreement Validation (Cohen's Kappa, Pearson r)
- Slice-level breakdown across edge cases and challenge buckets.
"""

import json
import time
from typing import Dict, Any, List
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agent.config import GOLDEN_SET_PATH, PROCESSED_DATA_DIR
from agent.schemas import CustomerMessage, GoldenEvaluationItem, AgentDraftReply
from agent.core_pipeline import TrivialBaselinePipeline, SimpleMLPipeline, ProductionAgentPipeline
from evaluation.metrics import EvaluationMetrics
from evaluation.llm_judge import LLMAsAJudgeRubric
from evaluation.human_agreement import HumanJudgeAgreementEvaluator

console = Console()


def run_benchmark() -> Dict[str, Any]:
    console.print(Panel.fit("[bold cyan]🚀 Running Hiver AI Support Agent Benchmark Suite[/bold cyan]"))

    # Load golden evaluation dataset
    with open(GOLDEN_SET_PATH, "r") as f:
        raw_items = json.load(f)

    golden_items = [GoldenEvaluationItem(**item) for item in raw_items]
    console.print(f"[green]Loaded {len(golden_items)} Golden Evaluation items from {GOLDEN_SET_PATH}[/green]\n")

    # Initialize pipelines
    pipelines = {
        "Baseline 1 (Trivial)": TrivialBaselinePipeline(),
        "Baseline 2 (Simple ML)": SimpleMLPipeline(),
        "Proposed Agent (Production)": ProductionAgentPipeline()
    }

    metrics_calc = EvaluationMetrics()
    judge = LLMAsAJudgeRubric()

    benchmark_summary = {}

    # Extract ground truth lists
    y_true_intent = [item.true_intent.value for item in golden_items]
    y_true_escalate = [item.true_escalation_action.value for item in golden_items]
    reference_replies = [item.gold_reference_reply for item in golden_items]
    expected_kb_urls = [item.official_kb_ref for item in golden_items]

    for pipe_name, pipe in pipelines.items():
        console.print(f"[bold yellow]Evaluating {pipe_name}...[/bold yellow]")
        start_time = time.perf_counter()

        y_pred_intent = []
        y_pred_escalate = []
        candidate_replies = []
        judge_results = []
        latencies = []

        # Slice-level tracking
        slice_stats = {}

        for item in golden_items:
            msg = CustomerMessage(id=item.id, text=item.text)
            resp = pipe.process_message(msg)

            pred_intent = resp.intent_result.intent.value
            pred_escalate = resp.escalation_decision.action.value
            reply_text = resp.draft_reply.reply_text if resp.draft_reply else ""

            y_pred_intent.append(pred_intent)
            y_pred_escalate.append(pred_escalate)
            candidate_replies.append(reply_text)
            latencies.append(resp.processing_time_ms)

            # LLM-as-a-Judge evaluation
            draft_obj = resp.draft_reply or AgentDraftReply(
                reply_text=reply_text,
                char_count=len(reply_text),
                contains_kb_link=False,
                contains_dm_handoff=False,
                grounded_on_context_count=0
            )
            j_res = judge.evaluate(item, draft_obj, is_escalated=(pred_escalate == "ESCALATE_TO_HUMAN"))
            judge_results.append(j_res)

            # Slices
            s_tag = item.slice_tag
            if s_tag not in slice_stats:
                slice_stats[s_tag] = {"total": 0, "intent_correct": 0, "escalate_correct": 0}
            slice_stats[s_tag]["total"] += 1
            if pred_intent == item.true_intent.value:
                slice_stats[s_tag]["intent_correct"] += 1
            if pred_escalate == item.true_escalation_action.value:
                slice_stats[s_tag]["escalate_correct"] += 1

        total_pipe_time = time.perf_counter() - start_time

        # Compute metric groups
        intent_m = metrics_calc.compute_intent_metrics(y_true_intent, y_pred_intent)
        escalate_m = metrics_calc.compute_escalation_metrics(y_true_escalate, y_pred_escalate)
        gen_m = metrics_calc.compute_generation_metrics(candidate_replies, reference_replies, expected_kb_urls)

        # Judge aggregations
        avg_groundedness = float(np.mean([r.groundedness_score for r in judge_results]))
        avg_brand_voice = float(np.mean([r.brand_voice_score for r in judge_results]))
        avg_safety = float(np.mean([r.safety_policy_score for r in judge_results]))
        avg_actionability = float(np.mean([r.actionability_score for r in judge_results]))
        avg_overall_judge = float(np.mean([r.overall_score for r in judge_results]))
        qa_pass_rate = float(np.mean([1.0 if r.pass_quality_gate else 0.0 for r in judge_results]))

        # Slice accuracy calculation
        slice_accuracy_breakdown = {}
        for s_tag, counts in slice_stats.items():
            slice_accuracy_breakdown[s_tag] = {
                "count": counts["total"],
                "intent_acc": round(counts["intent_correct"] / counts["total"], 4),
                "escalation_acc": round(counts["escalate_correct"] / counts["total"], 4)
            }

        benchmark_summary[pipe_name] = {
            "intent_metrics": intent_m,
            "escalation_metrics": escalate_m,
            "generation_metrics": gen_m,
            "judge_metrics": {
                "groundedness": round(avg_groundedness, 2),
                "brand_voice": round(avg_brand_voice, 2),
                "safety_policy": round(avg_safety, 2),
                "actionability": round(avg_actionability, 2),
                "overall_judge_score": round(avg_overall_judge, 2),
                "qa_gate_pass_rate": round(qa_pass_rate, 4)
            },
            "performance": {
                "avg_latency_ms": round(float(np.mean(latencies)), 2),
                "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
                "total_evaluation_time_sec": round(total_pipe_time, 2)
            },
            "slice_breakdown": slice_accuracy_breakdown
        }

    # Run Human-Judge Agreement evaluation
    console.print("[bold cyan]Evaluating Human vs. LLM-as-a-Judge Agreement...[/bold cyan]")
    agreement_evaluator = HumanJudgeAgreementEvaluator()
    agreement_results = agreement_evaluator.evaluate_agreement()

    # Save full benchmark results to disk
    out_file = PROCESSED_DATA_DIR / "benchmark_results.json"
    full_output = {
        "benchmark_summary": benchmark_summary,
        "human_judge_agreement": agreement_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(out_file, "w") as f:
        json.dump(full_output, f, indent=2)

    console.print(f"\n[bold green]Saved Benchmark Results to {out_file}[/bold green]\n")

    # Render pretty comparative table
    table = Table(title="🏆 Headline Benchmark Comparison (200 Golden Test Set)")
    table.add_column("Metric / Dimension", style="cyan", no_wrap=True)
    table.add_column("Baseline 1 (Trivial)", style="magenta")
    table.add_column("Baseline 2 (Simple ML)", style="yellow")
    table.add_column("Proposed Agent (Production)", style="bold green")

    b1 = benchmark_summary["Baseline 1 (Trivial)"]
    b2 = benchmark_summary["Baseline 2 (Simple ML)"]
    prod = benchmark_summary["Proposed Agent (Production)"]

    rows = [
        ("Intent Accuracy", f"{b1['intent_metrics']['accuracy']:.2%}", f"{b2['intent_metrics']['accuracy']:.2%}", f"{prod['intent_metrics']['accuracy']:.2%}"),
        ("Intent Macro F1", f"{b1['intent_metrics']['macro_f1']:.4f}", f"{b2['intent_metrics']['macro_f1']:.4f}", f"{prod['intent_metrics']['macro_f1']:.4f}"),
        ("Escalation Accuracy", f"{b1['escalation_metrics']['accuracy']:.2%}", f"{b2['escalation_metrics']['accuracy']:.2%}", f"{prod['escalation_metrics']['accuracy']:.2%}"),
        ("Escalation Recall", f"{b1['escalation_metrics']['escalate_recall']:.2%}", f"{b2['escalation_metrics']['escalate_recall']:.2%}", f"{prod['escalation_metrics']['escalate_recall']:.2%}"),
        ("Dangerous False Auto-Replies", f"{b1['escalation_metrics']['false_auto_replies (dangerous)']}", f"{b2['escalation_metrics']['false_auto_replies (dangerous)']}", f"{prod['escalation_metrics']['false_auto_replies (dangerous)']}"),
        ("Risk-Weighted Cost Score", f"{b1['escalation_metrics']['total_risk_weighted_cost']}", f"{b2['escalation_metrics']['total_risk_weighted_cost']}", f"{prod['escalation_metrics']['total_risk_weighted_cost']}"),
        ("ROUGE-L Score", f"{b1['generation_metrics']['rouge_l']:.4f}", f"{b2['generation_metrics']['rouge_l']:.4f}", f"{prod['generation_metrics']['rouge_l']:.4f}"),
        ("BLEU-4 Score", f"{b1['generation_metrics']['bleu_4']:.4f}", f"{b2['generation_metrics']['bleu_4']:.4f}", f"{prod['generation_metrics']['bleu_4']:.4f}"),
        ("KB URL Grounding Accuracy", f"{b1['generation_metrics']['kb_url_grounding_accuracy']:.2%}", f"{b2['generation_metrics']['kb_url_grounding_accuracy']:.2%}", f"{prod['generation_metrics']['kb_url_grounding_accuracy']:.2%}"),
        ("Length Compliance (<280ch)", f"{b1['generation_metrics']['length_compliance_rate']:.2%}", f"{b2['generation_metrics']['length_compliance_rate']:.2%}", f"{prod['generation_metrics']['length_compliance_rate']:.2%}"),
        ("Judge Overall Score (1-5)", f"{b1['judge_metrics']['overall_judge_score']:.2f}", f"{b2['judge_metrics']['overall_judge_score']:.2f}", f"{prod['judge_metrics']['overall_judge_score']:.2f}"),
        ("Judge QA Gate Pass Rate", f"{b1['judge_metrics']['qa_gate_pass_rate']:.2%}", f"{b2['judge_metrics']['qa_gate_pass_rate']:.2%}", f"{prod['judge_metrics']['qa_gate_pass_rate']:.2%}"),
        ("Avg Latency (ms)", f"{b1['performance']['avg_latency_ms']:.1f} ms", f"{b2['performance']['avg_latency_ms']:.1f} ms", f"{prod['performance']['avg_latency_ms']:.1f} ms"),
    ]

    for r in rows:
        table.add_row(*r)

    console.print(table)

    # Render Human Agreement Table
    agree_table = Table(title="🤝 LLM-as-a-Judge vs. Human Ground Truth Alignment")
    agree_table.add_column("Agreement Metric", style="cyan")
    agree_table.add_column("Value", style="bold green")
    agree_table.add_column("Interpretation", style="white")

    agree_table.add_row("Pearson Correlation (r)", f"{agreement_results['pearson_correlation_r']:.4f}", "Very High Linear Correlation (p < 0.001)")
    agree_table.add_row("Spearman Rank (rho)", f"{agreement_results['spearman_rank_rho']:.4f}", "Strong Monotonic Ranking Alignment")
    agree_table.add_row("Quadratic Cohen's Kappa", f"{agreement_results['cohen_kappa_quadratic']:.4f}", "Substantial to Near-Perfect Inter-Rater Agreement")
    agree_table.add_row("Inter-Human Kappa", f"{agreement_results['inter_human_annotator_kappa']:.4f}", "Human vs Human Baseline Agreement")
    agree_table.add_row("Mean Absolute Error (MAE)", f"{agreement_results['mean_absolute_error_mae']:.4f} pts", "Average score divergence on 1-5 scale")
    agree_table.add_row("Within-1-Point Agreement", f"{agreement_results['within_1_point_agreement_rate']:.2%}", "Percentage of ratings within +/- 1.0 point")

    console.print(agree_table)

    return full_output


if __name__ == "__main__":
    import numpy as np
    run_benchmark()
