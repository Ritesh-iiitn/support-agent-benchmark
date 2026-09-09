"""
Command Line Interface (CLI) for AppleSupport AI Agent.
Usage:
  python -m agent.cli query "My iPhone 14 battery is draining fast on iOS 17"
  python -m agent.cli interactive
  python -m agent.cli benchmark
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from agent.schemas import CustomerMessage
from agent.core_pipeline import ProductionAgentPipeline, SimpleMLPipeline, TrivialBaselinePipeline
from agent.config import EscalationAction
from evaluation.llm_judge import LLMAsAJudgeRubric
from evaluation.run_benchmark import run_benchmark

console = Console()


def display_agent_response(resp, show_judge: bool = True):
    console.print()
    console.print(Panel(
        f"[bold white]{resp.original_text}[/bold white]",
        title=f"[cyan]Incoming Customer Tweet (ID: {resp.message_id})[/cyan]",
        border_style="cyan"
    ))

    # Intent Table
    int_table = Table(title="🎯 Intent Classification", show_header=True, header_style="bold blue")
    int_table.add_column("Property", style="dim")
    int_table.add_column("Value", style="bold")
    int_table.add_row("Primary Intent", f"[green]{resp.intent_result.intent.value}[/green]")
    int_table.add_row("Confidence", f"{resp.intent_result.confidence:.2%}")
    if resp.intent_result.secondary_intent:
        int_table.add_row("Secondary Intent", resp.intent_result.secondary_intent.value)
    int_table.add_row("Reasoning", resp.intent_result.reasoning or "N/A")
    console.print(int_table)

    # Escalation Decision
    esc = resp.escalation_decision
    esc_color = "red" if esc.action == EscalationAction.ESCALATE_TO_HUMAN else "green"
    esc_table = Table(title="🛡️ Escalation & Safety Engine Decision", show_header=True, header_style="bold yellow")
    esc_table.add_column("Decision Dimension", style="dim")
    esc_table.add_column("Output", style="bold")
    esc_table.add_row("Action", f"[{esc_color}]{esc.action.value}[/{esc_color}]")
    esc_table.add_row("Urgency", f"[{esc_color}]{esc.urgency.value}[/{esc_color}]")
    esc_table.add_row("Routing Department", esc.routing_department.value)
    esc_table.add_row("Confidence", f"{esc.confidence:.2%}")
    esc_table.add_row("Reason", esc.reason)
    if esc.risk_flags:
        esc_table.add_row("Risk Flags", ", ".join(esc.risk_flags))
    console.print(esc_table)

    # Draft Reply
    if resp.draft_reply:
        rep = resp.draft_reply
        rep_panel = Panel(
            f"[bold white]{rep.reply_text}[/bold white]\n\n"
            f"[dim]Character Count: {rep.char_count}/280 | KB Link: {'✅' if rep.contains_kb_link else '❌'} | DM Prompt: {'✅' if rep.contains_dm_handoff else '❌'} | Grounded on: {rep.grounded_on_context_count} context(s)[/dim]",
            title="💬 Grounded Draft Reply (Apple Voice)",
            border_style="green"
        )
        console.print(rep_panel)

    # Retrieved Contexts
    if resp.retrieved_contexts:
        ctx_table = Table(title="📚 Top Retrieved Historical Resolutions (RAG)", show_header=True, header_style="bold magenta")
        ctx_table.add_column("#", style="dim", width=4)
        ctx_table.add_column("Similarity", width=12)
        ctx_table.add_column("Historical Resolution Snippet")
        for idx, ctx in enumerate(resp.retrieved_contexts, 1):
            ctx_table.add_row(
                str(idx),
                f"{ctx.similarity_score:.4f}",
                f"{ctx.response[:120]}..."
            )
        console.print(ctx_table)

    console.print(f"[dim]Processing Latency: {resp.processing_time_ms} ms[/dim]\n")


def interactive_mode():
    console.print(Panel.fit("[bold green]🤖 AppleSupport AI Agent Interactive Shell[/bold green]\nType a customer tweet below to inspect agent analysis. Type 'exit' to quit."))
    pipeline = ProductionAgentPipeline()

    while True:
        try:
            user_input = console.input("[bold cyan]Customer Tweet > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                break

            msg = CustomerMessage(id="msg_interactive", text=user_input)
            resp = pipeline.process_message(msg)
            display_agent_response(resp)
        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[yellow]Exiting interactive session. Goodbye![/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Support Agent CLI")
    subparsers = parser.add_subparsers(dest="command")

    query_parser = subparsers.add_parser("query", help="Process a single customer tweet")
    query_parser.add_argument("text", type=str, help="Customer tweet text")
    query_parser.add_argument("--pipeline", choices=["production", "ml", "trivial"], default="production", help="Pipeline version to run")

    subparsers.add_parser("interactive", help="Start interactive live terminal shell")
    subparsers.add_parser("benchmark", help="Run full 200-sample benchmark and human judge agreement suite")

    args = parser.parse_args()

    if args.command == "query":
        if args.pipeline == "trivial":
            pipe = TrivialBaselinePipeline()
        elif args.pipeline == "ml":
            pipe = SimpleMLPipeline()
        else:
            pipe = ProductionAgentPipeline()

        msg = CustomerMessage(id="msg_cli", text=args.text)
        resp = pipe.process_message(msg)
        display_agent_response(resp)

    elif args.command == "interactive":
        interactive_mode()

    elif args.command == "benchmark":
        run_benchmark()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
