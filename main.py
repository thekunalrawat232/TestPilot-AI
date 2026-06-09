#!/usr/bin/env python3
"""AI Test Agent — Entry point.

Usage:
    python main.py "As a user I can log in with email and password"
    python main.py --rebuild-rag "feature description here"
    python main.py --retries 5 "feature description here"
    python main.py --resume <run_id>              # resume a crashed run
    python main.py --resume latest                # resume the most recent run
    python main.py --list-checkpoints             # show available checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from config.settings import paths, exec_config

console = Console()


def _print_header() -> None:
    console.print(Panel.fit(
        "[bold cyan]AI Test Agent[/bold cyan] — Multi-Agent QA Pipeline\n"
        "LangGraph · RAG · Playwright",
        border_style="cyan",
    ))


def _print_stage(name: str, status: str = "running") -> None:
    icons = {"running": "⏳", "done": "✅", "error": "❌"}
    console.print(f"\n{icons.get(status, '•')} [bold]{name}[/bold]")


def _print_state_summary(state: dict) -> None:
    """Print a rich summary of the final pipeline state."""
    console.print("\n")
    console.rule("[bold green]Pipeline Complete[/bold green]")

    # Requirement analysis summary
    ra = state.get("requirement_analysis", {})
    if ra.get("feature_name"):
        console.print(f"  Feature: [cyan]{ra['feature_name']}[/cyan]")
        console.print(f"  Summary: {ra.get('summary', 'N/A')}")

    # Test plan summary (new schema: a flat list of tests)
    tp = state.get("test_plan", {})
    tests = tp.get("tests", [])
    console.print(f"  Tests planned: {len(tests)}")

    # Generated files
    gc = state.get("generated_code", {})
    written = gc.get("written_files", [])
    if written:
        console.print(f"  Generated files: {len(written)}")
        for f in written:
            console.print(f"    → {f}")

    # Execution result
    er = state.get("execution_result", {})
    all_passed = er.get("all_passed", False)
    status_str = "[bold green]ALL PASSED[/bold green]" if all_passed else "[bold red]FAILURES[/bold red]"
    console.print(f"  Execution: {status_str}")
    console.print(f"  Retries used: {state.get('retry_count', 0)} / {state.get('max_retries', 3)}")

    # Bug reports
    da = state.get("debug_analysis", {})
    bugs = da.get("bug_reports", [])
    if bugs:
        console.print(f"\n  [bold red]Real Bugs Found: {len(bugs)}[/bold red]")
        for b in bugs:
            console.print(f"    • [{b.get('severity', '?')}] {b.get('title', 'Untitled')}")

    # Errors
    errors = state.get("error_log", [])
    if errors:
        console.print(f"\n  [yellow]Pipeline errors:[/yellow]")
        for e in errors:
            console.print(f"    ⚠ {e}")

    console.print()


def _handle_list_checkpoints() -> None:
    """Print a table of available checkpoints."""
    from agents.checkpoint import list_checkpoints

    checkpoints = list_checkpoints()
    if not checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    table = Table(title="Available Checkpoints")
    table.add_column("Run ID", style="cyan")
    table.add_column("Last Node", style="green")
    table.add_column("Feature", style="white")
    table.add_column("Saved At", style="dim")

    for cp in checkpoints:
        table.add_row(
            cp["run_id"],
            cp["last_completed_node"],
            cp["feature"],
            cp["timestamp"],
        )

    console.print(table)
    console.print("\nTo resume: [bold]python main.py --resume <run_id>[/bold]")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Test Agent — Multi-Agent QA Pipeline")
    parser.add_argument("requirement", nargs="?", default=None, help="Feature requirement or user story text")
    parser.add_argument("--retries", type=int, default=None, help="Max debug retry attempts")
    parser.add_argument("--rebuild-rag", action="store_true", help="Force rebuild the RAG vector store")
    parser.add_argument("--resume", type=str, default=None, metavar="RUN_ID",
                        help="Resume a crashed/stopped pipeline run. Use 'latest' for the most recent.")
    parser.add_argument("--list-checkpoints", action="store_true", help="List all saved pipeline checkpoints")
    args = parser.parse_args()

    _print_header()

    # List checkpoints mode
    if args.list_checkpoints:
        _handle_list_checkpoints()
        return

    # Resume mode
    resume_run_id = None
    if args.resume:
        from agents.checkpoint import load_checkpoint, get_latest_checkpoint

        if args.resume == "latest":
            checkpoint = get_latest_checkpoint()
            if not checkpoint:
                console.print("[bold red]No checkpoints found to resume.[/bold red]")
                sys.exit(1)
            resume_run_id = checkpoint["run_id"]
        else:
            resume_run_id = args.resume
            checkpoint = load_checkpoint(resume_run_id)
            if not checkpoint:
                console.print(f"[bold red]No checkpoint found for run_id '{resume_run_id}'.[/bold red]")
                console.print("Use --list-checkpoints to see available runs.")
                sys.exit(1)

        console.print(f"\n[bold]Resuming run:[/bold] {resume_run_id}")
        console.print(f"[bold]Last completed node:[/bold] {checkpoint['last_completed_node']}")
        console.print(f"[bold]Saved at:[/bold] {checkpoint['timestamp']}")

        # Use the requirement from the checkpoint if not provided
        requirement = args.requirement or checkpoint["state"].get("raw_requirement", "")
        if not requirement:
            console.print("[bold red]No requirement found in checkpoint or arguments.[/bold red]")
            sys.exit(1)
    else:
        if not args.requirement:
            parser.error("requirement is required when not using --resume or --list-checkpoints")
        requirement = args.requirement

    # Optionally rebuild RAG index
    if args.rebuild_rag:
        _print_stage("Rebuilding RAG Vector Store")
        from rag.vectorstore import build_vectorstore
        build_vectorstore(force_rebuild=True)
        _print_stage("RAG Vector Store", "done")

    console.print(f"\n[bold]Requirement:[/bold] {requirement}")
    console.print(f"[bold]Max retries:[/bold] {args.retries or exec_config.max_retries}")
    console.print()

    # Run the pipeline
    from agents.graph import run_pipeline

    _print_stage("Running Pipeline")

    try:
        final_state = run_pipeline(
            requirement=requirement,
            max_retries=args.retries,
            resume_run_id=resume_run_id,
        )
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline crashed:[/bold red] {exc}")
        if resume_run_id:
            console.print(f"[yellow]State was checkpointed. Resume with:[/yellow]")
            console.print(f"  python main.py --resume {resume_run_id}")
        else:
            console.print("[yellow]Check --list-checkpoints for any saved state.[/yellow]")
        raise

    # The final_state from langgraph is a dict
    if isinstance(final_state, dict):
        state_dict = final_state
    else:
        state_dict = final_state.dict() if hasattr(final_state, "dict") else dict(final_state)

    _print_state_summary(state_dict)

    # Save full state as JSON for programmatic consumption
    reports_dir = paths.generated_reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    state_path = reports_dir / f"state_{ts}.json"

    # Serialize — handle non-serializable values gracefully
    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        return repr(obj)

    state_path.write_text(json.dumps(state_dict, indent=2, default=_default))
    console.print(f"Full state saved to: {state_path}")


if __name__ == "__main__":
    main()
