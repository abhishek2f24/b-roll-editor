"""
agent.py — the top-level orchestrator.

    python agent.py run --script my_script.txt
    python agent.py run --script my_script.txt --top-k 5 --output ujjain_reel
    python agent.py index
    python agent.py index --reindex
    python agent.py search --script my_script.txt
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

import config
import indexer
import searcher
import assembler

console = Console()
app = typer.Typer(add_completion=False, help="AI B-roll editor agent")


# ── Sub-commands ──────────────────────────────────────────────────────────────

@app.command()
def index(
    reindex: bool = typer.Option(False, "--reindex", help="Wipe and rebuild the index"),
):
    """Index all videos in the videos/ folder."""
    console.print(Panel("[bold]Step 1: Video Indexing[/bold]", style="cyan"))
    indexer.index_videos(reindex=reindex)


@app.command()
def search(
    script: Path = typer.Option(..., "--script", "-s", exists=True, help="Script .txt file"),
    top_k:  int  = typer.Option(3, "--top-k", "-k", help="Candidates per scene"),
):
    """Search the index for a script — show matches, don't render."""
    console.print(Panel("[bold]Scene Search[/bold]", style="cyan"))
    text    = script.read_text(encoding="utf-8")
    queries = searcher.script_to_queries(text)
    results = searcher.search(queries, top_k=top_k)
    searcher._print_results(queries, results)


@app.command()
def run(
    script:     Path = typer.Option(...,      "--script", "-s", exists=True),
    top_k:      int  = typer.Option(3,        "--top-k",  "-k"),
    output:     str  = typer.Option("reel",   "--output", "-o", help="Output filename stem"),
    skip_index: bool = typer.Option(False,    "--skip-index", help="Skip indexing step"),
):
    """Full pipeline: index → search → assemble."""
    console.print(Panel("[bold yellow]AI B-roll Editor Agent[/bold yellow]", style="yellow"))

    # 1. Index (if needed)
    if not skip_index:
        console.print("\n[bold cyan][ 1 / 3 ]  Indexing videos...[/bold cyan]")
        indexer.index_videos(reindex=False)
    else:
        console.print("\n[dim]Skipping index step (--skip-index)[/dim]")

    # 2. Search
    console.print("\n[bold cyan][ 2 / 3 ]  Finding matching scenes...[/bold cyan]")
    text    = script.read_text(encoding="utf-8")
    queries = searcher.script_to_queries(text)

    console.print(f"  Script has [yellow]{len(queries)}[/yellow] sentences → will find {top_k} candidates each")
    results = searcher.search(queries, top_k=top_k)

    # Show summary
    for q, matches in zip(queries, results):
        if matches:
            best = matches[0]
            console.print(
                f"  [green]✓[/green] [dim]{q.sentence[:55]}[/dim]"
                f" → [cyan]{best.video_path}[/cyan] @ {best.start_sec:.1f}s  (score {best.score:.3f})"
            )
        else:
            console.print(f"  [red]✗[/red] [dim]{q.sentence[:55]}[/dim] — no match found")

    # 3. Assemble
    console.print("\n[bold cyan][ 3 / 3 ]  Assembling timeline...[/bold cyan]")
    plan   = assembler.matches_to_plan(results)
    output_path = assembler.assemble(plan, output_name=output)

    console.print(Panel(
        f"[bold green]Export complete![/bold green]\n\n"
        f"File:   [cyan]{output_path}[/cyan]\n"
        f"Scenes: {len(plan)}\n\n"
        f"Next:  Record your voice-over and mix it in.",
        title="Done",
        style="green",
    ))


@app.command()
def export_plan(
    script: Path = typer.Option(..., "--script", "-s", exists=True),
    top_k:  int  = typer.Option(3, "--top-k"),
    out:    Path = typer.Option(Path("plan.json"), "--out"),
):
    """Export clip plan as JSON for manual editing before assembly."""
    text    = script.read_text(encoding="utf-8")
    queries = searcher.script_to_queries(text)
    results = searcher.search(queries, top_k=top_k)
    plan    = assembler.matches_to_plan(results)
    data    = [vars(s) for s in plan]
    out.write_text(json.dumps(data, indent=2))
    console.print(f"Plan written to [cyan]{out}[/cyan]  — edit then run `assembler.py --plan {out}`")


if __name__ == "__main__":
    app()
