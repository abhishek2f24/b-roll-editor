"""
searcher.py — takes a script (plain text), splits into scenes,
              and returns ranked video segments for each scene.

Usage:
    python searcher.py --script script.txt --top-k 3
    python searcher.py --query "people walking at sunrise temple"
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
import typer
from rich.console import Console
from rich.table import Table

import config

console = Console()
app = typer.Typer(add_completion=False)


@dataclass
class SceneQuery:
    sentence: str
    keywords: list[str]      # extracted by simple heuristics or LLM
    combined: str            # what actually gets embedded


@dataclass
class Match:
    scene_query:    SceneQuery
    video_path:     str
    start_sec:      float
    end_sec:        float
    caption:        str
    score:          float


# ── Script → scene queries ────────────────────────────────────────────────────

def _expand_sentence(sentence: str) -> str:
    """
    Heuristic expansion: convert first-person narration into visual descriptors.
    e.g. "We reached Ujjain before sunrise" → "train arriving station city sunrise dawn"
    A proper LLM expansion would be better but this avoids an extra API call.
    """
    # Remove filler words that confuse search
    fillers = r"\b(we|i|our|us|my|the|a|an|is|was|were|are|to|of|and|in|at|on|by|with|for)\b"
    cleaned = re.sub(fillers, " ", sentence.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def script_to_queries(script_text: str) -> list[SceneQuery]:
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", script_text) if s.strip()]
    queries = []
    for sent in sentences:
        expanded = _expand_sentence(sent)
        words = [w for w in expanded.split() if len(w) > 3]
        queries.append(SceneQuery(
            sentence=sent,
            keywords=words,
            combined=f"{sent}. {expanded}",
        ))
    return queries


# ── Vector search ─────────────────────────────────────────────────────────────

def load_index():
    meta_path  = config.INDEX_DIR / "segments.json"
    faiss_path = config.INDEX_DIR / "index.faiss"
    if not meta_path.exists() or not faiss_path.exists():
        raise FileNotFoundError("No index found. Run `python indexer.py` first.")

    with open(meta_path) as f:
        segments = json.load(f)
    index = faiss.read_index(str(faiss_path))
    return segments, index


def search(
    queries: list[SceneQuery],
    top_k: int = 3,
    avoid_repeats: bool = True,
) -> list[list[Match]]:
    """Return top_k matches per query. Each match is a different segment."""
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(config.EMBED_MODEL)

    segments, index = load_index()

    used_keys: set[str] = set()   # "video:start" already assigned
    results: list[list[Match]] = []

    for q in queries:
        vec = embedder.encode(q.combined, normalize_embeddings=True).astype(np.float32)
        scores, idxs = index.search(vec.reshape(1, -1), top_k * 5)   # over-fetch to allow dedup

        matches: list[Match] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            seg = segments[idx]
            key = f"{seg['video_path']}:{seg['start_sec']}"
            if avoid_repeats and key in used_keys:
                continue
            matches.append(Match(
                scene_query=q,
                video_path=seg["video_path"],
                start_sec=seg["start_sec"],
                end_sec=seg["end_sec"],
                caption=seg["caption"],
                score=float(score),
            ))
            used_keys.add(key)
            if len(matches) >= top_k:
                break

        results.append(matches)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_results(queries: list[SceneQuery], results: list[list[Match]]):
    for q, matches in zip(queries, results):
        console.rule(f"[bold yellow]{q.sentence}[/bold yellow]")
        t = Table(show_header=True)
        t.add_column("Score", style="green", width=6)
        t.add_column("File", style="cyan")
        t.add_column("Time", width=12)
        t.add_column("Caption")
        for m in matches:
            t.add_row(
                f"{m.score:.3f}",
                m.video_path,
                f"{m.start_sec:.1f}s–{m.end_sec:.1f}s",
                m.caption,
            )
        console.print(t)


@app.command()
def main(
    script: Path = typer.Option(None, "--script", "-s", help="Plain-text script file"),
    query:  str  = typer.Option(None, "--query",  "-q", help="Single search query"),
    top_k:  int  = typer.Option(3,    "--top-k",  "-k"),
):
    if script:
        text = script.read_text(encoding="utf-8")
    elif query:
        text = query
    else:
        typer.echo("Provide --script or --query"); raise typer.Exit(1)

    queries = script_to_queries(text)
    results = search(queries, top_k=top_k)
    _print_results(queries, results)


if __name__ == "__main__":
    app()
