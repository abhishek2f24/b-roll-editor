"""
assembler.py — takes a list of Match objects (from searcher.py) and renders
               a final video with cross-fades using FFmpeg.

Usage (via agent.py):
    timeline = assemble(matches_per_scene, output_name="reel")

Direct CLI test:
    python assembler.py --plan plan.json --output reel.mp4
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

import config

console = Console()
app = typer.Typer(add_completion=False)


@dataclass
class ClipSpec:
    video_path: str     # relative to VIDEOS_DIR
    start_sec:  float
    duration:   float
    label:      str     # human-readable (the script sentence)


def matches_to_plan(matches_per_scene: list[list]) -> list[ClipSpec]:
    """Pick the best match per scene and build a ClipSpec list."""
    plan = []
    for matches in matches_per_scene:
        if not matches:
            continue
        best = matches[0]   # highest score
        dur = best.end_sec - best.start_sec
        if dur < 1.0:
            dur = config.DEFAULT_CLIP_DURATION
        plan.append(ClipSpec(
            video_path=best.video_path,
            start_sec=best.start_sec,
            duration=dur,
            label=best.scene_query.sentence,
        ))
    return plan


def _trim_clip(src: Path, start: float, duration: float, dst: Path):
    """Cut a clip with FFmpeg (no re-encode for speed, then re-encode for concat)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-vf", f"scale={config.OUTPUT_RESOLUTION[0]}:{config.OUTPUT_RESOLUTION[1]}:force_original_aspect_ratio=decrease,"
               f"pad={config.OUTPUT_RESOLUTION[0]}:{config.OUTPUT_RESOLUTION[1]}:(ow-iw)/2:(oh-ih)/2",
        "-r", str(config.OUTPUT_FPS),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-ar", "44100", "-ac", "2",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]FFmpeg error:[/red] {result.stderr[-500:]}")
        raise RuntimeError(f"FFmpeg failed for {src}")


def _concat_with_fade(clips: list[Path], output: Path, fade_dur: float):
    """
    Concatenate clips with cross-fades using FFmpeg filter_complex.
    Falls back to simple concat if there's only one clip.
    """
    if len(clips) == 1:
        clips[0].rename(output)
        return

    # Build a filter_complex xfade chain
    n = len(clips)
    inputs = " ".join(f"-i {str(c)}" for c in clips)

    # Compute cumulative offsets
    # We need durations for offset math
    durations = []
    for c in clips:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(c)],
            capture_output=True, text=True,
        )
        durations.append(float(result.stdout.strip() or 0))

    # Build xfade filter chain
    filter_parts = []
    offset = durations[0] - fade_dur
    prev_v = "[0:v]"
    prev_a = "[0:a]"

    for i in range(1, n):
        out_v = f"[v{i}]" if i < n - 1 else "[vout]"
        out_a = f"[a{i}]" if i < n - 1 else "[aout]"
        filter_parts.append(
            f"{prev_v}[{i}:v]xfade=transition=fade:duration={fade_dur}:offset={offset:.3f}{out_v}"
        )
        filter_parts.append(
            f"{prev_a}[{i}:a]acrossfade=d={fade_dur}{out_a}"
        )
        prev_v = out_v
        prev_a = out_a
        if i < n - 1:
            offset += durations[i] - fade_dur

    filter_complex = ";".join(filter_parts)

    input_args = []
    for c in clips:
        input_args += ["-i", str(c)]

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Fallback: simple concat without transitions
        console.print("[yellow]xfade failed, falling back to simple concat[/yellow]")
        _simple_concat(clips, output)


def _simple_concat(clips: list[Path], output: Path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for c in clips:
            f.write(f"file '{c.resolve()}'\n")
        list_file = Path(f.name)

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy", str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    list_file.unlink()


def assemble(plan: list[ClipSpec], output_name: str = "reel") -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = config.OUTPUT_DIR / f"{output_name}.mp4"

    console.print(f"\n[bold]Assembling {len(plan)} scenes → [cyan]{output}[/cyan][/bold]")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        clip_paths = []

        for i, spec in enumerate(plan):
            src = config.VIDEOS_DIR / spec.video_path
            dst = tmp / f"clip_{i:03d}.mp4"
            console.log(f"  [{i+1}/{len(plan)}] {spec.label[:60]}")
            _trim_clip(src, spec.start_sec, spec.duration, dst)
            clip_paths.append(dst)

        _concat_with_fade(clip_paths, output, config.TRANSITION_DURATION)

    console.print(f"[bold green]Done! → {output}[/bold green]")
    return output


# ── CLI (accepts a JSON plan file for testing) ────────────────────────────────

@app.command()
def main(
    plan:   Path = typer.Option(..., "--plan",   help="JSON file with clip specs"),
    output: str  = typer.Option("reel", "--output", help="Output filename stem"),
):
    with open(plan) as f:
        data = json.load(f)
    specs = [ClipSpec(**d) for d in data]
    assemble(specs, output_name=output)


if __name__ == "__main__":
    app()
