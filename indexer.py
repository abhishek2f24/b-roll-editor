"""
indexer.py — watches VIDEOS_DIR, captions every N-second frame, stores
             embeddings in FAISS + a JSON sidecar for fast retrieval.

Usage:
    python indexer.py                  # index everything new
    python indexer.py --reindex        # nuke + rebuild from scratch
"""

import json
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import faiss
import typer
from tqdm import tqdm
from rich.console import Console
from rich import print as rprint

import config

console = Console()
app = typer.Typer(add_completion=False)

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """One captioned segment of a video file."""
    video_path: str     # relative to VIDEOS_DIR
    start_sec:  float
    end_sec:    float
    caption:    str

# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Quick 8-char hash of the file — used to skip already-indexed files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:8]


def _load_captioner():
    """Load the caption model once and return a callable."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch

    console.log(f"[bold cyan]Loading caption model:[/] {config.CAPTION_MODEL}")
    device = config.DEVICE if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"

    if "Florence" in config.CAPTION_MODEL:
        processor = AutoProcessor.from_pretrained(config.CAPTION_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.CAPTION_MODEL, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        def caption(pil_images: list) -> list[str]:
            from PIL import Image
            task = "<MORE_DETAILED_CAPTION>"
            results = []
            for img in pil_images:
                inputs = processor(text=task, images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=128)
                out = processor.batch_decode(ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(out, task=task, image_size=img.size)
                results.append(parsed[task].strip())
            return results

    elif "Qwen" in config.CAPTION_MODEL:
        from transformers import Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        processor = AutoProcessor.from_pretrained(config.CAPTION_MODEL)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.CAPTION_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
        )

        def caption(pil_images: list) -> list[str]:
            results = []
            for img in pil_images:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text": "Describe this video frame in detail for a travel vlog context. One sentence."},
                    ],
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=100)
                trimmed = ids[:, inputs.input_ids.shape[1]:]
                results.append(processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip())
            return results

    else:
        raise ValueError(f"Unknown caption model: {config.CAPTION_MODEL}")

    return caption, device


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    console.log(f"[bold cyan]Loading embed model:[/] {config.EMBED_MODEL}")
    return SentenceTransformer(config.EMBED_MODEL)


def _extract_frames(video_path: Path, interval: float) -> list[tuple[float, object]]:
    """Return [(timestamp_sec, PIL.Image), ...] sampled every `interval` seconds."""
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total / fps

    frames = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((t, Image.fromarray(rgb)))
        t += interval

    cap.release()
    return frames


# ── Main indexing logic ───────────────────────────────────────────────────────

def index_videos(reindex: bool = False):
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path  = config.INDEX_DIR / "segments.json"
    faiss_path = config.INDEX_DIR / "index.faiss"
    hashes_path= config.INDEX_DIR / "hashes.json"

    # Load existing state
    segments: list[Segment] = []
    embeddings: list[np.ndarray] = []
    known_hashes: dict[str, str] = {}  # filename → hash

    if not reindex and meta_path.exists() and faiss_path.exists():
        with open(meta_path) as f:
            segments = [Segment(**s) for s in json.load(f)]
        with open(hashes_path) as f:
            known_hashes = json.load(f)
        idx = faiss.read_index(str(faiss_path))
        # Reconstruct embeddings list from index (for appending)
        if idx.ntotal > 0:
            embeddings = [idx.reconstruct(i) for i in range(idx.ntotal)]
        console.log(f"Loaded existing index: {len(segments)} segments")

    # Discover new videos
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    all_videos = [p for p in config.VIDEOS_DIR.rglob("*") if p.suffix.lower() in video_exts]

    new_videos = []
    for vp in all_videos:
        rel = str(vp.relative_to(config.VIDEOS_DIR))
        h = _file_hash(vp)
        if reindex or known_hashes.get(rel) != h:
            new_videos.append(vp)
            known_hashes[rel] = h

    if not new_videos:
        console.print("[green]Index is up to date. Nothing to do.[/green]")
        return

    console.print(f"[yellow]Indexing {len(new_videos)} new/changed video(s)...[/yellow]")

    caption_fn, device = _load_captioner()
    embedder = _load_embedder()

    for vp in tqdm(new_videos, desc="Videos"):
        rel = str(vp.relative_to(config.VIDEOS_DIR))
        console.log(f"  Processing [cyan]{rel}[/cyan]")

        frames = _extract_frames(vp, config.FRAME_INTERVAL_SEC)
        if not frames:
            continue

        # Remove old segments for this file (in case of reindex of changed file)
        segments = [s for s in segments if s.video_path != rel]

        # Caption in batches
        batch_size = config.CAPTION_BATCH_SIZE
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            timestamps = [t for t, _ in batch]
            pil_imgs   = [img for _, img in batch]
            captions   = caption_fn(pil_imgs)

            for j, (ts, cap_text) in enumerate(zip(timestamps, captions)):
                seg = Segment(
                    video_path=rel,
                    start_sec=ts,
                    end_sec=min(ts + config.FRAME_INTERVAL_SEC, frames[-1][0] + config.FRAME_INTERVAL_SEC),
                    caption=cap_text,
                )
                emb = embedder.encode(cap_text, normalize_embeddings=True).astype(np.float32)
                segments.append(seg)
                embeddings.append(emb)

    # Build / rebuild FAISS index
    if embeddings:
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatIP(dim)  # inner-product = cosine on normalized vecs
        index.add(np.stack(embeddings))
        faiss.write_index(index, str(faiss_path))

    with open(meta_path, "w") as f:
        json.dump([asdict(s) for s in segments], f, indent=2)
    with open(hashes_path, "w") as f:
        json.dump(known_hashes, f, indent=2)

    console.print(f"[bold green]Done! Index now has {len(segments)} segments from {len(known_hashes)} video(s).[/bold green]")


@app.command()
def main(reindex: bool = typer.Option(False, "--reindex", help="Wipe and rebuild from scratch")):
    index_videos(reindex=reindex)


if __name__ == "__main__":
    app()
