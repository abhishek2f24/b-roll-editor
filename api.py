"""
api.py — optional FastAPI web interface so you can use the system from a browser
         or call it from any frontend (React, Android, etc.).

    python api.py                    # starts on http://localhost:8000
    python api.py --port 9000

Endpoints:
    POST /index              — trigger indexing
    POST /run                — full pipeline, returns JSON timeline + output path
    POST /search             — search only, returns match list
    GET  /status             — index stats
    GET  /output/{filename}  — download rendered video
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import config

app = FastAPI(title="AI B-roll Editor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index_running  = False
_render_running = False


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    script:  str           # plain-text script
    top_k:   int = 3
    output:  str = "reel"

class SearchRequest(BaseModel):
    script: str
    top_k:  int = 3


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    meta = config.INDEX_DIR / "segments.json"
    if not meta.exists():
        return {"indexed": False, "segments": 0, "videos": 0}
    with open(meta) as f:
        segs = json.load(f)
    hashes_path = config.INDEX_DIR / "hashes.json"
    n_videos = len(json.load(open(hashes_path))) if hashes_path.exists() else 0
    return {"indexed": True, "segments": len(segs), "videos": n_videos}


@app.post("/index")
async def trigger_index(
    background_tasks: BackgroundTasks,
    reindex: bool = False,
):
    global _index_running
    if _index_running:
        return {"status": "already_running"}
    _index_running = True

    def _run():
        global _index_running
        try:
            import indexer
            indexer.index_videos(reindex=reindex)
        finally:
            _index_running = False

    background_tasks.add_task(_run)
    return {"status": "started"}


@app.post("/search")
async def do_search(req: SearchRequest):
    import searcher
    queries = searcher.script_to_queries(req.script)
    results = searcher.search(queries, top_k=req.top_k)
    out = []
    for q, matches in zip(queries, results):
        out.append({
            "sentence": q.sentence,
            "matches": [
                {
                    "video": m.video_path,
                    "start": m.start_sec,
                    "end":   m.end_sec,
                    "caption": m.caption,
                    "score":   round(m.score, 4),
                }
                for m in matches
            ],
        })
    return {"scenes": out}


@app.post("/run")
async def full_run(req: RunRequest, background_tasks: BackgroundTasks):
    global _render_running
    if _render_running:
        raise HTTPException(409, "A render is already in progress")

    result_holder: dict = {}
    _render_running = True

    def _run():
        global _render_running
        try:
            import searcher, assembler
            queries  = searcher.script_to_queries(req.script)
            results  = searcher.search(queries, top_k=req.top_k)
            plan     = assembler.matches_to_plan(results)
            out_path = assembler.assemble(plan, output_name=req.output)
            result_holder["path"] = str(out_path)
            result_holder["scenes"] = len(plan)
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            _render_running = False

    # Run synchronously (blocking) in background task
    # For production use run_in_executor with ProcessPoolExecutor
    background_tasks.add_task(_run)
    return {"status": "started", "output_name": req.output}


@app.get("/render_status")
async def render_status():
    return {"running": _render_running}


@app.get("/output/{filename}")
async def download(filename: str):
    path = config.OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a raw video into the videos/ folder."""
    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.VIDEOS_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"saved": file.filename, "path": str(dest)}


if __name__ == "__main__":
    import typer
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting API on http://localhost:{port}")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
