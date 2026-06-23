# AI B-roll Editor

Fully local, automated travel vlog assembler.

## How it works

```
videos/ folder
      ↓
Florence-2 captions every 3s frame
      ↓
FAISS vector index (.index/)
      ↓
Paste script → sentence embeddings
      ↓
Top-k matching segments per sentence
      ↓
FFmpeg cuts + cross-fades
      ↓
output/reel.mp4  ← record voice-over on top
```

## Setup

```powershell
.\setup.ps1
```

Needs: Python 3.11+, FFmpeg (`winget install ffmpeg`)

## Usage

### Full pipeline (index + search + render)
```powershell
.\venv\Scripts\python.exe agent.py run --script example_script.txt
```

### Step by step
```powershell
# 1. Index your videos (only needed once, incremental after that)
.\venv\Scripts\python.exe agent.py index

# 2. Preview matches without rendering
.\venv\Scripts\python.exe agent.py search --script example_script.txt

# 3. Export editable clip plan (JSON) then render
.\venv\Scripts\python.exe agent.py export-plan --script example_script.txt --out plan.json
# edit plan.json manually if you want
.\venv\Scripts\python.exe assembler.py --plan plan.json --output my_reel

# 4. Full run
.\venv\Scripts\python.exe agent.py run --script example_script.txt --output ujjain_reel
```

### Web API (for React / Android frontend)
```powershell
.\venv\Scripts\python.exe api.py
# → http://localhost:8000/docs  (Swagger UI)
```

API endpoints:
- `GET  /status`           — index stats
- `POST /upload_video`     — add a video to the library
- `POST /index`            — trigger indexing (background)
- `POST /search`           — find scenes for a script
- `POST /run`              — full pipeline (background)
- `GET  /output/{file}`    — download rendered video

## Config

Edit `config.py`:

| Key | Default | Notes |
|-----|---------|-------|
| `CAPTION_MODEL` | `Florence-2-large` | Swap to `Qwen2.5-VL-7B` for better captions (needs 16GB VRAM) |
| `FRAME_INTERVAL_SEC` | `3` | Lower = more granular index, slower |
| `DEFAULT_CLIP_DURATION` | `5` | Seconds per scene |
| `DEVICE` | `cuda` | Falls back to `cpu` automatically |

## Hardware requirements

| Setup | RAM | VRAM | Speed |
|-------|-----|------|-------|
| CPU only | 16GB | — | ~2 min/video |
| RTX 3060 (Florence-2-base) | 16GB | 6GB | ~15s/video |
| RTX 4070 (Florence-2-large) | 32GB | 12GB | ~8s/video |
| RTX 4090 (Qwen2.5-VL-7B) | 32GB | 24GB | ~5s/video |

Index once, query forever.
