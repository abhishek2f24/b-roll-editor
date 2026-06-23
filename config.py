"""Central config — edit these paths and settings."""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
VIDEOS_DIR  = ROOT / "videos"           # drop your raw footage here
INDEX_DIR   = ROOT / ".index"           # auto-created; stores FAISS + metadata
OUTPUT_DIR  = ROOT / "output"           # rendered timelines land here

# ── Indexing ─────────────────────────────────────────────────────────────────
FRAME_INTERVAL_SEC  = 3        # sample one frame every N seconds
CAPTION_BATCH_SIZE  = 4        # frames per model forward pass (lower if OOM)
DEVICE              = "cuda"   # "cuda" | "cpu"  (cuda is ~10x faster)

# ── Caption model (local, free) ───────────────────────────────────────────────
# Options ranked by quality: "Qwen/Qwen2.5-VL-7B-Instruct" (best, needs 16GB VRAM)
#                             "microsoft/Florence-2-large"   (great, 6GB VRAM)
#                             "microsoft/Florence-2-base"    (good, 3GB VRAM, fast)
CAPTION_MODEL = "microsoft/Florence-2-large"

# ── Embedding model (sentence-transformers, CPU-friendly) ─────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Assembly ─────────────────────────────────────────────────────────────────
DEFAULT_CLIP_DURATION   = 5     # seconds per scene if no match is long enough
TRANSITION_DURATION     = 0.3   # cross-fade seconds
OUTPUT_FPS              = 30
OUTPUT_RESOLUTION       = (1920, 1080)   # (width, height)
