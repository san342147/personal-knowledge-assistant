"""GroundedDesk: local retrieval, cited answers, measurable grounding."""
import os
from pathlib import Path

# Keep downloadable model weights alongside the ignored project caches for all entrypoints.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
