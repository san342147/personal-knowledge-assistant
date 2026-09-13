"""
Application configuration.

Loads settings from environment variables and optional .env file.
Supports xAI / Grok and any OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: personal-knowledge-assistant/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable among *names*."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # LLM (OpenAI-compatible)
    api_key: str
    base_url: str
    llm_model: str

    # Embeddings (local sentence-transformers)
    embedding_model: str

    # Chunking / retrieval
    chunk_size: int
    chunk_overlap: int
    top_k: int

    # Paths
    project_root: Path
    upload_dir: Path
    vector_dir: Path

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key)


def get_settings() -> Settings:
    """Build settings from environment (re-read each call so .env edits apply after restart)."""
    upload = PROJECT_ROOT / "data" / "uploads"
    vector = PROJECT_ROOT / "vectorstore"
    upload.mkdir(parents=True, exist_ok=True)
    vector.mkdir(parents=True, exist_ok=True)

    return Settings(
        # Prefer Groq, then xAI / Grok aliases, then generic OpenAI-compatible key
        api_key=_first_env(
            "GROQ_API_KEY",
            "XAI_API_KEY",
            "GROK_API_KEY",
            "OPENAI_API_KEY",
        ),
        base_url=_first_env(
            "OPENAI_BASE_URL",
            "GROQ_BASE_URL",
            "XAI_BASE_URL",
            default="https://api.x.ai/v1",
        ).rstrip("/"),
        llm_model=_first_env("LLM_MODEL", "XAI_MODEL", "GROQ_MODEL", default="grok-4.5"),
        embedding_model=_first_env(
            "EMBEDDING_MODEL",
            default="sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=int(_first_env("CHUNK_SIZE", default="1000")),
        chunk_overlap=int(_first_env("CHUNK_OVERLAP", default="200")),
        top_k=int(_first_env("TOP_K", default="4")),
        project_root=PROJECT_ROOT,
        upload_dir=upload,
        vector_dir=vector,
    )
