"""
LLM client factory — OpenAI-compatible chat models (xAI Grok by default).
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the chat model cannot be initialized or called."""


def get_chat_model(settings: Settings, temperature: float = 0.2):
    """
    Create a LangChain chat model pointed at an OpenAI-compatible API.

    Defaults to xAI: base_url=https://api.x.ai/v1, model=grok-4.5
    """
    if not settings.api_key:
        raise LLMError(
            "No API key found. In the project folder, edit .env and set "
            "GROQ_API_KEY=... (or XAI_API_KEY / OPENAI_API_KEY)."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMError(
            "langchain-openai is required. Run: pip install -r requirements.txt"
        ) from exc

    logger.info(
        "Initializing chat model model=%s base_url=%s",
        settings.llm_model,
        settings.base_url,
    )

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
        max_tokens=2048,
        timeout=120,
        max_retries=2,
    )


def probe_llm(settings: Settings) -> tuple[bool, str]:
    """
    Lightweight connectivity check. Returns (ok, message).
    """
    try:
        llm = get_chat_model(settings, temperature=0)
        # Minimal invoke — some providers reject empty content
        msg = llm.invoke("Reply with exactly: ok")
        text = getattr(msg, "content", str(msg))
        return True, f"Connected to {settings.llm_model} @ {settings.base_url} ({text[:80]})"
    except Exception as exc:
        return False, str(exc)
