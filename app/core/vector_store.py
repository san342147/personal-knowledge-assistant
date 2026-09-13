"""
Vector store management with ChromaDB + local sentence-transformer embeddings.

Embeddings run entirely on-device (no embedding API key required).
"""

from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "personal_knowledge"


class VectorStoreError(Exception):
    """Raised when the vector database cannot be created or queried."""


@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> Embeddings:
    """
    Load (and cache) a HuggingFace sentence-transformer embedding model.

    First call downloads the model weights (~80MB for MiniLM).
    """
    # Prefer langchain-huggingface; fall back to community package.
    HuggingFaceEmbeddings = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        except ImportError as exc:
            raise VectorStoreError(
                "Install embedding support: pip install langchain-huggingface "
                "sentence-transformers"
            ) from exc

    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: List[Document],
    persist_directory: Path,
    embedding_model: str,
) -> "Chroma":
    """
    Create a fresh Chroma vector store from document chunks and persist to disk.
    """
    if not chunks:
        raise VectorStoreError("No chunks provided to build the knowledge base.")

    persist_directory = Path(persist_directory)
    # Wipe previous collection directory for a clean rebuild
    clear_vector_store(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    try:
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise VectorStoreError(
                "Chroma integration missing. Install langchain-chroma and chromadb."
            ) from exc

    embeddings = get_embeddings(embedding_model)

    try:
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name=COLLECTION_NAME,
        )
        # Older chromadb/langchain combos need explicit persist
        if hasattr(store, "persist"):
            store.persist()
        logger.info(
            "Vector store built: %d chunks → %s",
            len(chunks),
            persist_directory,
        )
        return store
    except Exception as exc:
        logger.exception("Failed to build vector store")
        raise VectorStoreError(f"Failed to create vector store: {exc}") from exc


def load_vector_store(
    persist_directory: Path,
    embedding_model: str,
) -> Optional["Chroma"]:
    """
    Open an existing on-disk Chroma store, or return None if empty/missing.
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.exists() or not any(persist_directory.iterdir()):
        return None

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings(embedding_model)

    try:
        store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        # Empty collection check
        try:
            count = store._collection.count()  # noqa: SLF001 — practical emptiness check
            if count == 0:
                return None
        except Exception:
            pass
        return store
    except Exception as exc:
        logger.warning("Could not load vector store: %s", exc)
        return None


def clear_vector_store(persist_directory: Path) -> None:
    """Remove all persisted vector data."""
    persist_directory = Path(persist_directory)
    if persist_directory.exists():
        shutil.rmtree(persist_directory, ignore_errors=True)
    persist_directory.mkdir(parents=True, exist_ok=True)
    # Keep folder in git
    gitkeep = persist_directory / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.write_text("", encoding="utf-8")
    # Clear embedding cache so model can be reloaded cleanly if needed
    get_embeddings.cache_clear()
    logger.info("Vector store cleared: %s", persist_directory)


def as_retriever(store, top_k: int = 4):
    """Return a LangChain retriever for similarity search."""
    if store is None:
        raise VectorStoreError("Knowledge base is empty. Upload documents first.")
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max(1, int(top_k))},
    )
