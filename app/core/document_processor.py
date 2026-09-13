"""
Document loading and chunking.

Supports PDF and plain-text uploads. Returns LangChain Document objects
with source metadata for citation in the UI.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".text"}


class DocumentProcessingError(Exception):
    """Raised when a file cannot be loaded or chunked."""


def _safe_filename(name: str) -> str:
    """Sanitize uploaded filenames for safe local storage."""
    name = Path(name).name
    name = re.sub(r"[^\w.\- ()\[\]]+", "_", name)
    return name.strip() or "upload.bin"


def save_uploads(files: Sequence, upload_dir: Path) -> List[Path]:
    """
    Persist Streamlit UploadedFile objects to disk.

    Returns paths of saved files. Skips empty files.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for f in files:
        try:
            raw_name = getattr(f, "name", "upload.bin")
            filename = _safe_filename(raw_name)
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                logger.warning("Skipping unsupported file type: %s", filename)
                continue

            dest = upload_dir / filename
            # Avoid overwrite collisions
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                i = 1
                while dest.exists():
                    dest = upload_dir / f"{stem}_{i}{suffix}"
                    i += 1

            data = f.getvalue() if hasattr(f, "getvalue") else f.read()
            if not data:
                logger.warning("Empty file skipped: %s", filename)
                continue

            dest.write_bytes(data)
            saved.append(dest)
            logger.info("Saved upload: %s (%d bytes)", dest.name, len(data))
        except Exception as exc:
            logger.exception("Failed to save upload")
            raise DocumentProcessingError(f"Failed to save '{getattr(f, 'name', '?')}': {exc}") from exc

    return saved


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load PDF/TXT files into LangChain Documents with source metadata."""
    docs: List[Document] = []

    for path in paths:
        path = Path(path)
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {path}")

        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                docs.extend(_load_pdf(path))
            elif ext in {".txt", ".md", ".text"}:
                docs.extend(_load_text(path))
            else:
                raise DocumentProcessingError(f"Unsupported file type: {ext}")
        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.exception("Error loading %s", path)
            raise DocumentProcessingError(f"Could not read '{path.name}': {exc}") from exc

    if not docs:
        raise DocumentProcessingError("No text could be extracted from the uploaded files.")

    return docs


def _load_pdf(path: Path) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    # Normalize metadata for UI citations
    for i, page in enumerate(pages):
        page.metadata = {
            **(page.metadata or {}),
            "source": path.name,
            "file_path": str(path),
            "page": page.metadata.get("page", i),
            "file_type": "pdf",
        }
        if page.page_content:
            page.page_content = page.page_content.strip()
    return [p for p in pages if p.page_content]


def _load_text(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page": 0,
                "file_type": path.suffix.lstrip(".").lower() or "txt",
            },
        )
    ]


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks for embedding."""
    if chunk_size < 100:
        raise DocumentProcessingError("chunk_size must be at least 100")
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Ensure every chunk keeps a readable source label
    for idx, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        meta.setdefault("source", "unknown")
        meta["chunk_id"] = idx
        chunk.metadata = meta

    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def process_files(
    files: Sequence,
    upload_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    End-to-end: save uploads → load → chunk.

    Returns ready-to-embed Document chunks.
    """
    paths = save_uploads(files, upload_dir)
    if not paths:
        raise DocumentProcessingError(
            "No valid PDF/TXT files were uploaded. Allowed: .pdf, .txt, .md"
        )
    docs = load_documents(paths)
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def clear_upload_dir(upload_dir: Path) -> None:
    """Delete all files inside the upload directory (keeps the folder)."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    for item in upload_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", item, exc)
