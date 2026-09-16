"""Page-local recursive splitting with exact source offsets."""
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.models import Chunk


@dataclass
class ParsedDocument:
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    document_id: str = ""
    filename: str = ""


def recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")

    def boundary(start: int, stop: int, separators: tuple[str, ...]) -> int:
        if not separators:
            return stop
        sep, *rest = separators
        pos = text.rfind(sep, start + size // 2, stop)
        return pos + len(sep) if pos >= 0 else boundary(start, stop, tuple(rest))

    spans = []
    start = 0
    while start < len(text):
        stop = min(start + size, len(text))
        end = boundary(start, stop, ("\n\n", "\n", ". ", " ")) if stop < len(text) else stop
        if text[start:end].strip():
            spans.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return spans


def parse_document(filename: str, content: bytes, size=900, overlap=150,
                   max_chars=2_000_000) -> ParsedDocument:
    # Treat both Windows and POSIX client paths as untrusted display names.
    filename = re.sub(r"[\x00-\x1f]", "", filename.replace("\\", "/").split("/")[-1])[:180]
    result = ParsedDocument(filename=filename)
    result.document_id = hashlib.sha256(filename.encode() + b"\0" + content).hexdigest()
    suffix = Path(filename).suffix.lower()
    pages = []
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                result.warnings.append("Encrypted PDF: export an unlocked copy first.")
                return result
            if len(reader.pages) > 500:
                result.warnings.append("PDF exceeds the 500-page local demo limit.")
                return result
            chars = 0
            for number, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    result.warnings.append(f"Page {number}: text extraction failed.")
                    continue
                chars += len(text)
                if chars > max_chars:
                    result.warnings.append("Document exceeds the extracted-text limit.")
                    return result
                if not text.strip():
                    result.warnings.append(f"Page {number}: no text; scanned pages need OCR.")
                else:
                    pages.append((number, text))
        except Exception:
            result.warnings.append("PDF could not be read. It may be damaged or unsupported.")
            return result
    elif suffix in {".txt", ".md"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            result.warnings.append("Text files must use UTF-8 encoding.")
            return result
        if len(text) > max_chars:
            result.warnings.append("Document exceeds the extracted-text limit.")
            return result
        pages = [(1, text)]
    else:
        result.warnings.append("Only PDF, TXT, and MD files are supported.")
        return result
    for number, text in pages:
        for start, end in recursive_spans(text, size, overlap):
            identity = f"{result.document_id}:{number}:{start}:{end}:{size}:{overlap}"
            result.chunks.append(Chunk(hashlib.sha256(identity.encode()).hexdigest()[:24],
                                       text[start:end], filename, number,
                                       result.document_id, start, end))
    if not result.chunks:
        result.warnings.append("No readable text found; nothing was indexed.")
    return result
