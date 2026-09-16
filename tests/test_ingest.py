import io

import pytest
from pypdf import PdfWriter

from app.ingest import parse_document, recursive_spans


def test_chunk_offsets_overlap_and_coverage():
    text = ("Alpha beta gamma. Delta epsilon.\n\n" * 30) + "LAST"
    spans = recursive_spans(text, 120, 25)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(0 < end-start <= 120 for start, end in spans)
    assert all(b[0] == a[1]-25 for a, b in zip(spans, spans[1:]))
    covered = {i for a, b in spans for i in range(a, b)}
    assert covered == set(range(len(text)))


def test_bad_overlap():
    with pytest.raises(ValueError):
        recursive_spans("test", 10, 10)


def test_empty_and_invalid_pdf():
    assert not parse_document("empty.txt", b"  ").chunks
    bad = parse_document("broken.pdf", b"not a PDF")
    assert bad.warnings and not bad.chunks


def test_scanned_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    buffer = io.BytesIO()
    writer.write(buffer)
    result = parse_document("scan.pdf", buffer.getvalue())
    assert not result.chunks
    assert "OCR" in result.warnings[0]


def test_text_identity_metadata_and_sanitized_filename():
    first = parse_document("C:\\private\\notes.md", b"Useful evidence " * 40, 150, 20)
    again = parse_document("notes.md", b"Useful evidence " * 40, 150, 20)
    assert first.chunks == again.chunks
    assert all(c.filename == "notes.md" and c.page == 1 for c in first.chunks)
    assert len({c.id for c in first.chunks}) == len(first.chunks)


def test_document_size_limit():
    assert not parse_document("large.txt", b"hello", max_chars=3).chunks
