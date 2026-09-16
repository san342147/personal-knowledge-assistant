from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import api
from app.config import Settings
from app.models import Hit


class MemoryRetriever:
    def __init__(self, settings):
        self.chunks = {}

    def add(self, chunks):
        old = len(self.chunks)
        self.chunks.update({c.id: c for c in chunks})
        return len(self.chunks)-old

    def search(self, question, top_k):
        return [Hit(c, 5) for c in list(self.chunks.values())[:top_k]]

    def health(self):
        return {"status": "ok", "chunks": len(self.chunks)}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: Settings(_env_file=None, groq_api_key=""))
    monkeypatch.setattr(api, "Retriever", MemoryRetriever)
    monkeypatch.setattr(api, "Telemetry", lambda: SimpleNamespace(record=lambda event: None,
                                                                snapshot=lambda: {"requests": 0}))
    with TestClient(api.app) as client:
        yield client


def test_ingest_idempotent_and_invalid_document(client):
    for expected in (1, 0):
        response = client.post("/ingest", files={"file": ("note.txt", b"Loan duration is seven days.")})
        assert response.status_code == 200 and response.json()["chunks_added"] == expected
    response = client.post("/ingest", files={"file": ("bad.pdf", b"broken")})
    assert response.status_code == 422 and response.json()["warnings"]


def test_missing_key_health_and_ask(client):
    assert client.get("/health").status_code == 503
    client.post("/ingest", files={"file": ("note.md", b"Loan duration is seven days.")})
    response = client.post("/ask", json={"question": "loan duration?"})
    assert response.status_code == 503
    assert response.json()["reason"] == "generation_unavailable"


def test_empty_store_and_input_validation(client):
    response = client.post("/ask", json={"question": "salary?"})
    assert response.status_code == 200 and response.json()["refused"]
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "q", "top_k": 11}).status_code == 422
    assert client.get("/metrics").status_code == 200
