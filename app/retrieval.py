import hashlib
import re
import threading
from dataclasses import asdict

import numpy as np
from rank_bm25 import BM25Okapi

from app.models import Chunk, Hit


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], limit: int, constant=60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), 1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (constant + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))[:limit]


class Retriever:
    """One process owns the local collection; lock protects writes and BM25 snapshots."""

    def __init__(self, settings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.lock = threading.RLock()
        self.client = chromadb.PersistentClient(path=str(settings.vectorstore_path),
                                               settings=ChromaSettings(anonymized_telemetry=False))
        fingerprint = hashlib.sha256(
            f"{settings.embedding_model}:{settings.chunk_size}:{settings.chunk_overlap}".encode()
        ).hexdigest()[:12]
        self.collection = self.client.get_or_create_collection(
            f"groundeddesk-{fingerprint}", embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}})
        self._embedder = None
        self._reranker = None
        self._refresh()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            try:
                # Avoid slow Hub metadata retries when the model is already cached.
                self._embedder = SentenceTransformer(
                    self.settings.embedding_model, device="cpu", local_files_only=True)
            except OSError:
                # First run: allow the library to download the public model normally.
                self._embedder = SentenceTransformer(self.settings.embedding_model, device="cpu")
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            if self.settings.reranker_backend == "sentence":
                from app.reranking import SentenceReranker
                self._reranker = SentenceReranker(self.embedder)
            else:
                import torch
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self.settings.reranker_model, device="cpu",
                                              activation_fn=torch.nn.Identity())
        return self._reranker

    def _refresh(self):
        data = self.collection.get(include=["documents", "metadatas"])
        self.chunks = {i: Chunk(id=i, text=t, **m) for i, t, m in
                       zip(data["ids"], data["documents"], data["metadatas"])}
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i].text) or [""] for i in self.ids]) if self.ids else None

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        with self.lock:
            fresh = [c for c in chunks if c.id not in self.chunks]
            for offset in range(0, len(fresh), 128):
                batch = fresh[offset:offset + 128]
                vectors = self.embedder.encode([c.text for c in batch], normalize_embeddings=True).tolist()
                self.collection.upsert(ids=[c.id for c in batch], documents=[c.text for c in batch],
                    embeddings=vectors, metadatas=[{k: v for k, v in asdict(c).items()
                                                   if k not in {"id", "text"}} for c in batch])
            self._refresh()
            return len(fresh)

    def search(self, query: str, top_k: int, stage="reranked") -> list[Hit]:
        with self.lock:
            if not self.ids:
                return []
            n = min(max(top_k, self.settings.candidate_k), len(self.ids))
            vector = self.embedder.encode([query], normalize_embeddings=True).tolist()
            dense = self.collection.query(query_embeddings=vector, n_results=n)["ids"][0]
            if stage == "dense":
                return [Hit(self.chunks[i]) for i in dense[:top_k]]
            scores = self.bm25.get_scores(tokenize(query))
            keyword = [self.ids[j] for j in np.argsort(-scores, kind="stable")[:n] if scores[j] > 0]
            merged = reciprocal_rank_fusion([dense, keyword], n)
            if stage == "hybrid":
                return [Hit(self.chunks[i]) for i in merged[:top_k]]
            scores = self.reranker.predict([(query, self.chunks[i].text) for i in merged])
            hits = [Hit(self.chunks[i], float(s)) for i, s in zip(merged, scores)]
            return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:top_k]

    def health(self):
        with self.lock:
            self.client.heartbeat()
            return {"status": "ok", "chunks": self.collection.count(), "collection": self.collection.name}
