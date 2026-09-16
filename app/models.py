from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    id: str
    text: str
    filename: str
    page: int
    document_id: str
    start: int = 0
    end: int = 0


@dataclass
class Hit:
    chunk: Chunk
    score: float = 0.0


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Citation(BaseModel):
    chunk_id: str
    filename: str
    page: int
    snippet: str


class Answer(BaseModel):
    request_id: str
    answer: str
    refused: bool
    reason: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    latency_ms: float = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0
