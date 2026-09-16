from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    groq_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_backend: Literal["sentence", "cross_encoder"] = "sentence"
    chunk_size: int = Field(900, ge=100, le=4000)
    chunk_overlap: int = Field(150, ge=0)
    top_k: int = Field(4, ge=1, le=10)
    candidate_k: int = Field(20, ge=10, le=100)
    rerank_threshold: float = 0.5
    enable_rewrite: bool = True
    vectorstore_path: Path = ROOT / "vectorstore"
    input_usd_per_million: float = Field(0.075, ge=0)
    output_usd_per_million: float = Field(0.30, ge=0)
    max_upload_bytes: int = 20 * 1024 * 1024
    max_document_chars: int = 2_000_000

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not self.vectorstore_path.is_absolute():
            self.vectorstore_path = ROOT / self.vectorstore_path
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
