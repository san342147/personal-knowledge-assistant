"""
RAG orchestration: retrieve relevant chunks → generate grounded answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import Settings
from app.core.llm import LLMError, get_chat_model
from app.core.vector_store import VectorStoreError, as_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise Personal Knowledge Assistant.
Answer the user's question using ONLY the provided context excerpts from their documents.
Rules:
- If the context is insufficient, say you cannot find the answer in the uploaded documents.
- Be clear and concise. Use bullet points when helpful.
- Do not invent facts that are not supported by the context.
- When relevant, mention which source file a fact comes from.
"""

HUMAN_PROMPT = """Context from knowledge base:
{context}

---
Question: {question}

Answer based on the context above:"""


@dataclass
class RAGResult:
    """Structured answer with supporting source documents."""

    answer: str
    sources: List[Document] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(No relevant passages retrieved.)"
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        page_str = f", page {page}" if page is not None and page != "" else ""
        parts.append(f"[{i}] Source: {src}{page_str}\n{d.page_content.strip()}")
    return "\n\n".join(parts)


def _unique_source_labels(docs: List[Document]) -> List[str]:
    seen = set()
    labels = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        label = f"{src}" + (f" (page {page})" if page is not None and str(page) != "" else "")
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


class RAGService:
    """
    High-level RAG service used by the Streamlit UI.
    """

    def __init__(self, settings: Settings, vector_store: Any = None):
        self.settings = settings
        self.vector_store = vector_store

    def set_store(self, store: Any) -> None:
        self.vector_store = store

    def ask(self, question: str, top_k: Optional[int] = None) -> RAGResult:
        """Retrieve + generate an answer for *question*."""
        question = (question or "").strip()
        if not question:
            return RAGResult(answer="", error="Please enter a question.")

        if self.vector_store is None:
            return RAGResult(
                answer="",
                error="Knowledge base is empty. Upload and process documents first.",
            )

        k = top_k if top_k is not None else self.settings.top_k

        try:
            retriever = as_retriever(self.vector_store, top_k=k)
            docs: List[Document] = retriever.invoke(question)
        except VectorStoreError as exc:
            return RAGResult(answer="", error=str(exc))
        except Exception as exc:
            logger.exception("Retrieval failed")
            return RAGResult(answer="", error=f"Retrieval failed: {exc}")

        if not docs:
            return RAGResult(
                answer=(
                    "I could not find relevant passages in your knowledge base "
                    "for that question. Try rephrasing or uploading more documents."
                ),
                sources=[],
            )

        try:
            llm = get_chat_model(self.settings)
        except LLMError as exc:
            return RAGResult(answer="", error=str(exc), sources=docs)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )
        chain = prompt | llm

        try:
            response = chain.invoke(
                {
                    "context": _format_docs(docs),
                    "question": question,
                }
            )
            answer = getattr(response, "content", str(response))
            if isinstance(answer, list):
                # Some providers return content blocks
                answer = " ".join(
                    block.get("text", str(block)) if isinstance(block, dict) else str(block)
                    for block in answer
                )
            return RAGResult(answer=str(answer).strip(), sources=docs)
        except Exception as exc:
            logger.exception("LLM generation failed")
            # Never echo raw API keys back into the UI
            msg = str(exc)
            for secret in filter(None, [self.settings.api_key]):
                if secret and secret in msg:
                    msg = msg.replace(secret, secret[:6] + "…" + secret[-4:])
            if "invalid api key" in msg.lower() or "unauthorized" in msg.lower():
                msg += (
                    " | The key in .env was rejected. Open the project folder, "
                    "edit .env, set GROQ_API_KEY=your_key (no quotes, no spaces), "
                    "save, then fully stop Streamlit and run start.bat again. "
                    "Get a new key at https://console.groq.com/keys if this one is old."
                )
            return RAGResult(
                answer="",
                error=(
                    f"LLM request failed: {msg} "
                    f"(using {self.settings.llm_model} @ {self.settings.base_url})."
                ),
                sources=docs,
            )

    @staticmethod
    def source_summary(docs: List[Document]) -> List[str]:
        return _unique_source_labels(docs)
